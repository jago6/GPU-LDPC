#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// BG1最大尺寸定义 (根据5G NR标准)
#define BG1_MAX_INFO_BITS 8448    // 最大信息比特数
#define BG1_MAX_CODE_BITS 25344   // 最大码字长度 (8448 * 3)
#define BG1_LIFTING_SIZE 384      // 最大提升因子
#define BG1_BASE_ROWS 46          // 基础校验矩阵行数
#define BG1_BASE_COLS 68          // 基础校验矩阵列数

// LLR量化参数
#define LLR_BITS 8                // 8bit量化
#define LLR_MAX_VALUE 127         // 8bit有符号数最大值
#define LLR_MIN_VALUE -128        // 8bit有符号数最小值

// LLR数据结构
struct LLR_Data {
    int8_t* llr_values;           // LLR值数组
    int length;                   // 数组长度
    int lifting_size;             // 当前提升因子
    float snr_db;                 // 信噪比(dB)
    float noise_variance;         // 噪声方差
};

// GPU全局内存指针
int8_t* d_llr_data = NULL;
int* d_llr_length = NULL;
float* d_snr_info = NULL;

// 生成高斯随机数 (Box-Muller方法)
float generate_gaussian(float mean, float variance) {
    static int has_spare = 0;
    static float spare;
    
    if (has_spare) {
        has_spare = 0;
        return spare * sqrt(variance) + mean;
    }
    
    has_spare = 1;
    static float u, v, mag;
    do {
        u = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        v = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        mag = u * u + v * v;
    } while (mag >= 1.0f || mag == 0.0f);
    
    mag = sqrt(-2.0f * log(mag) / mag);
    spare = v * mag;
    return u * mag * sqrt(variance) + mean;
}

// 8bit量化函数
int8_t quantize_llr(float llr_value) {
    // 限制在合理范围内
    if (llr_value > 10.0f) llr_value = 10.0f;
    if (llr_value < -10.0f) llr_value = -10.0f;
    
    // 量化到8bit范围
    float scaled = llr_value * 12.8f; // 将[-10,10]映射到[-128,127]
    
    int quantized = (int)round(scaled);
    if (quantized > LLR_MAX_VALUE) quantized = LLR_MAX_VALUE;
    if (quantized < LLR_MIN_VALUE) quantized = LLR_MIN_VALUE;
    
    return (int8_t)quantized;
}

// 生成随机LLR数据
struct LLR_Data generate_random_llr_data(int lifting_size, float snr_db) {
    struct LLR_Data llr_data;
    
    // 计算实际码字长度
    llr_data.length = BG1_BASE_COLS * lifting_size;
    llr_data.lifting_size = lifting_size;
    llr_data.snr_db = snr_db;
    
    // 计算噪声方差 (假设码率为1/3)
    float code_rate = 1.0f / 3.0f;
    float snr_linear = pow(10.0f, snr_db / 10.0f);
    llr_data.noise_variance = 1.0f / (2.0f * code_rate * snr_linear);
    
    printf("生成LLR数据参数:\n");
    printf("  码字长度: %d\n", llr_data.length);
    printf("  提升因子: %d\n", lifting_size);
    printf("  信噪比: %.2f dB\n", snr_db);
    printf("  噪声方差: %.6f\n", llr_data.noise_variance);
    
    // 分配内存
    llr_data.llr_values = (int8_t*)malloc(llr_data.length * sizeof(int8_t));
    if (!llr_data.llr_values) {
        printf("错误: 内存分配失败\n");
        llr_data.length = 0;
        return llr_data;
    }
    
    // 生成随机LLR值
    printf("正在生成随机LLR数据...\n");
    
    for (int i = 0; i < llr_data.length; i++) {
        // 随机生成发送比特 (0或1)
        int sent_bit = rand() % 2;
        float sent_symbol = sent_bit ? -1.0f : 1.0f; // BPSK调制
        
        // 添加高斯噪声
        float received_symbol = sent_symbol + generate_gaussian(0.0f, llr_data.noise_variance);
        
        // 计算LLR = 2 * received_symbol / noise_variance
        float llr_value = 2.0f * received_symbol / llr_data.noise_variance;
        
        // 8bit量化
        llr_data.llr_values[i] = quantize_llr(llr_value);
    }
    
    return llr_data;
}

// 分配GPU全局内存
cudaError_t allocate_gpu_memory(int max_length) {
    cudaError_t err;
    
    // 分配LLR数据内存
    err = cudaMalloc((void**)&d_llr_data, max_length * sizeof(int8_t));
    if (err != cudaSuccess) {
        printf("GPU内存分配失败 (LLR数据): %s\n", cudaGetErrorString(err));
        return err;
    }
    
    // 分配长度信息内存
    err = cudaMalloc((void**)&d_llr_length, sizeof(int));
    if (err != cudaSuccess) {
        printf("GPU内存分配失败 (长度): %s\n", cudaGetErrorString(err));
        return err;
    }
    
    // 分配SNR信息内存
    err = cudaMalloc((void**)&d_snr_info, 2 * sizeof(float)); // snr_db + noise_variance
    if (err != cudaSuccess) {
        printf("GPU内存分配失败 (SNR信息): %s\n", cudaGetErrorString(err));
        return err;
    }
    
    printf("GPU内存分配成功:\n");
    printf("  LLR数据: %d 字节\n", max_length * (int)sizeof(int8_t));
    printf("  控制信息: %d 字节\n", (int)(sizeof(int) + 2 * sizeof(float)));
    
    return cudaSuccess;
}

// 传输数据到GPU
cudaError_t transfer_llr_to_gpu(const struct LLR_Data* llr_data) {
    cudaError_t err;
    
    printf("开始传输数据到GPU...\n");
    
    // 传输LLR数据
    err = cudaMemcpy(d_llr_data, llr_data->llr_values, 
                     llr_data->length * sizeof(int8_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("LLR数据传输失败: %s\n", cudaGetErrorString(err));
        return err;
    }
    
    // 传输长度信息
    err = cudaMemcpy(d_llr_length, &llr_data->length, 
                     sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("长度信息传输失败: %s\n", cudaGetErrorString(err));
        return err;
    }
    
    // 传输SNR信息
    float snr_info[2] = {llr_data->snr_db, llr_data->noise_variance};
    err = cudaMemcpy(d_snr_info, snr_info, 
                     2 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("SNR信息传输失败: %s\n", cudaGetErrorString(err));
        return err;
    }
    
    printf("数据传输完成！\n");
    return cudaSuccess;
}

// GPU验证内核
__global__ void verify_llr_data_kernel(int8_t* llr_data, int* length, float* snr_info) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        printf("\n=== GPU数据验证 ===\n");
        printf("LLR数据长度: %d\n", *length);
        printf("SNR: %.2f dB, 噪声方差: %.6f\n", snr_info[0], snr_info[1]);
        
        // 统计LLR值分布
        int pos_count = 0, neg_count = 0, zero_count = 0;
        int max_val = -128, min_val = 127;
        
        for (int i = 0; i < *length && i < 10000; i++) { // 限制检查范围避免超时
            if (llr_data[i] > 0) pos_count++;
            else if (llr_data[i] < 0) neg_count++;
            else zero_count++;
            
            if (llr_data[i] > max_val) max_val = llr_data[i];
            if (llr_data[i] < min_val) min_val = llr_data[i];
        }
        
        printf("LLR值统计 (前10000个):\n");
        printf("  正值: %d, 负值: %d, 零值: %d\n", pos_count, neg_count, zero_count);
        printf("  最大值: %d, 最小值: %d\n", max_val, min_val);
        
        // 显示前20个LLR值
        printf("前20个LLR值: [");
        for (int i = 0; i < 20 && i < *length; i++) {
            printf("%d%s", llr_data[i], (i < 19 && i < *length-1) ? ", " : "");
        }
        printf("]\n");
    }
}

// 性能测试内核
__global__ void performance_test_kernel(int8_t* llr_data, int* length, int* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    int local_sum = 0;
    for (int i = tid; i < *length; i += stride) {
        // 模拟简单的LLR处理操作
        local_sum += abs(llr_data[i]);
    }
    
    // 使用原子操作累加结果
    atomicAdd(result, local_sum);
}

// 清理GPU内存
void cleanup_gpu_memory() {
    if (d_llr_data) {
        cudaFree(d_llr_data);
        d_llr_data = NULL;
    }
    if (d_llr_length) {
        cudaFree(d_llr_length);
        d_llr_length = NULL;
    }
    if (d_snr_info) {
        cudaFree(d_snr_info);
        d_snr_info = NULL;
    }
    printf("GPU内存清理完成\n");
}

// 打印LLR数据统计信息
void print_llr_statistics(const struct LLR_Data* llr_data) {
    printf("\n=== LLR数据统计 ===\n");
    
    int pos_count = 0, neg_count = 0, zero_count = 0;
    int max_val = -128, min_val = 127;
    long long sum = 0;
    
    for (int i = 0; i < llr_data->length; i++) {
        int8_t val = llr_data->llr_values[i];
        
        if (val > 0) pos_count++;
        else if (val < 0) neg_count++;
        else zero_count++;
        
        if (val > max_val) max_val = val;
        if (val < min_val) min_val = val;
        sum += val;
    }
    
    float average = (float)sum / llr_data->length;
    
    printf("统计信息:\n");
    printf("  总长度: %d\n", llr_data->length);
    printf("  正值数量: %d (%.2f%%)\n", pos_count, 100.0f * pos_count / llr_data->length);
    printf("  负值数量: %d (%.2f%%)\n", neg_count, 100.0f * neg_count / llr_data->length);
    printf("  零值数量: %d (%.2f%%)\n", zero_count, 100.0f * zero_count / llr_data->length);
    printf("  数值范围: [%d, %d]\n", min_val, max_val);
    printf("  平均值: %.3f\n", average);
    
    // 显示前50个值作为样本
    printf("\n前50个LLR值:\n");
    for (int i = 0; i < 50 && i < llr_data->length; i++) {
        printf("%4d", llr_data->llr_values[i]);
        if ((i + 1) % 10 == 0) printf("\n");
    }
    if (llr_data->length > 0 && (llr_data->length >= 50 ? 50 : llr_data->length) % 10 != 0) {
        printf("\n");
    }
}

int main() {
    printf("=== LDPC BG1 LLR数据生成与GPU传输程序 ===\n\n");
    
    // 初始化随机数种子
    srand((unsigned int)time(NULL));
    
    // 设置参数 - 使用BG1最大配置
    int lifting_size = BG1_LIFTING_SIZE;  // 384
    float snr_db = 2.0f;                  // 2dB信噪比
    
    // 计算最大长度
    int max_length = BG1_BASE_COLS * BG1_LIFTING_SIZE;
    printf("BG1最大配置:\n");
    printf("  基础矩阵: %d x %d\n", BG1_BASE_ROWS, BG1_BASE_COLS);
    printf("  最大提升因子: %d\n", BG1_LIFTING_SIZE);
    printf("  最大码字长度: %d\n", max_length);
    printf("  最大信息长度: %d\n", BG1_MAX_INFO_BITS);
    printf("  内存需求: %.2f MB\n\n", max_length * sizeof(int8_t) / (1024.0f * 1024.0f));
    
    // 分配GPU内存
    cudaError_t err = allocate_gpu_memory(max_length);
    if (err != cudaSuccess) {
        return -1;
    }
    
    // 生成随机LLR数据
    struct LLR_Data llr_data = generate_random_llr_data(lifting_size, snr_db);
    if (llr_data.length == 0) {
        cleanup_gpu_memory();
        return -1;
    }
    
    // 打印统计信息
    print_llr_statistics(&llr_data);
    
    // 传输到GPU
    err = transfer_llr_to_gpu(&llr_data);
    if (err != cudaSuccess) {
        free(llr_data.llr_values);
        cleanup_gpu_memory();
        return -1;
    }
    
    // GPU验证
    printf("\n开始GPU验证...\n");
    verify_llr_data_kernel<<<1, 1>>>(d_llr_data, d_llr_length, d_snr_info);
    cudaDeviceSynchronize();
    
    // 性能测试
    printf("\n开始GPU性能测试...\n");
    int *d_result;
    cudaMalloc((void**)&d_result, sizeof(int));
    cudaMemset(d_result, 0, sizeof(int));
    
    // 记录时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // 启动性能测试内核
    int threads = 256;
    int blocks = (llr_data.length + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // 限制块数
    
    performance_test_kernel<<<blocks, threads>>>(d_llr_data, d_llr_length, d_result);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    // 获取结果
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("性能测试结果:\n");
    printf("  处理时间: %.3f ms\n", elapsed_time);
    printf("  吞吐量: %.2f MB/s\n", 
           (llr_data.length * sizeof(int8_t)) / (elapsed_time * 1000.0f));
    printf("  累加结果: %d\n", result);
    
    // 清理内存
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(llr_data.llr_values);
    cleanup_gpu_memory();
    
    printf("\n程序执行完成！\n");
    return 0;
}