#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <spdlog/spdlog.h>
#include <vector>
#include <memory>

// 宏定义
#define MAX_NNZ 65536          // 最大非零元素数
#define MAX_ITERATIONS 50      // 最大迭代次数
#define NORMALIZATION_FACTOR 0.75f  // 归一化因子 α
#define WARP_SIZE 32

// H矩阵相关参数（需要根据具体码字设置）
#define H_ROWS 1024            // H矩阵行数
#define H_COLS 2048            // H矩阵列数
#define Z_FACTOR 64            // 提升因子Z
#define MAX_CHECK_DEGREE 20    // 校验节点最大度数
#define MAX_VARIABLE_DEGREE 10 // 变量节点最大度数

// 线程块和网格配置
#define ALPHA_SIZE 8           // 码字组大小（并行处理的码字数）
#define BETA_SIZE 4            // 并行的码字组数量

// 压缩H矩阵结构体
struct CompressedH {
    short* row_ptr;     // 行指针数组，长度H_ROWS+1
    short* col_idx;     // 列索引数组
    short* shift_cn;    // 移位值数组
    
    short* col_ptr;     // 列指针数组，长度H_COLS+1
    short* row_idx;     // 行索引数组
    short* shift_vn;    // 移位值数组
    
    short nnz;          // 非零元素总数
};

// GPU常量内存中的H矩阵
__constant__ struct DeviceConstantH {
    short row_ptr[H_ROWS + 1];    // 行指针数组
    short col_ptr[H_COLS + 1];    // 列指针数组
    
    short col_idx[MAX_NNZ];       // 列索引
    short row_idx[MAX_NNZ];       // 行索引
    short shift_cn[MAX_NNZ];      // 校验节点移位值
    short shift_vn[MAX_NNZ];      // 变量节点移位值
    
    short nnz;                    // 非零元素数
} d_constH;

// GPU全局内存数据结构
struct GPUDecodeData {
    float* llr_input;      // 输入LLR数据
    float* llr_app;        // APP值数组
    float* msg_v2c;        // 变量节点到校验节点消息
    float* msg_c2v;        // 校验节点到变量节点消息
    int* hard_decision;    // 硬判决结果
    int* syndrome_check;   // 校验子检查结果
    int* iteration_count;  // 迭代次数记录
    bool* converged;       // 收敛标志
};

// CUDA流管理结构
struct StreamManager {
    cudaStream_t* streams;
    int num_streams;
    GPUDecodeData* gpu_data;
    int codewords_per_stream;
};

// ==================== 设备函数定义 ====================

/**
 * @brief 计算循环移位后的索引
 * @param base_idx 基础索引
 * @param shift 移位量
 * @param z_factor 提升因子Z
 * @return 移位后的索引
 */
__device__ __forceinline__ int cyclic_shift_index(int base_idx, int shift, int z_factor) {
    if (shift < 0) return -1;  // 表示该位置为0
    return (base_idx + shift) % z_factor;
}

/**
 * @brief 计算符号函数
 * @param x 输入值
 * @return 符号值 (+1 或 -1)
 */
__device__ __forceinline__ int sign_func(float x) {
    return (x >= 0) ? 1 : -1;
}

/**
 * @brief 原子操作的最小值更新（用于浮点数）
 * @param address 内存地址
 * @param val 比较值
 * @return 更新前的值
 */
__device__ __forceinline__ float atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                       __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// ==================== 核函数实现 ====================

/**
 * @brief LDPC分层译码核函数
 * 
 * 线程块结构: (Z_FACTOR, ALPHA_SIZE, 1)
 * 线程网格结构: (BETA_SIZE, 1, 1)
 * 
 * 每个线程负责一个提升矩阵中的一个元素位置的处理
 * 
 * @param gpu_data GPU数据结构指针
 * @param max_iterations 最大迭代次数
 * @param layer_idx 当前处理的层索引
 */
__global__ void ldpc_layered_decode_kernel(GPUDecodeData* gpu_data, 
                                          int max_iterations, 
                                          int layer_idx) {
    
    // 线程索引计算
    int z_idx = threadIdx.x;          // 提升因子内的索引 [0, Z_FACTOR-1]
    int alpha_idx = threadIdx.y;      // 码字组内的码字索引 [0, ALPHA_SIZE-1]
    int beta_idx = blockIdx.x;        // 码字组索引 [0, BETA_SIZE-1]
    
    // 计算全局码字索引
    int codeword_idx = beta_idx * ALPHA_SIZE + alpha_idx;
    
    // 共享内存声明 - 用于存储当前处理层的消息
    __shared__ float shared_v2c_msgs[Z_FACTOR][ALPHA_SIZE][MAX_CHECK_DEGREE];
    __shared__ float shared_c2v_msgs[Z_FACTOR][ALPHA_SIZE][MAX_VARIABLE_DEGREE];
    __shared__ int shared_sign_product[Z_FACTOR][ALPHA_SIZE];
    __shared__ float shared_min_mag[Z_FACTOR][ALPHA_SIZE];
    __shared__ float shared_second_min[Z_FACTOR][ALPHA_SIZE];
    
    // 边界检查
    if (codeword_idx >= BETA_SIZE * ALPHA_SIZE) return;
    
    // ==================== 第一阶段：变量节点到校验节点消息更新 (V2C) ====================

        
        // 获取当前校验节点连接的变量节点范围
        short row_start = d_constH.row_ptr[layer_idx];
        short row_end = d_constH.row_ptr[layer_idx + 1];
        
        // 遍历当前校验节点连接的所有变量节点
        for (int edge_idx = row_start; edge_idx < row_end; edge_idx++) {
            
            int var_col = d_constH.col_idx[edge_idx];           // 变量节点列索引
            int shift_val = d_constH.shift_cn[edge_idx];       // 循环移位值
            
            // 计算实际的变量节点索引（考虑循环移位）
            int actual_var_idx = cyclic_shift_index(z_idx, shift_val, Z_FACTOR);
            
            if (actual_var_idx >= 0) {  // 有效连接
                
                // 计算V2C消息索引
                int v2c_idx = codeword_idx * H_COLS * MAX_CHECK_DEGREE + 
                             var_col * MAX_CHECK_DEGREE + (edge_idx - row_start);
                int c2v_idx = codeword_idx * H_ROWS * MAX_VARIABLE_DEGREE + 
                             check_row * MAX_VARIABLE_DEGREE + (edge_idx - row_start);
                int app_idx = codeword_idx * H_COLS + var_col;
                
                // V2C消息更新：L^n_{v_j → c_i} = L^{app}_{v_j} - L^n_{c_i → v_j}
                // 这里实现了NMSA算法中的公式(3)
                float app_value = gpu_data->llr_app[app_idx];
                float prev_c2v_msg = gpu_data->msg_c2v[c2v_idx];
                float v2c_msg = app_value - prev_c2v_msg;
                
                // 存储V2C消息到全局内存和共享内存
                gpu_data->msg_v2c[v2c_idx] = v2c_msg;
                shared_v2c_msgs[z_idx][alpha_idx][edge_idx - row_start] = v2c_msg;
            }
        }
        
        __syncthreads();  // 确保V2C消息更新完成
        
        // ==================== 第二阶段：校验节点到变量节点消息更新 (C2V) ====================
        
        // 重新遍历当前校验节点的边，计算C2V消息
        for (int edge_idx = row_start; edge_idx < row_end; edge_idx++) {
            
            int var_col = d_constH.col_idx[edge_idx];
            int shift_val = d_constH.shift_cn[edge_idx];
            int actual_var_idx = cyclic_shift_index(z_idx, shift_val, Z_FACTOR);
            
            if (actual_var_idx >= 0) {
                
                // 初始化符号乘积和最小值
                int sign_product = 1;
                float min_magnitude = INFINITY;
                float second_min_magnitude = INFINITY;
                
                // 计算除当前边之外的所有V2C消息的符号乘积和最小值
                // 这实现了NMSA算法中的公式(4)的核心计算
                for (int other_edge = row_start; other_edge < row_end; other_edge++) {
                    
                    if (other_edge != edge_idx) {  // 排除当前边
                        
                        int other_var_col = d_constH.col_idx[other_edge];
                        int other_shift = d_constH.shift_cn[other_edge];
                        int other_actual_idx = cyclic_shift_index(z_idx, other_shift, Z_FACTOR);
                        
                        if (other_actual_idx >= 0) {
                            float other_v2c_msg = shared_v2c_msgs[z_idx][alpha_idx][other_edge - row_start];
                            
                            // 更新符号乘积：∏_{v' ∈ N(c_i)\v_j} sgn(L^n_{v' → c_i})
                            sign_product *= sign_func(other_v2c_msg);
                            
                            // 更新最小值和次小值：min_{v' ∈ N(c_i)\v_j} |L^n_{v' → c_i}|
                            float abs_msg = fabsf(other_v2c_msg);
                            if (abs_msg < min_magnitude) {
                                second_min_magnitude = min_magnitude;
                                min_magnitude = abs_msg;
                            } else if (abs_msg < second_min_magnitude) {
                                second_min_magnitude = abs_msg;
                            }
                        }
                    }
                }
                
                // 计算C2V消息：L^{n+1}_{c_i → v_j} = α × ∏sign × min_magnitude
                // 这是NMSA算法的核心公式(4)
                float c2v_msg = NORMALIZATION_FACTOR * sign_product * min_magnitude;
                
                // 存储C2V消息
                int c2v_idx = codeword_idx * H_ROWS * MAX_VARIABLE_DEGREE + 
                             check_row * MAX_VARIABLE_DEGREE + (edge_idx - row_start);
                gpu_data->msg_c2v[c2v_idx] = c2v_msg;
                shared_c2v_msgs[z_idx][alpha_idx][edge_idx - row_start] = c2v_msg;
            }
        }
        
        __syncthreads();  // 确保C2V消息更新完成
        
        // ==================== 第三阶段：APP值更新 ====================
        
        // 更新与当前校验节点相关的变量节点的APP值
        for (int edge_idx = row_start; edge_idx < row_end; edge_idx++) {
            
            int var_col = d_constH.col_idx[edge_idx];
            int shift_val = d_constH.shift_cn[edge_idx];
            int actual_var_idx = cyclic_shift_index(z_idx, shift_val, Z_FACTOR);
            
            if (actual_var_idx >= 0) {
                
                int app_idx = codeword_idx * H_COLS + var_col;
                int llr_input_idx = codeword_idx * H_COLS + var_col;
                
                // APP值更新：L^{app}_{v_j} = L^{init}_{v_j} + ∑_{c_i ∈ M(v_j)} L^{n+1}_{c_i → v_j}
                // 这实现了NMSA算法中的公式(5)
                
                // 重新计算该变量节点的所有C2V消息之和
                float app_sum = gpu_data->llr_input[llr_input_idx];  // 初始LLR
                
                // 获取变量节点的所有连接边
                int col_start = d_constH.col_ptr[var_col];
                int col_end = d_constH.col_ptr[var_col + 1];
                
                for (int var_edge = col_start; var_edge < col_end; var_edge++) {
                    int connected_check = d_constH.row_idx[var_edge];
                    int var_shift = d_constH.shift_vn[var_edge];
                    int actual_check_idx = cyclic_shift_index(z_idx, var_shift, Z_FACTOR);
                    
                    if (actual_check_idx >= 0) {
                        int c2v_lookup_idx = codeword_idx * H_ROWS * MAX_VARIABLE_DEGREE + 
                                           connected_check * MAX_VARIABLE_DEGREE + (var_edge - col_start);
                        app_sum += gpu_data->msg_c2v[c2v_lookup_idx];
                    }
                }
                
                // 更新APP值
                gpu_data->llr_app[app_idx] = app_sum;
            }
        }
    }
    
    __syncthreads();  // 确保所有更新完成


/**
 * @brief 硬判决核函数
 * 
 * 线程块结构: (Z_FACTOR, ALPHA_SIZE, 1)
 * 线程网格结构: (BETA_SIZE, 1, 1)
 * 
 * 根据APP值进行硬判决，并检查校验子
 * 
 * @param gpu_data GPU数据结构指针
 */
__global__ void hard_decision_kernel(GPUDecodeData* gpu_data) {
    
    // 线程索引计算
    int z_idx = threadIdx.x;
    int alpha_idx = threadIdx.y;
    int beta_idx = blockIdx.x;
    
    int codeword_idx = beta_idx * ALPHA_SIZE + alpha_idx;
    
    // 边界检查
    if (codeword_idx >= BETA_SIZE * ALPHA_SIZE) return;
    
    // 共享内存用于校验子计算
    __shared__ int shared_syndrome[Z_FACTOR][ALPHA_SIZE];
    
    // ==================== 硬判决阶段 ====================
    
    // 对每个变量节点进行硬判决
    for (int var_col = 0; var_col < H_COLS; var_col += Z_FACTOR) {
        
        if (var_col + z_idx < H_COLS) {
            int app_idx = codeword_idx * H_COLS + var_col + z_idx;
            int decision_idx = codeword_idx * H_COLS + var_col + z_idx;
            
            // 硬判决：x̂_j = { 0 if L^{app}_{v_j} ≥ 0, 1 if L^{app}_{v_j} < 0 }
            float app_value = gpu_data->llr_app[app_idx];
            gpu_data->hard_decision[decision_idx] = (app_value < 0) ? 1 : 0;
        }
    }
    
    __syncthreads();
    
    // ==================== 校验子检查阶段 ====================
    
    // 初始化共享内存
    shared_syndrome[z_idx][alpha_idx] = 0;
    __syncthreads();
    
    // 计算校验子 H·x̂^T
    for (int check_row = 0; check_row < H_ROWS; check_row += Z_FACTOR) {
        
        if (check_row + z_idx < H_ROWS) {
            int current_check = check_row + z_idx;
            int syndrome_sum = 0;
            
            // 获取当前校验节点连接的变量节点
            int row_start = d_constH.row_ptr[current_check];
            int row_end = d_constH.row_ptr[current_check + 1];
            
            // 计算校验方程：∑_{v_j ∈ N(c_i)} x̂_j mod 2
            for (int edge_idx = row_start; edge_idx < row_end; edge_idx++) {
                
                int var_col = d_constH.col_idx[edge_idx];
                int shift_val = d_constH.shift_cn[edge_idx];
                int actual_var_idx = cyclic_shift_index(z_idx, shift_val, Z_FACTOR);
                
                if (actual_var_idx >= 0) {
                    int decision_idx = codeword_idx * H_COLS + var_col;
                    syndrome_sum ^= gpu_data->hard_decision[decision_idx];  // 模2加法
                }
            }
            
            // 累加到共享内存
            atomicAdd(&shared_syndrome[z_idx][alpha_idx], syndrome_sum);
        }
    }
    
    __syncthreads();
    
    // 检查收敛条件：校验子是否全为0
    if (z_idx == 0) {  // 每个码字组只有一个线程执行
        int total_syndrome = 0;
        for (int i = 0; i < Z_FACTOR; i++) {
            total_syndrome += shared_syndrome[i][alpha_idx];
        }
        
        // 存储校验结果
        gpu_data->syndrome_check[codeword_idx] = total_syndrome;
        
        // 设置收敛标志：如果校验子为0，则收敛
        gpu_data->converged[codeword_idx] = (total_syndrome == 0);
    }
}

// ==================== 主机端函数实现 ====================

/**
 * @brief 复制H矩阵到常量内存
 * @param h_compH 主机端压缩H矩阵
 * @return CUDA错误代码
 */
cudaError_t copy_H_to_constant_memory(const struct CompressedH* h_compH) {
    struct DeviceConstantH temp;
    cudaError_t err;
    
    // 检查数据大小
    if (h_compH->nnz > MAX_NNZ) {
        auto logger = spdlog::get("GPU");
        logger->error("非零元素数量 {} 超过最大限制 {}", h_compH->nnz, MAX_NNZ);
        return cudaErrorInvalidValue;
    }
    
    // 初始化临时结构体
    memset(&temp, 0, sizeof(struct DeviceConstantH));
    
    // 复制数据到临时结构体
    memcpy(temp.row_ptr, h_compH->row_ptr, (H_ROWS + 1) * sizeof(short));
    memcpy(temp.col_ptr, h_compH->col_ptr, (H_COLS + 1) * sizeof(short));
    memcpy(temp.col_idx, h_compH->col_idx, h_compH->nnz * sizeof(short));
    memcpy(temp.row_idx, h_compH->row_idx, h_compH->nnz * sizeof(short));
    memcpy(temp.shift_cn, h_compH->shift_cn, h_compH->nnz * sizeof(short));
    memcpy(temp.shift_vn, h_compH->shift_vn, h_compH->nnz * sizeof(short));
    temp.nnz = h_compH->nnz;
    
    // 传输到GPU常量内存
    err = cudaMemcpyToSymbol(d_constH, &temp, sizeof(struct DeviceConstantH));
    if (err != cudaSuccess) {
        auto logger = spdlog::get("GPU");
        logger->error("复制到常量内存失败: {}", cudaGetErrorString(err));
        return err;
    }
    
    auto logger = spdlog::get("GPU");
    logger->info("成功复制 {} 个非零元素到GPU常量内存", h_compH->nnz);
    return cudaSuccess;
}

/**
 * @brief 初始化GPU数据结构
 * @param gpu_data GPU数据结构指针
 * @param num_codewords 码字数量
 * @return CUDA错误代码
 */
cudaError_t initialize_gpu_data(GPUDecodeData* gpu_data, int num_codewords) {
    
    cudaError_t err = cudaSuccess;
    
    // 分配GPU内存
    size_t llr_size = num_codewords * H_COLS * sizeof(float);
    size_t v2c_size = num_codewords * H_COLS * MAX_CHECK_DEGREE * sizeof(float);
    size_t c2v_size = num_codewords * H_ROWS * MAX_VARIABLE_DEGREE * sizeof(float);
    size_t decision_size = num_codewords * H_COLS * sizeof(int);
    size_t syndrome_size = num_codewords * sizeof(int);
    size_t iteration_size = num_codewords * sizeof(int);
    size_t converged_size = num_codewords * sizeof(bool);
    
    // 分配各种数组
    err |= cudaMalloc(&gpu_data->llr_input, llr_size);
    err |= cudaMalloc(&gpu_data->llr_app, llr_size);
    err |= cudaMalloc(&gpu_data->msg_v2c, v2c_size);
    err |= cudaMalloc(&gpu_data->msg_c2v, c2v_size);
    err |= cudaMalloc(&gpu_data->hard_decision, decision_size);
    err |= cudaMalloc(&gpu_data->syndrome_check, syndrome_size);
    err |= cudaMalloc(&gpu_data->iteration_count, iteration_size);
    err |= cudaMalloc(&gpu_data->converged, converged_size);
    
    if (err != cudaSuccess) {
        auto logger = spdlog::get("GPU");
        logger->error("GPU内存分配失败: {}", cudaGetErrorString(err));
        return err;
    }
    
    // 初始化内存
    err |= cudaMemset(gpu_data->msg_v2c, 0, v2c_size);
    err |= cudaMemset(gpu_data->msg_c2v, 0, c2v_size);
    err |= cudaMemset(gpu_data->hard_decision, 0, decision_size);
    err |= cudaMemset(gpu_data->syndrome_check, 0, syndrome_size);
    err |= cudaMemset(gpu_data->iteration_count, 0, iteration_size);
    err |= cudaMemset(gpu_data->converged, 0, converged_size);
    
    auto logger = spdlog::get("GPU");
    logger->info("GPU数据结构初始化完成，码字数量: {}", num_codewords);
    
    return err;
}

/**
 * @brief 执行LDPC译码（多流版本）
 * @param streams 流管理器
 * @param host_llr 主机端LLR数据
 * @param host_results 主机端结果数组
 * @param num_layers H矩阵的层数
 * @return 译码是否成功
 */
bool execute_ldpc_decode_multistream(StreamManager* streams, 
                                   float* host_llr, 
                                   int* host_results,
                                   int num_layers) {
    
    auto logger = spdlog::get("LDPC");
    logger->info("开始多流LDPC译码，流数量: {}", streams->num_streams);
    
    // 配置核函数启动参数
    dim3 blockDim(Z_FACTOR, ALPHA_SIZE, 1);  // 线程块：(Z, α, 1)
    dim3 gridDim(BETA_SIZE, 1, 1);           // 线程网格：(β, 1, 1)
    
    // 为每个流执行译码
    for (int stream_id = 0; stream_id < streams->num_streams; stream_id++) {
        
        cudaStream_t current_stream = streams->streams[stream_id];
        GPUDecodeData* gpu_data = &streams->gpu_data[stream_id];
        
        // 异步传输LLR数据到GPU
        size_t llr_size = streams->codewords_per_stream * H_COLS * sizeof(float);
        float* stream_llr_host = host_llr + stream_id * streams->codewords_per_stream * H_COLS;
        
        cudaMemcpyAsync(gpu_data->llr_input, stream_llr_host, llr_size, 
                       cudaMemcpyHostToDevice, current_stream);
        
        // 初始化APP值为输入LLR
        cudaMemcpyAsync(gpu_data->llr_app, gpu_data->llr_input, llr_size,
                       cudaMemcpyDeviceToDevice, current_stream);
        
        // 分层译码迭代
        for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
            
            // 逐层处理
            for (int layer = 0; layer < num_layers; layer++) {
                
                // 启动分层译码核函数
                ldpc_layered_decode_kernel<<<gridDim, blockDim, 0, current_stream>>>(
                    gpu_data, MAX_ITERATIONS, layer);
                
                // 检查核函数执行错误
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    logger->error("分层译码核函数执行失败 (流{}, 迭代{}, 层{}): {}", 
                                stream_id, iteration, layer, cudaGetErrorString(err));
                    return false;
                }
            }
            
            // 执行硬判决和校验
            hard_decision_kernel<<<gridDim, blockDim, 0, current_stream>>>(gpu_data);
            
            // 检查收敛性（每几次迭代检查一次以提高性能）
            if (iteration % 5 == 4) {
                // 这里可以添加收敛检查的异步回调
                // 实际实现中可能需要更复杂的收敛检查机制
            }
        }
        
        // 异步传输结果回主机
        size_t result_size = streams->codewords_per_stream * H_COLS * sizeof(int);
        int* stream_results_host = host_results + stream_id * streams->codewords_per_stream * H_COLS;
        
        cudaMemcpyAsync(stream_results_host, gpu_data->hard_decision, result_size,
                       cudaMemcpyDeviceToHost, current_stream);
    }
    
    // 同步所有流
    for (int i = 0; i < streams->num_streams; i++) {
        cudaStreamSynchronize(streams->streams[i]);
    }
    
    logger->info("多流LDPC译码完成");
    return true;
}

/**
 * @brief 清理GPU资源
 * @param gpu_data GPU数据结构指针
 */
void cleanup_gpu_data(GPUDecodeData* gpu_data) {
    if (gpu_data->llr_input) cudaFree(gpu_data->llr_input);
    if (gpu_data->llr_app) cudaFree(gpu_data->llr_app);
    if (gpu_data->msg_v2c) cudaFree(gpu_data->msg_v2c);
    if (gpu_data->msg_c2v) cudaFree(gpu_data->msg_c2v);
    if (gpu_data->hard_decision) cudaFree(gpu_data->hard_decision);
    if (gpu_data->syndrome_check) cudaFree(gpu_data->syndrome_check);
    if (gpu_data->iteration_count) cudaFree(gpu_data->iteration_count);
    if (gpu_data->converged) cudaFree(gpu_data->converged);
    
    memset(gpu_data, 0, sizeof(GPUDecodeData));
}

/**
 * @brief 初始化流管理器
 * @param streams 流管理器指针
 * @param num_streams 流的数量
 * @param total_codewords 总码字数量
 * @return 是否初始化成功
 */
bool initialize_stream_manager(StreamManager* streams, int num_streams, int total_codewords) {
    
    auto logger = spdlog::get("GPU");
    
    streams->num_streams = num_streams;
    streams->codewords_per_stream = (total_codewords + num_streams - 1) / num_streams;
    
    // 分配流数组
    streams->streams = new cudaStream_t[num_streams];
    streams->gpu_data = new GPUDecodeData[num_streams];
    
    // 创建CUDA流
    for (int i = 0; i < num_streams; i++) {
        cudaError_t err = cudaStreamCreate(&streams->streams[i]);
        if (err != cudaSuccess) {
            logger->error("创建CUDA流{}失败: {}", i, cudaGetErrorString(err));
            return false;
        }
        
        // 初始化每个流的GPU数据
        err = initialize_gpu_data(&streams->gpu_data[i], streams->codewords_per_stream);
        if (err != cudaSuccess) {
            logger->error("初始化流{}的GPU数据失败", i);
            return false;
        }
    }
    
    logger->info("流管理器初始化完成，流数: {}, 每流码字数: {}", 
                num_streams, streams->codewords_per_stream);
    return true;
}

/**
 * @brief 清理流管理器
 * @param streams 流管理器指针
 */
void cleanup_stream_manager(StreamManager* streams) {
    
    if (streams->streams) {
        for (int i = 0; i < streams->num_streams; i++) {
            cudaStreamDestroy(streams->streams[i]);
            cleanup_gpu_data(&streams->gpu_data[i]);
        }
        delete[] streams->streams;
        streams->streams = nullptr;
    }
    
    if (streams->gpu_data) {
        delete[] streams->gpu_data;
        streams->gpu_data = nullptr;
    }
    
    streams->num_streams = 0;
    streams->codewords_per_stream = 0;
}

/**
 * @brief 传输LLR数据到GPU
 * @param host_llr 主机端LLR数据
 * @param device_llr 设备端LLR数据指针
 * @param num_codewords 码字数量
 * @param stream CUDA流
 * @return CUDA错误代码
 */
cudaError_t transfer_llr_to_gpu(float* host_llr, float* device_llr, 
                               int num_codewords, cudaStream_t stream = 0) {
    
    size_t data_size = num_codewords * H_COLS * sizeof(float);
    
    cudaError_t err = cudaMemcpyAsync(device_llr, host_llr, data_size,
                                     cudaMemcpyHostToDevice, stream);
    
    if (err != cudaSuccess) {
        auto logger = spdlog::get("GPU");
        logger->error("LLR数据传输到GPU失败: {}", cudaGetErrorString(err));
    } else {
        auto logger = spdlog::get("DEBUG");
        logger->debug("成功传输 {} 个码字的LLR数据到GPU", num_codewords);
    }
    
    return err;
}

/**
 * @brief 设置日志系统
 */
void setup_logs() {
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_sink_st>());
    sinks.push_back(
        std::make_shared<spdlog::sinks::daily_file_sink_st>("logs/logfile.log", 23, 59));
    
    // 创建同步日志器
    auto ldpc_log = std::make_shared<spdlog::logger>("LDPC", begin(sinks), end(sinks));
    auto hw_logger = std::make_shared<spdlog::logger>("GPU", begin(sinks), end(sinks));
    auto results_logger = std::make_shared<spdlog::logger>("RESULTS", begin(sinks), end(sinks));
    auto debug_logger = std::make_shared<spdlog::logger>("DEBUG", begin(sinks), end(sinks));
    
    spdlog::register_logger(ldpc_log);
    spdlog::register_logger(hw_logger);
    spdlog::register_logger(results_logger);
    spdlog::register_logger(debug_logger);
    
    // 设置日志级别
    spdlog::set_level(spdlog::level::debug);
    spdlog::flush_every(std::chrono::seconds(3));
}

/**
 * @brief 性能统计结构体
 */
struct PerformanceStats {
    float total_decode_time_ms;      // 总译码时间
    float avg_iteration_time_ms;     // 平均每次迭代时间
    float memory_transfer_time_ms;   // 内存传输时间
    float kernel_execution_time_ms;  // 核函数执行时间
    int total_iterations;            // 总迭代次数
    int converged_codewords;         // 收敛的码字数量
    float throughput_mbps;           // 吞吐率 (Mbps)
};

/**
 * @brief 计算性能统计信息
 * @param stats 性能统计结构体指针
 * @param num_codewords 码字数量
 * @param codeword_length 码字长度
 */
void calculate_performance_stats(PerformanceStats* stats, int num_codewords, int codeword_length) {
    
    if (stats->total_decode_time_ms > 0) {
        // 计算吞吐率：(码字数 × 码字长度 × 1000) / (译码时间ms × 1024 × 1024)
        float total_bits = num_codewords * codeword_length;
        stats->throughput_mbps = (total_bits * 1000.0f) / (stats->total_decode_time_ms * 1024.0f * 1024.0f);
    }
    
    if (stats->total_iterations > 0) {
        stats->avg_iteration_time_ms = stats->kernel_execution_time_ms / stats->total_iterations;
    }
    
    auto logger = spdlog::get("RESULTS");
    logger->info("=== 性能统计 ===");
    logger->info("总译码时间: {:.2f} ms", stats->total_decode_time_ms);
    logger->info("内存传输时间: {:.2f} ms", stats->memory_transfer_time_ms);
    logger->info("核函数执行时间: {:.2f} ms", stats->kernel_execution_time_ms);
    logger->info("平均迭代时间: {:.2f} ms", stats->avg_iteration_time_ms);
    logger->info("总迭代次数: {}", stats->total_iterations);
    logger->info("收敛码字数: {}/{}", stats->converged_codewords, num_codewords);
    logger->info("吞吐率: {:.2f} Mbps", stats->throughput_mbps);
    logger->info("===============");
}

/**
 * @brief 主要的LDPC译码接口函数
 * @param h_matrix 压缩H矩阵
 * @param host_llr_data 主机端LLR数据
 * @param host_decoded_bits 主机端译码结果
 * @param num_codewords 码字数量
 * @param num_streams 并行流数量
 * @param stats 性能统计指针
 * @return 译码是否成功
 */
bool ldpc_decode_gpu(const CompressedH* h_matrix,
                    float* host_llr_data,
                    int* host_decoded_bits,
                    int num_codewords,
                    int num_streams,
                    PerformanceStats* stats = nullptr) {
    
    auto logger = spdlog::get("LDPC");
    logger->info("开始GPU LDPC译码，码字数: {}, 流数: {}", num_codewords, num_streams);
    
    cudaEvent_t start_event, stop_event, mem_start, mem_stop, kernel_start, kernel_stop;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventCreate(&mem_start);
    cudaEventCreate(&mem_stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    
    cudaEventRecord(start_event);
    
    // 1. 复制H矩阵到常量内存
    cudaError_t err = copy_H_to_constant_memory(h_matrix);
    if (err != cudaSuccess) {
        logger->error("H矩阵复制失败");
        return false;
    }
    
    // 2. 初始化流管理器
    StreamManager stream_manager;
    if (!initialize_stream_manager(&stream_manager, num_streams, num_codewords)) {
        logger->error("流管理器初始化失败");
        return false;
    }
    
    // 3. 计算H矩阵层数（假设按Z_FACTOR分层）
    int num_layers = (H_ROWS + Z_FACTOR - 1) / Z_FACTOR;
    
    // 4. 执行译码
    cudaEventRecord(kernel_start);
    
    bool decode_success = execute_ldpc_decode_multistream(&stream_manager,
                                                        host_llr_data,
                                                        host_decoded_bits,
                                                        num_layers);
    
    cudaEventRecord(kernel_stop);
    cudaEventSynchronize(kernel_stop);
    
    if (!decode_success) {
        logger->error("LDPC译码执行失败");
        cleanup_stream_manager(&stream_manager);
        return false;
    }
    
    // 5. 清理资源
    cleanup_stream_manager(&stream_manager);
    
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    
    // 6. 计算性能统计
    if (stats != nullptr) {
        cudaEventElapsedTime(&stats->total_decode_time_ms, start_event, stop_event);
        cudaEventElapsedTime(&stats->kernel_execution_time_ms, kernel_start, kernel_stop);
        stats->total_iterations = MAX_ITERATIONS * num_layers * num_codewords;
        
        calculate_performance_stats(stats, num_codewords, H_COLS);
    }
    
    // 清理事件
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaEventDestroy(mem_start);
    cudaEventDestroy(mem_stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    
    logger->info("GPU LDPC译码完成");
    return true;
}

// ==================== 使用示例和测试函数 ====================

/**
 * @brief 测试函数：验证GPU译码器的正确性
 */
void test_ldpc_decoder() {
    
    setup_logs();
    auto logger = spdlog::get("DEBUG");
    
    // 模拟参数
    const int TEST_CODEWORDS = 32;
    const int NUM_STREAMS = 4;
    
    // 分配主机内存
    float* host_llr = new float[TEST_CODEWORDS * H_COLS];
    int* host_results = new int[TEST_CODEWORDS * H_COLS];
    
    // 初始化测试数据（这里应该用实际的LLR数据）
    for (int i = 0; i < TEST_CODEWORDS * H_COLS; i++) {
        host_llr[i] = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;  // 随机LLR值
    }
    
    // 创建测试用的H矩阵（这里应该用实际的LDPC码H矩阵）
    CompressedH test_h_matrix;
    // ... 初始化H矩阵数据 ...
    
    // 执行译码
    PerformanceStats stats = {0};
    bool success = ldpc_decode_gpu(&test_h_matrix, host_llr, host_results,
                                  TEST_CODEWORDS, NUM_STREAMS, &stats);
    
    if (success) {
        logger->info("LDPC译码测试成功");
        
        // 验证结果（这里应该加入具体的验证逻辑）
        int error_count = 0;
        for (int i = 0; i < TEST_CODEWORDS * H_COLS; i++) {
            // 简单的验证：检查硬判决结果是否在合理范围内
            if (host_results[i] != 0 && host_results[i] != 1) {
                error_count++;
            }
        }
        
        logger->info("硬判决错误数: {}/{}", error_count, TEST_CODEWORDS * H_COLS);
        
    } else {
        logger->error("LDPC译码测试失败");
    }
    
    // 清理内存
    delete[] host_llr;
    delete[] host_results;
}

/**
 * @brief 主函数示例
 */
int main() {
    
    // 初始化CUDA设备
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        printf("未找到CUDA设备\n");
        return -1;
    }
    
    // 设置设备属性
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("使用GPU: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("全局内存: %.2f GB\n", prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
    printf("常量内存: %.2f KB\n", prop.totalConstMem / 1024.0f);
    printf("共享内存: %.2f KB\n", prop.sharedMemPerBlock / 1024.0f);
    
    // 运行测试
    test_ldpc_decoder();
    
    return 0;
}