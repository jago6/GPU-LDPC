
#ifndef ALLOCATE_GPU_MEMORY_H
#define ALLOCATE_GPU_MEMORY_H
#define MAX_NNZ 316 // 最大非零元素数，假设为316




#include <cuda_runtime.h>

// 在CUDA全局作用域定义常量内存结构
// 要不要放到这里？ 要不要内存对齐？
__constant__ struct DeviceConstantH {
    short row_ptr[H_ROWS + 1];    // 行指针数组
    short col_ptr[H_COLS + 1];    // 列指针数组
    
    // 压缩数据 - 使用固定最大尺寸
    short col_idx[MAX_NNZ];       // 列索引
    short row_idx[MAX_NNZ];       // 行索引
    short shift_cn[MAX_NNZ];      // 校验节点移位值
    short shift_vn[MAX_NNZ];      // 变量节点移位值
    
    short nnz         // 最大非零元素数
} d_constH;


cudaError_t copy_H_to_constant_memory(const struct CompressedH* h_compH) {
    struct DeviceConstantH temp;
    cudaError_t err;
    
    // 检查数据大小
    if (h_compH->nnz > MAX_NNZ) {
        printf("错误: 非零元素数量 %d 超过最大限制 %d\n", h_compH->nnz, MAX_NNZ);
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
        printf("复制到常量内存失败: %s\n", cudaGetErrorString(err));
        return err;
    }
    
    printf("成功复制 %d 个非零元素到GPU常量内存\n", h_compH->nnz);
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
    
    return cudaSuccess;
}


#endif