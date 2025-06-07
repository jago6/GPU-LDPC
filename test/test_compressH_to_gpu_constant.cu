#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../src/h_matrices/h_bg_1_i_0.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define  H_ROWS 46
#define  H_COLS 68
#define  MAX_NNZ 400

struct CompressedH {
    short* row_ptr;
    short* col_idx;
    short* shift_cn;
    short* col_ptr;
    short* row_idx;
    short* shift_vn;
    short nnz;
};

// GPU常量内存结构体
__constant__ struct DeviceConstantH {
    short row_ptr[H_ROWS + 1];
    short col_ptr[H_COLS + 1];
    short col_idx[MAX_NNZ];
    short row_idx[MAX_NNZ];
    short shift_cn[MAX_NNZ];
    short shift_vn[MAX_NNZ];
    short nnz;
} d_constH;

// 验证内核 - 将GPU数据复制回主机进行验证
__global__ void verify_constant_memory_kernel(short* d_row_ptr_out, short* d_col_ptr_out,
                                             short* d_col_idx_out, short* d_row_idx_out,
                                             short* d_shift_cn_out, short* d_shift_vn_out,
                                             short* d_nnz_out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // 复制行指针
    if (tid <= H_ROWS) {
        d_row_ptr_out[tid] = d_constH.row_ptr[tid];
    }
    
    // 复制列指针
    if (tid <= H_COLS) {
        d_col_ptr_out[tid] = d_constH.col_ptr[tid];
    }
    
    // 复制非零元素数据
    if (tid < d_constH.nnz) {
        d_col_idx_out[tid] = d_constH.col_idx[tid];
        d_row_idx_out[tid] = d_constH.row_idx[tid];
        d_shift_cn_out[tid] = d_constH.shift_cn[tid];
        d_shift_vn_out[tid] = d_constH.shift_vn[tid];
    }
    
    // 复制统计信息
    if (tid == 0) {
        *d_nnz_out = d_constH.nnz;
    }
}

// 测试访问内核 - 验证数据访问功能
__global__ void test_access_kernel() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        printf("=== GPU常量内存访问测试 ===\n");
        printf("非零元素总数: %d\n", d_constH.nnz);
        printf("第一行指针: %d -> %d\n", d_constH.row_ptr[0], d_constH.row_ptr[1]);
        printf("第一列指针: %d -> %d\n", d_constH.col_ptr[0], d_constH.col_ptr[1]);
        
        // 显示前几个非零元素
        printf("前5个CSR元素:\n");
        for (int i = 0; i < 5 && i < d_constH.nnz; i++) {
            printf("  [%d]: col=%d, shift=%d\n", i, d_constH.col_idx[i], d_constH.shift_cn[i]);
        }
        
        printf("前5个CSC元素:\n");
        for (int i = 0; i < 5 && i < d_constH.nnz; i++) {
            printf("  [%d]: row=%d, shift=%d\n", i, d_constH.row_idx[i], d_constH.shift_vn[i]);
        }
    }
}

struct CompressedH compress_H_matrix(int H[H_ROWS][H_COLS]) {
    struct CompressedH compH;
    
    // 统计非零元素总数
    compH.nnz = 0;
    for (int i = 0; i < H_ROWS; i++) {
        for (int j = 0; j < H_COLS; j++) {
            if (H[i][j] != -1) compH.nnz++;
        }
    }
    
    // 分配内存
    compH.row_ptr = (short*)malloc((H_ROWS + 1) * sizeof(short));
    compH.col_idx = (short*)malloc(compH.nnz * sizeof(short));
    compH.shift_cn = (short*)malloc(compH.nnz * sizeof(short));
    compH.col_ptr = (short*)malloc((H_COLS + 1) * sizeof(short));
    compH.row_idx = (short*)malloc(compH.nnz * sizeof(short));
    compH.shift_vn = (short*)malloc(compH.nnz * sizeof(short));
    
    // 构建CSR格式
    int csr_index = 0;
    compH.row_ptr[0] = 0;
    
    for (int i = 0; i < H_ROWS; i++) {
        for (int j = 0; j < H_COLS; j++) {
            if (H[i][j] != -1) {
                compH.col_idx[csr_index] = j;
                compH.shift_cn[csr_index] = H[i][j];
                csr_index++;
            }
        }
        compH.row_ptr[i + 1] = csr_index;
    }
    
    // 构建CSC格式
    int col_nnz[H_COLS] = {0};
    for (int j = 0; j < H_COLS; j++) {
        for (int i = 0; i < H_ROWS; i++) {
            if (H[i][j] != -1) col_nnz[j]++;
        }
    }
    
    compH.col_ptr[0] = 0;
    for (int j = 0; j < H_COLS; j++) {
        compH.col_ptr[j + 1] = compH.col_ptr[j] + col_nnz[j];
    }
    
    int csc_index[H_COLS] = {0};
    for (int i = 0; i < H_ROWS; i++) {
        for (int j = 0; j < H_COLS; j++) {
            if (H[i][j] != -1) {
                int pos = compH.col_ptr[j] + csc_index[j];
                compH.row_idx[pos] = i;
                compH.shift_vn[pos] = H[i][j];
                csc_index[j]++;
            }
        }
    }
    
    return compH;
}

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

// 验证函数 - 比较主机和GPU数据
int verify_gpu_data(const struct CompressedH* h_compH) {
    printf("\n=== 开始验证GPU常量内存数据 ===\n");
    
    // 分配设备内存用于接收GPU数据
    short *d_row_ptr_out, *d_col_ptr_out, *d_col_idx_out;
    short *d_row_idx_out, *d_shift_cn_out, *d_shift_vn_out, *d_nnz_out;
    
    cudaMalloc(&d_row_ptr_out, (H_ROWS + 1) * sizeof(short));
    cudaMalloc(&d_col_ptr_out, (H_COLS + 1) * sizeof(short));
    cudaMalloc(&d_col_idx_out, MAX_NNZ * sizeof(short));
    cudaMalloc(&d_row_idx_out, MAX_NNZ * sizeof(short));
    cudaMalloc(&d_shift_cn_out, MAX_NNZ * sizeof(short));
    cudaMalloc(&d_shift_vn_out, MAX_NNZ * sizeof(short));
    cudaMalloc(&d_nnz_out, sizeof(short));
    
    // 启动验证内核
    int threads = 256;
    int blocks = (MAX_NNZ + threads - 1) / threads;
    verify_constant_memory_kernel<<<blocks, threads>>>(
        d_row_ptr_out, d_col_ptr_out, d_col_idx_out, d_row_idx_out,
        d_shift_cn_out, d_shift_vn_out, d_nnz_out);
    
    cudaDeviceSynchronize();
    
    // 分配主机内存接收GPU数据
    short* gpu_row_ptr = (short*)malloc((H_ROWS + 1) * sizeof(short));
    short* gpu_col_ptr = (short*)malloc((H_COLS + 1) * sizeof(short));
    short* gpu_col_idx = (short*)malloc(MAX_NNZ * sizeof(short));
    short* gpu_row_idx = (short*)malloc(MAX_NNZ * sizeof(short));
    short* gpu_shift_cn = (short*)malloc(MAX_NNZ * sizeof(short));
    short* gpu_shift_vn = (short*)malloc(MAX_NNZ * sizeof(short));
    short gpu_nnz;
    
    // 从GPU复制数据到主机
    cudaMemcpy(gpu_row_ptr, d_row_ptr_out, (H_ROWS + 1) * sizeof(short), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_col_ptr, d_col_ptr_out, (H_COLS + 1) * sizeof(short), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_col_idx, d_col_idx_out, MAX_NNZ * sizeof(short), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_row_idx, d_row_idx_out, MAX_NNZ * sizeof(short), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_shift_cn, d_shift_cn_out, MAX_NNZ * sizeof(short), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_shift_vn, d_shift_vn_out, MAX_NNZ * sizeof(short), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gpu_nnz, d_nnz_out, sizeof(short), cudaMemcpyDeviceToHost);
    
    // 验证数据一致性
    int errors = 0;
    
    // 验证nnz
    if (gpu_nnz != h_compH->nnz) {
        printf("错误: nnz不匹配 - 主机:%d, GPU:%d\n", h_compH->nnz, gpu_nnz);
        errors++;
    }
    
    // 验证行指针
    for (int i = 0; i <= H_ROWS; i++) {
        if (gpu_row_ptr[i] != h_compH->row_ptr[i]) {
            printf("错误: row_ptr[%d]不匹配 - 主机:%d, GPU:%d\n", 
                   i, h_compH->row_ptr[i], gpu_row_ptr[i]);
            errors++;
        }
    }
    
    // 验证列指针
    for (int i = 0; i <= H_COLS; i++) {
        if (gpu_col_ptr[i] != h_compH->col_ptr[i]) {
            printf("错误: col_ptr[%d]不匹配 - 主机:%d, GPU:%d\n", 
                   i, h_compH->col_ptr[i], gpu_col_ptr[i]);
            errors++;
        }
    }
    
    // 验证CSR数据
    for (int i = 0; i < h_compH->nnz; i++) {
        if (gpu_col_idx[i] != h_compH->col_idx[i]) {
            printf("错误: col_idx[%d]不匹配 - 主机:%d, GPU:%d\n", 
                   i, h_compH->col_idx[i], gpu_col_idx[i]);
            errors++;
        }
        
        if (gpu_shift_cn[i] != h_compH->shift_cn[i]) {
            printf("错误: shift_cn[%d]不匹配 - 主机:%d, GPU:%d\n", 
                   i, h_compH->shift_cn[i], gpu_shift_cn[i]);
            errors++;
        }
    }
    
    // 验证CSC数据
    for (int i = 0; i < h_compH->nnz; i++) {
        if (gpu_row_idx[i] != h_compH->row_idx[i]) {
            printf("错误: row_idx[%d]不匹配 - 主机:%d, GPU:%d\n", 
                   i, h_compH->row_idx[i], gpu_row_idx[i]);
            errors++;
        }
        
        if (gpu_shift_vn[i] != h_compH->shift_vn[i]) {
            printf("错误: shift_vn[%d]不匹配 - 主机:%d, GPU:%d\n", 
                   i, h_compH->shift_vn[i], gpu_shift_vn[i]);
            errors++;
        }
    }
    
    if (errors == 0) {
        printf("✓ 验证通过！主机和GPU数据完全一致\n");
        printf("  - 非零元素数量: %d\n", gpu_nnz);
        printf("  - CSR格式数据: 正确\n");
        printf("  - CSC格式数据: 正确\n");
    } else {
        printf("✗ 验证失败！发现 %d 个数据不一致\n", errors);
    }
    
    // 清理内存
    free(gpu_row_ptr); free(gpu_col_ptr); free(gpu_col_idx);
    free(gpu_row_idx); free(gpu_shift_cn); free(gpu_shift_vn);
    
    cudaFree(d_row_ptr_out); cudaFree(d_col_ptr_out); cudaFree(d_col_idx_out);
    cudaFree(d_row_idx_out); cudaFree(d_shift_cn_out); cudaFree(d_shift_vn_out);
    cudaFree(d_nnz_out);
    
    return errors;
}

void print_memory_usage() {
    size_t total = sizeof(struct DeviceConstantH);
    printf("\n=== 常量内存使用情况 ===\n");
    printf("结构体总大小: %zu 字节 (%.2f KB)\n", total, total / 1024.0);
    printf("行指针: %zu 字节\n", (H_ROWS + 1) * sizeof(short));
    printf("列指针: %zu 字节\n", (H_COLS + 1) * sizeof(short));
    printf("数组数据: %zu 字节\n", 4 * MAX_NNZ * sizeof(short));
    
    if (total > 65536) {
        printf("⚠️  警告: 可能超出64KB常量内存限制!\n");
    } else {
        printf("✓ 在64KB常量内存限制内\n");
    }
}

int main() {
    // 初始化矩阵
    int hhh[H_ROWS][H_COLS];
    for (int i = 0; i < H_ROWS; i++) {
        for (int j = 0; j < H_COLS; j++) {
            hhh[i][j] = h_base_1_i0[i * H_COLS + j];
        }
    }
    
    // 压缩矩阵
    struct CompressedH compH = compress_H_matrix(hhh);
    printf("压缩完成，非零元素数量: %d\n", compH.nnz);
    
    // 显示内存使用情况
    print_memory_usage();
    
    // 复制到GPU常量内存
    cudaError_t err = copy_H_to_constant_memory(&compH);
    if (err != cudaSuccess) {
        printf("复制到GPU失败\n");
        return -1;
    }
    
    // 在GPU上测试访问
    printf("\n=== GPU访问测试 ===\n");
    test_access_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    // 验证数据一致性
    int verification_errors = verify_gpu_data(&compH);
    
    // 清理主机内存
    free(compH.row_ptr); free(compH.col_idx); free(compH.shift_cn);
    free(compH.col_ptr); free(compH.row_idx); free(compH.shift_vn);
    
    printf("\n程序执行完成，验证结果: %s\n", 
           verification_errors == 0 ? "成功" : "失败");
    
    return verification_errors;
}