
#include "../src/h_matrices/h_bg_1_i_0.h"
#include <stdlib.h>
#include <stdio.h>
#define  H_ROWS 46
#define  H_COLS 68
struct CompressedH {
    // CSR格式（行优化）
    int* row_ptr;     // 行指针数组，长度H_ROWS+1
    int* col_idx;     // 列索引数组
    int* shift_cn;    // 移位值数组
    
    // CSC格式（列优化）
    int* col_ptr;     // 列指针数组，长度H_COLS+1
    int* row_idx;     // 行索引数组
    int* shift_vn;    // 移位值数组
    
    int nnz;          // 非零元素总数
};

// 预处理函数：将原始H矩阵转换为双格式压缩存储
struct CompressedH compress_H_matrix(int H[H_ROWS][H_COLS]) {
    struct CompressedH compH;
    
    // 步骤1: 统计非零元素总数
    compH.nnz = 0;
    for (int i = 0; i < H_ROWS; i++) {
        for (int j = 0; j < H_COLS; j++) {
            if (H[i][j] != -1) compH.nnz++;
        }
    }
    
    // 步骤2: 分配内存
    compH.row_ptr = (int*)malloc((H_ROWS + 1) * sizeof(int));
    compH.col_idx = (int*)malloc(compH.nnz * sizeof(int));
    compH.shift_cn = (int*)malloc(compH.nnz * sizeof(int));
    
    compH.col_ptr = (int*)malloc((H_COLS + 1) * sizeof(int));
    compH.row_idx = (int*)malloc(compH.nnz * sizeof(int));
    compH.shift_vn = (int*)malloc(compH.nnz * sizeof(int));
    
    // 步骤3: 构建CSR格式（行优化）
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
        compH.row_ptr[i + 1] = csr_index; // 下一行起始位置
    }
    
    // 步骤4: 构建CSC格式（列优化）
    // 4.1 统计每列非零元素数
    int col_nnz[H_COLS] = {0};
    for (int j = 0; j < H_COLS; j++) {
        for (int i = 0; i < H_ROWS; i++) {
            if (H[i][j] != -1) col_nnz[j]++;
        }
    }
    
    // 4.2 设置列指针
    compH.col_ptr[0] = 0;
    for (int j = 0; j < H_COLS; j++) {
        compH.col_ptr[j + 1] = compH.col_ptr[j] + col_nnz[j];
    }
    
    // 4.3 填充行索引和移位值
    int csc_index[H_COLS] = {0}; // 每列当前填充位置
    
    for (int i = 0; i < H_ROWS; i++) {
        for (int j = 0; j < H_COLS; j++) {
            if (H[i][j] != -1) {
                int col = j;
                int pos = compH.col_ptr[col] + csc_index[col];
                compH.row_idx[pos] = i;
                compH.shift_vn[pos] = H[i][j];
                csc_index[col]++;
            }
        }
    }
    
    return compH;
}

// 打印压缩矩阵信息（调试用）
void print_compressedH(struct CompressedH compH) {
    printf("=== CSR Format (Row-Optimized) ===\n");
    printf("Row Pointers: [");
    for (int i = 0; i <= H_ROWS; i++) {
        printf("%d%s", compH.row_ptr[i], (i < H_ROWS) ? ", " : "]\n");
    }
    
    printf("\nColumn Indices and Shifts:\n");
    for (int i = 0; i < H_ROWS; i++) {
        printf("Row %2d: ", i);
        for (int pos = compH.row_ptr[i]; pos < compH.row_ptr[i + 1]; pos++) {
            printf("(Col:%2d, Shift:%2d) ", compH.col_idx[pos], compH.shift_cn[pos]);
        }
        printf("\n");
    }
    
    printf("\n=== CSC Format (Column-Optimized) ===\n");
    printf("Column Pointers: [");
    for (int j = 0; j <= H_COLS; j++) {
        printf("%d%s", compH.col_ptr[j], (j < H_COLS) ? ", " : "]\n");
    }
    
    printf("\nRow Indices and Shifts:\n");
    for (int j = 0; j < H_COLS; j++) {
        printf("Col %2d: ", j);
        for (int pos = compH.col_ptr[j]; pos < compH.col_ptr[j + 1]; pos++) {
            printf("(Row:%2d, Shift:%2d) ", compH.row_idx[pos], compH.shift_vn[pos]);
        }
        printf("\n");
    }
    
    printf("\nTotal Non-zero Elements: %d\n", compH.nnz);
}

int main(){
    int hhh[46][68] =  {0};
    for (int i = 0; i < H_ROWS; i++) {
        for (int j = 0; j < H_COLS; j++) {
            hhh[i][j] = h_base_1_i0[i * H_COLS + j];
        }
    }
    struct CompressedH compH = compress_H_matrix(hhh);
    print_compressedH(compH);
    return 0;
}