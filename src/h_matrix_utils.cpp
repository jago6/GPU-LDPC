#include "h_matrix_utils.h"
#include <stdlib.h>

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
    compH.row_ptr = (short*)malloc((H_ROWS + 1) * sizeof(short));
    compH.col_idx = (short*)malloc(compH.nnz * sizeof(short));
    compH.shift_cn = (short*)malloc(compH.nnz * sizeof(short));
    
    compH.col_ptr = (short*)malloc((H_COLS + 1) * sizeof(short));
    compH.row_idx = (short*)malloc(compH.nnz * sizeof(short));
    compH.shift_vn = (short*)malloc(compH.nnz * sizeof(short));
    
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


