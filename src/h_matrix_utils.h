#ifndef H_MATRIX_UTILS_H
#define H_MATRIX_UTILS_H

struct CompressedH {
    // CSR格式（行优化）
    short* row_ptr;     // 行指针数组，长度H_ROWS+1
    short* col_idx;     // 列索引数组
    short* shift_cn;    // 移位值数组
    
    // CSC格式（列优化）
    short* col_ptr;     // 列指针数组，长度H_COLS+1
    short* row_idx;     // 行索引数组
    short* shift_vn;    // 移位值数组
    
    short nnz;          // 非零元素总数
};

struct CompressedH compress_H_matrix(int H[H_ROWS][H_COLS]);

#endif