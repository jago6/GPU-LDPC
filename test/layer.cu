#define ALPHA 4 // 每个码字组内的线程数
#define BETA 8  // 码字组数
#define Z 384    // 提升因子维度

#define H_ROWS 46 // H矩阵行数
#define H_COLS 68 // H矩阵列数
#define MAX_NNZ 316 // 最大度数

// 设备端常量H矩阵结构
__constant__ struct DeviceConstantH {
    short row_ptr[H_ROWS + 1];  // 行指针数组
    short col_ptr[H_COLS + 1];  // 列指针数组
    short col_idx[MAX_NNZ];     // 列索引
    short row_idx[MAX_NNZ];     // 行索引
    short shift_cn[MAX_NNZ];    // 校验节点移位值
    short shift_vn[MAX_NNZ];    // 变量节点移位值
    short nnz;                  // 非零元素数
} d_constH;


// 量化函数（8-bit）
__device__ __forceinline__ int8_t quantize(float val, float scale) {
    return static_cast<int8_t>(fminf(fmaxf(val / scale, -128.0f), 127.0f));
}

// 反量化函数
__device__ __forceinline__ float dequantize(int8_t val, float scale) {
    return static_cast<float>(val) * scale;
}

// 分层译码核函数
// 线程结构一个gird里有 BETA 个block，一个block有（Z，ALPHA）个thread


__global__ void layered_decoding_kernel(
    int max_iterations,      // 最大迭代次数
    int n_layers,            // H矩阵层数（基图行数）
    int8_t* d_llr,           // 输入LLR（全局内存）
    int8_t* d_app,           // 中间APP（全局内存）
    int8_t* d_c2v,           // 校验节点消息（全局内存）
    float norm_factor,       // 归一化因子α(Min-Sum)
    float quant_scale,       // 量化比例因子
    int total_codewords      // 总码字数
) {
    // --- 1. 线程索引计算 ---
    const int z_idx = threadIdx.x;                  // 提升因子维度 [0, Z-1]
    const int alpha_idx = threadIdx.y;              // 码字组内索引 [0, ALPHA_GROUP-1]
    const int beta_idx = blockIdx.x;                // 码字组批次索引
    const int codeword_idx = beta_idx * blockDim.y + alpha_idx;
    
    // 检测越界（总码字数保护）
    if (codeword_idx >= total_codewords) return;

    // --- 2. 共享内存分配 ---
    extern __shared__ int8_t dynamic_shared[];
    // 动态划分共享内存区域（不同层边数变化）
    int8_t* shared_v2c = dynamic_shared;  // V2C消息 [num_edges][ALPHA]
    float* shared_min_vals = (float*)(shared_v2c + d_constH.max_row_degree * blockDim.y);
    float* shared_second_min = shared_min_vals + blockDim.y;
    char*  shared_signs    = (char*)(shared_second_min + blockDim.y);
    int*   shared_min_idx   = (int*)(shared_signs + blockDim.y);

    // --- 3. 主迭代循环 ---
    for (int iter = 0; iter < max_iterations; ++iter) {
        // --- 4. 分层处理循环 ---
        for (int layer = 0; layer < n_layers; ++layer) {
            // 获取当前层数据
            const int start_idx = d_constH.row_ptr[layer];
            const int end_idx = d_constH.row_ptr[layer+1];
            const int num_edges = end_idx - start_idx; // 当前层边数
            
            // --- 4.1 计算V2C（变量->校验消息） ---
            for (int edge = 0; edge < num_edges; ++edge) {
                const int edge_idx = start_idx + edge;
                // H矩阵元素信息（基图列号+偏移）
                const short base_col = d_constH.col_idx[edge_idx];
                const short shift_val = d_constH.shift_cn[edge_idx];
                
                // 计算全局内存索引
                const int v_addr = codeword_idx * (H_COLS * blockDim.x) 
                                 + base_col * blockDim.x + (z_idx + shift_val) % blockDim.x;
                const int c2v_addr = codeword_idx * d_constH.nnz + edge_idx;
                
                // 读取并反量化
                float app_val = dequantize(d_app[v_addr], quant_scale);
                float old_c2v = dequantize(d_c2v[c2v_addr], quant_scale);
                
                // V2C = APP - C2V
                float v2c_val = app_val - old_c2v;
                
                // 存入共享内存（量化降低存储）
                int shared_pos = edge * blockDim.y + alpha_idx;
                shared_v2c[shared_pos] = quantize(v2c_val, quant_scale);
            }
            __syncthreads(); // 等待所有边处理完成

        for (int edge = 0; edge < num_edges; ++edge) {
        // 根据z_idx计算实际变量节点位置
        const int shift_val = d_constH.shift_cn[edge];
        const int v_pos = base_col * Z + (z_idx + shift_val) % Z;
        
        // 计算V2C (存储在寄存器)
        local_v2c[edge] = calc_v2c(...); 
        
        // 更新统计量
        sign_product *= sign(local_v2c[edge]);
        float abs_v2c = fabsf(local_v2c[edge]);
        if (abs_v2c < min_val) {
            second_min = min_val;
            min_val = abs_v2c;
            min_index = edge;
        } else if (abs_v2c < second_min) {
            second_min = abs_v2c;
        }
    }
    
    // 3. C2V计算和APP更新
    for (int edge = 0; edge < num_edges; ++edge) {
        // 计算新的C2V
        float new_c2v = calc_new_c2v(local_v2c[edge], min_val, second_min, ...);
        
        // 更新APP
        d_app[v_pos] = app - old_c2v + new_c2v;
        d_c2v[edge_idx] = new_c2v;
    }


            __syncthreads(); // 等待层更新完成
        } // 层循环结束
    } // 迭代循环结束
}