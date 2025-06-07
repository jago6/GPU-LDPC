#define ALPHA 4 // 每个码字组内的线程数
#define BETA 8  // 码字组数
#define Z 384    // 提升因子维度

#define H_ROWS 46 // H矩阵行数
#define H_COLS 68 // H矩阵列数
#define MAX_NNZ 316 // 最大度数
#define MAX_ROW_DEGREE 8 // 最大行度数

// 设备端常量H矩阵结构
__constant__ struct DeviceConstantH {
    short row_ptr[H_ROWS + 1];  // 行指针数组
    short col_ptr[H_COLS + 1];  // 列指针数组
    short col_idx[MAX_NNZ];     // 列索引
    short row_idx[MAX_NNZ];     // 行索引
    short shift_cn[MAX_NNZ];    // 校验节点移位值
    short shift_vn[MAX_NNZ];    // 变量节点移位值
    short nnz;                  // 非零元素数
    short max_row_degree;       // 最大行度数
} d_constH;

// 量化函数（8-bit）
__device__ __forceinline__ int8_t quantize(float val, float scale) {
    return static_cast<int8_t>(fminf(fmaxf(val / scale, -128.0f), 127.0f));
}

// 反量化函数
__device__ __forceinline__ float dequantize(int8_t val, float scale) {
    return static_cast<float>(val) * scale;
}

// 符号函数
__device__ __forceinline__ int sign(float val) {
    return (val >= 0) ? 1 : -1;
}

// 分层NMSA译码核函数
__global__ void nmsa_layered_decoding_kernel(
    int max_iterations,      // 最大迭代次数
    int n_layers,            // H矩阵层数（基图行数）
    int8_t* d_llr_init,      // 初始LLR（对应L_init_vj）
    int8_t* d_app,           // APP值（对应L_app_vj）
    int8_t* d_c2v,           // C2V消息（对应L_ci->vj）
    float norm_factor,       // 归一化因子α (0 < α ≤ 1)
    float quant_scale,       // 量化比例因子
    int total_codewords      // 总码字数
) {
    // --- 1. 线程索引计算 ---
    const int z_idx = threadIdx.x;                  // 提升因子维度 [0, Z-1]
    const int alpha_idx = threadIdx.y;              // 码字组内索引 [0, ALPHA-1]
    const int beta_idx = blockIdx.x;                // 码字组批次索引
    const int codeword_idx = beta_idx * blockDim.y + alpha_idx;
    
    // 检测越界（总码字数保护）
    if (codeword_idx >= total_codewords) return;

    // --- 2. 共享内存分配 ---
    extern __shared__ int8_t dynamic_shared[];
    
    // 动态划分共享内存区域
    int8_t* shared_v2c = dynamic_shared;  // V2C消息 [max_edges][ALPHA]
    float* shared_min_vals = (float*)(shared_v2c + MAX_ROW_DEGREE * blockDim.y);
    float* shared_second_min = shared_min_vals + blockDim.y;
    int*   shared_sign_product = (int*)(shared_second_min + blockDim.y);
    int*   shared_min_idx = shared_min_idx + blockDim.y;

    // --- 3. 初始化 (对应公式1,2) ---
    // L_app_vj = L_init_vj (公式1)
    // L^0_ci->vj = 0 (公式2)
    if (z_idx == 0 && alpha_idx == 0) {
        for (int col = 0; col < H_COLS; ++col) {
            int v_addr = codeword_idx * (H_COLS * Z) + col * Z;
            for (int z = 0; z < Z; ++z) {
                d_app[v_addr + z] = d_llr_init[v_addr + z];
            }
        }
        
        // 初始化C2V消息为0
        for (int edge = 0; edge < d_constH.nnz; ++edge) {
            int c2v_addr = codeword_idx * d_constH.nnz + edge;
            d_c2v[c2v_addr] = 0;
        }
    }
    __syncthreads();

    // --- 4. 主迭代循环 ---
    for (int iter = 0; iter < max_iterations; ++iter) {
        
        // --- 5. 分层处理循环 ---
        for (int layer = 0; layer < n_layers; ++layer) {
            
            // 获取当前层边信息
            const int start_idx = d_constH.row_ptr[layer];
            const int end_idx = d_constH.row_ptr[layer + 1];
            const int num_edges = end_idx - start_idx;
            
            // --- 5.1 V2C消息更新 (对应公式3) ---
            // L^n_vj->ci = L_app_vj - L^n_ci->vj
            for (int edge = 0; edge < num_edges; ++edge) {
                const int edge_idx = start_idx + edge;
                const short base_col = d_constH.col_idx[edge_idx];
                const short shift_val = d_constH.shift_cn[edge_idx];
                
                // 计算循环移位后的变量节点位置
                const int shifted_z = (z_idx + shift_val) % Z;
                const int v_addr = codeword_idx * (H_COLS * Z) + base_col * Z + shifted_z;
                const int c2v_addr = codeword_idx * d_constH.nnz + edge_idx;
                
                // 读取APP和C2V值
                float app_val = dequantize(d_app[v_addr], quant_scale);
                float c2v_val = dequantize(d_c2v[c2v_addr], quant_scale);
                
                // V2C = APP - C2V (公式3)
                float v2c_val = app_val - c2v_val;
                
                // 存储到共享内存
                int shared_pos = edge * blockDim.y + alpha_idx;
                shared_v2c[shared_pos] = quantize(v2c_val, quant_scale);
            }
            __syncthreads();
            
            // --- 5.2 保存旧的C2V值 ---
            __shared__ float old_c2v_layer[MAX_ROW_DEGREE * ALPHA];
            
            // 保存当前层的旧C2V值
            for (int edge = 0; edge < num_edges; ++edge) {
                const int edge_idx = start_idx + edge;
                const int c2v_addr = codeword_idx * d_constH.nnz + edge_idx;
                int shared_c2v_pos = edge * blockDim.y + alpha_idx;
                old_c2v_layer[shared_c2v_pos] = dequantize(d_c2v[c2v_addr], quant_scale);
            }
            __syncthreads();
            
            // --- 5.3 计算每个校验节点的统计量 ---
            // 计算符号乘积和最小值
            float min_val = FLT_MAX;
            float second_min = FLT_MAX;
            int min_index = -1;
            int sign_prod = 1;
            
            for (int edge = 0; edge < num_edges; ++edge) {
                int shared_pos = edge * blockDim.y + alpha_idx;
                float v2c_val = dequantize(shared_v2c[shared_pos], quant_scale);
                
                // 计算符号乘积
                sign_prod *= sign(v2c_val);
                
                // 计算最小值和次小值
                float abs_v2c = fabsf(v2c_val);
                if (abs_v2c < min_val) {
                    second_min = min_val;
                    min_val = abs_v2c;
                    min_index = edge;
                } else if (abs_v2c < second_min) {
                    second_min = abs_v2c;
                }
            }
            
            // 存储统计量到共享内存
            shared_min_vals[alpha_idx] = min_val;
            shared_second_min[alpha_idx] = second_min;
            shared_sign_product[alpha_idx] = sign_prod;
            shared_min_idx[alpha_idx] = min_index;
            __syncthreads();
            
            // --- 5.4 C2V消息更新 (对应公式4) ---
            // L^(n+1)_ci->vj = α * (∏ sgn(L^n_v'->ci)) * min|L^n_v'->ci|
            for (int edge = 0; edge < num_edges; ++edge) {
                const int edge_idx = start_idx + edge;
                const short base_col = d_constH.col_idx[edge_idx];
                const short shift_val = d_constH.shift_cn[edge_idx];
                
                // 计算循环移位后的变量节点位置
                const int shifted_z = (z_idx + shift_val) % Z;
                const int v_addr = codeword_idx * (H_COLS * Z) + base_col * Z + shifted_z;
                const int c2v_addr = codeword_idx * d_constH.nnz + edge_idx;
                
                // 获取当前V2C消息
                int shared_pos = edge * blockDim.y + alpha_idx;
                float current_v2c = dequantize(shared_v2c[shared_pos], quant_scale);
                
                // 计算除当前边外的符号乘积
                int sign_except_current = shared_sign_product[alpha_idx] * sign(current_v2c);
                
                // 选择最小值（排除当前边）
                float min_except_current;
                if (edge == shared_min_idx[alpha_idx]) {
                    min_except_current = shared_second_min[alpha_idx];
                } else {
                    min_except_current = shared_min_vals[alpha_idx];
                }
                
                // 计算新的C2V消息 (公式4)
                float new_c2v = norm_factor * sign_except_current * min_except_current;
                
                // 存储新的C2V消息
                d_c2v[c2v_addr] = quantize(new_c2v, quant_scale);
            }
            __syncthreads();
            
            // --- 5.5 APP值更新 (对应公式5) ---
            // L_app_vj = L_init_vj + Σ L^(n+1)_ci->vj
            // 在分层译码中，我们使用增量更新：APP = APP - old_C2V + new_C2V
            
            for (int edge = 0; edge < num_edges; ++edge) {
                const int edge_idx = start_idx + edge;
                const short base_col = d_constH.col_idx[edge_idx];
                const short shift_val = d_constH.shift_cn[edge_idx];
                
                // 计算循环移位后的变量节点位置
                const int shifted_z = (z_idx + shift_val) % Z;
                const int v_addr = codeword_idx * (H_COLS * Z) + base_col * Z + shifted_z;
                const int c2v_addr = codeword_idx * d_constH.nnz + edge_idx;
                
                // 读取当前APP值
                float current_app = dequantize(d_app[v_addr], quant_scale);
                
                // 获取新的C2V值
                float new_c2v = dequantize(d_c2v[c2v_addr], quant_scale);
                
                // 获取旧的C2V值
                int shared_c2v_pos = edge * blockDim.y + alpha_idx;
                float old_c2v = old_c2v_layer[shared_c2v_pos];
                
                // 增量更新APP: APP = APP - old_C2V + new_C2V
                // 这样保证了 APP = L_init + Σ(所有最新的C2V消息)
                float new_app = current_app - old_c2v + new_c2v;
                
                // 存储更新后的APP值
                d_app[v_addr] = quantize(new_app, quant_scale);
            }
            __syncthreads();
            
        } // 层循环结束
        
        // 可以在这里添加收敛检查
        // 硬判决: x̂_j = 0 if L_app_vj ≥ 0, else 1
        
    } // 迭代循环结束
}

// 辅助函数：计算共享内存大小
__host__ size_t calculate_shared_memory_size() {
    size_t v2c_size = MAX_ROW_DEGREE * ALPHA * sizeof(int8_t);
    size_t stats_size = 4 * ALPHA * sizeof(float); // min_vals, second_min, sign_product, min_idx
    return v2c_size + stats_size;
}