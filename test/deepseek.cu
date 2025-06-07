#include <cuda_runtime.h>
#include <vector>
#include <spdlog/spdlog.h>

// 常量定义 (根据实际情况调整)
#define H_ROWS 100       // 校验节点行数
#define H_COLS 200       // 变量节点列数
#define Z 64             // 提升因子
#define MAX_NNZ 5000     // 最大非零元素数
#define MAX_DEGREE 20    // 最大节点度数
#define ALPHA_GROUP 4    // 码字组内并行码字数
#define BETA_GROUPS 16   // 并行码字组数量
#define MAX_ITER 10      // 最大迭代次数
#define NORM_FACTOR 0.75 // 归一化因子α

// 压缩H矩阵结构 (主机端)
struct CompressedH {
    short* row_ptr;    // 行指针数组
    short* col_idx;    // 列索引数组
    short* shift_cn;   // 校验节点移位值
    short* col_ptr;    // 列指针数组
    short* row_idx;    // 行索引数组
    short* shift_vn;   // 变量节点移位值
    short nnz;         // 非零元素总数
};

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

// 量化函数 (float <-> int8)
__device__ __host__ inline int8_t quantize(float value, float scale) {
    return static_cast<int8_t>(value * scale);
}

__device__ __host__ inline float dequantize(int8_t value, float scale) {
    return static_cast<float>(value) / scale;
}

// 分层译码核函数
__global__ void layered_decoding_kernel(
    int layer_id,         // 当前处理的层ID
    int8_t* d_llr,        // 初始LLR值 (设备全局内存)
    int8_t* d_app,        // APP值 (设备全局内存)
    int8_t* d_c2v,        // C2V消息 (设备全局内存)
    float norm_factor,    // 归一化因子α
    float quant_scale,    // 量化比例因子
    int total_codewords   // 总码字数
) {
    // 计算全局索引
    const int z_idx = threadIdx.x;           // 提升因子维度 (0~Z-1)
    const int alpha_idx = threadIdx.y;       // 码字组内索引 (0~ALPHA_GROUP-1)
    const int beta_idx = blockIdx.x;         // 码字组索引 (0~BETA_GROUPS-1)
    
    // 计算全局码字索引
    const int codeword_idx = beta_idx * ALPHA_GROUP + alpha_idx;
    if (codeword_idx >= total_codewords) return;

    // 获取当前层对应的基图行
    const int base_row = layer_id;

    // 获取当前行在H矩阵中的起始/结束位置
    // start_idx和end_idx分别表示当前行的起始和结束索引
    // num_edges表示当前行的非零元素数量
    const int start_idx = d_constH.row_ptr[base_row];
    const int end_idx = d_constH.row_ptr[base_row + 1];
    const int num_edges = end_idx - start_idx;

    // 共享内存声明 (存储临时V2C消息)
    extern __shared__ int8_t shared_v2c[];
    __shared__ float min_vals[MAX_DEGREE];
    __shared__ char global_signs[MAX_DEGREE];

    // 遍历当前层所有边 (非零元素)
    for (int edge_offset = 0; edge_offset < num_edges; edge_offset++) {
        const int edge_idx = start_idx + edge_offset;
        
        // 获取基图列索引和移位值
        const short base_col = d_constH.col_idx[edge_idx];
        const short shift_val = d_constH.shift_cn[edge_idx];
        
        // 计算实际变量节点索引 (考虑循环移位)
        const int v_node_idx = base_col * Z + (z_idx + shift_val) % Z;
        
        // 全局内存索引
        const int gmem_app_idx = codeword_idx * (H_COLS * Z) + v_node_idx;
        const int gmem_c2v_idx = codeword_idx * d_constH.nnz + edge_idx;
        
        // 读取量化的APP和C2V值并反量化
        float app_val = dequantize(d_app[gmem_app_idx], quant_scale);
        float old_c2v = dequantize(d_c2v[gmem_c2v_idx], quant_scale);
        
        // V2C消息计算 (公式3)
        float v2c_val = app_val - old_c2v;
        
        // 存储V2C到共享内存 (量化存储)
        shared_v2c[edge_offset] = quantize(v2c_val, quant_scale);
        __syncthreads();
        
        // ---- 计算最小幅度和符号乘积 (核心NMSA步骤) ----
        if (z_idx == 0 && alpha_idx == 0) {
            // 初始化最小值和符号
            float min_mag = FLT_MAX;
            float second_min = FLT_MAX;
            int min_index = -1;
            char sign_product = 1;
            
            // 遍历所有边 (当前层)
            for (int i = 0; i < num_edges; i++) {
                // 反量化V2C值
                float v2c = dequantize(shared_v2c[i], quant_scale);
                
                // 符号计算
                char sign = (v2c >= 0) ? 1 : -1;
                sign_product *= sign;
                
                // 幅度计算
                float mag = fabsf(v2c);
                
                // 更新最小值和次小值
                if (mag < min_mag) {
                    second_min = min_mag;
                    min_mag = mag;
                    min_index = i;
                } else if (mag < second_min) {
                    second_min = mag;
                }
            }
            
            // 存储全局最小值信息
            min_vals[edge_offset] = (edge_offset == min_index) ? second_min : min_mag;
            global_signs[edge_offset] = sign_product;
        }
        __syncthreads();
        
        // ---- 计算新的C2V消息 ----
        if (min_vals[edge_offset] > 0) { // 有效值检查
            // 获取当前边的符号
            float cur_v2c = dequantize(shared_v2c[edge_offset], quant_scale);
            char cur_sign = (cur_v2c >= 0) ? 1 : -1;
            
            // 排除自身后的符号乘积
            char sign_without_self = global_signs[edge_offset] * cur_sign;
            
            // 计算新的C2V (公式4)
            float new_c2v = norm_factor * sign_without_self * min_vals[edge_offset];
            
            // 更新APP值 (公式5)
            app_val = app_val - old_c2v + new_c2v;
            
            // 写回APP值 (量化)
            d_app[gmem_app_idx] = quantize(app_val, quant_scale);
            
            // 存储新的C2V值 (量化)
            d_c2v[gmem_c2v_idx] = quantize(new_c2v, quant_scale);
        }
        __syncthreads();
    }
}

// 硬判决核函数
__global__ void hard_decision_kernel(
    int8_t* d_app,         // APP值
    uint8_t* d_hard_bits,  // 硬判决结果
    float quant_scale,     // 量化比例因子
    int total_codewords    // 总码字数
) {
    // 计算全局索引
    const int z_idx = threadIdx.x;
    const int alpha_idx = threadIdx.y;
    const int beta_idx = blockIdx.x;
    
    // 计算全局码字索引
    const int codeword_idx = beta_idx * ALPHA_GROUP + alpha_idx;
    if (codeword_idx >= total_codewords) return;
    
    // 遍历所有变量节点
    for (int col = 0; col < H_COLS; col++) {
        const int v_node_idx = col * Z + z_idx;
        const int gmem_idx = codeword_idx * (H_COLS * Z) + v_node_idx;
        
        // 反量化APP值
        float app_val = dequantize(d_app[gmem_idx], quant_scale);
        
        // 硬判决 (公式6)
        uint8_t bit = (app_val >= 0) ? 0 : 1;
        
        // 存储结果 (按位打包)
        const int bit_idx = v_node_idx % 8;
        const int byte_idx = v_node_idx / 8;
        atomicOr(&d_hard_bits[codeword_idx * (H_COLS * Z / 8) + byte_idx], bit << bit_idx);
    }
}

// 拷贝H矩阵到常量内存 (已实现)
cudaError_t copy_H_to_constant_memory(const CompressedH* h_compH) {
    // ... (使用提供的实现)
}

// 初始化日志系统 (已实现)
void setup_logs() {
    // ... (使用提供的实现)
}

// LDPC解码器类
class LDPCDecoder {
public:
    LDPCDecoder(int num_streams, float quant_scale = 127.0f / 5.0f)
        : num_streams_(num_streams), quant_scale_(quant_scale) {
        
        // 初始化CUDA流
        streams_.resize(num_streams_);
        for (int i = 0; i < num_streams_; ++i) {
            cudaStreamCreate(&streams_[i]);
        }
        
        // 分配设备内存
        cudaMalloc(&d_llr_, MAX_CODES * H_COLS * Z * sizeof(int8_t));
        cudaMalloc(&d_app_, MAX_CODES * H_COLS * Z * sizeof(int8_t));
        cudaMalloc(&d_c2v_, MAX_CODES * MAX_NNZ * sizeof(int8_t));
        cudaMalloc(&d_hard_bits_, MAX_CODES * (H_COLS * Z / 8) * sizeof(uint8_t));
    }

    ~LDPCDecoder() {
        // 释放资源
        for (auto& stream : streams_) {
            cudaStreamDestroy(stream);
        }
        cudaFree(d_llr_);
        cudaFree(d_app_);
        cudaFree(d_c2v_);
        cudaFree(d_hard_bits_);
    }

    // 解码一批码字
    void decode_batch(const int8_t* h_llr, int num_codewords, int stream_id) {
        auto logger = spdlog::get("LDPC");
        cudaStream_t stream = streams_[stream_id];
        
        // 1. 拷贝数据到设备
        cudaMemcpyAsync(d_llr_, h_llr, num_codewords * H_COLS * Z * sizeof(int8_t),
                        cudaMemcpyHostToDevice, stream);
        
        // 2. 初始化APP和C2V
        cudaMemsetAsync(d_app_, 0, MAX_CODES * H_COLS * Z * sizeof(int8_t), stream);
        cudaMemsetAsync(d_c2v_, 0, MAX_CODES * MAX_NNZ * sizeof(int8_t), stream);
        
        // 3. 分层译码迭代
        for (int iter = 0; iter < MAX_ITER; iter++) {
            for (int layer = 0; layer < H_ROWS; layer++) {
                // 设置线程结构 (Z, ALPHA_GROUP, 1) 块 x (BETA_GROUPS, 1, 1) 网格
                dim3 block_dim(Z, ALPHA_GROUP);
                dim3 grid_dim(BETA_GROUPS);
                
                // 计算共享内存大小 (存储临时V2C消息)
                size_t shared_mem_size = MAX_DEGREE * sizeof(int8_t) + 
                                        MAX_DEGREE * sizeof(float) +
                                        MAX_DEGREE * sizeof(char);
                
                // 启动核函数
                layered_decoding_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
                    layer, d_llr_, d_app_, d_c2v_, 
                    NORM_FACTOR, quant_scale_, num_codewords
                );
                
                // 错误检查
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    logger->error("解码错误: {}", cudaGetErrorString(err));
                }
            }
            
            // 4. 每迭代一次执行硬判决
            dim3 block_dim_hd(Z, ALPHA_GROUP);
            dim3 grid_dim_hd(BETA_GROUPS);
            
            hard_decision_kernel<<<grid_dim_hd, block_dim_hd, 0, stream>>>(
                d_app_, d_hard_bits_, quant_scale_, num_codewords
            );
            
            // 5. 可添加停止条件检查 (此处省略)
        }
        
        // 6. 拷贝结果回主机 (可选)
        // cudaMemcpyAsync(...);
    }

private:
    int num_streams_;
    float quant_scale_;
    std::vector<cudaStream_t> streams_;
    
    // 设备内存指针
    int8_t* d_llr_ = nullptr;
    int8_t* d_app_ = nullptr;
    int8_t* d_c2v_ = nullptr;
    uint8_t* d_hard_bits_ = nullptr;
    
    static constexpr int MAX_CODES = ALPHA_GROUP * BETA_GROUPS;
};

// 示例用法
int main() {
    setup_logs();
    auto logger = spdlog::get("LDPC");
    
    // 1. 准备H矩阵
    CompressedH h_compH;
    // ... (初始化H矩阵)
    
    // 2. 拷贝H矩阵到常量内存
    if (copy_H_to_constant_memory(&h_compH) != cudaSuccess) {
        logger->critical("无法加载H矩阵到设备");
        return 1;
    }
    
    // 3. 创建解码器 (4个流)
    LDPCDecoder decoder(4);
    
    // 4. 准备测试数据
    std::vector<int8_t> llr_data(/* 大小 = 码字数*H_COLS*Z */);
    
    // 5. 多流并行解码
    #pragma omp parallel for
    for (int i = 0; i < 4; i++) {
        decoder.decode_batch(llr_data.data() + i * BATCH_SIZE, 
                            BATCH_SIZE, i);
    }
    
    logger->info("解码完成");
    return 0;
}