#ifndef COMMON_ERROR_HANDLER_H
#define COMMON_ERROR_HANDLER_H

#include <cuda_runtime.h> // For cudaError_t

// --- 常见定义 ---
// 宏定义，用于包装错误处理函数调用
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// 函数声明
// 此函数将在 error_handler.cu 文件中定义
// 它检查CUDA API调用返回的错误代码
// 如果发生错误，它会打印错误信息（包括文件名和行号）并退出程序
static void HandleError(cudaError_t err, const char *file, int line);

#endif // COMMON_ERROR_HANDLER_H
