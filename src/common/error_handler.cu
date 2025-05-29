#include <stdio.h>      // For fprintf, stderr
#include <stdlib.h>     // For exit, EXIT_FAILURE
#include <cuda_runtime.h> // For cudaError_t, cudaGetErrorString, cudaSuccess

// 包含对应的头文件 (可选，但好的实践是包含自身头文件以检查声明与定义是否匹配)
#include "error_handler.h"

// --- 错误处理函数定义 ---
// 检查CUDA API调用返回的错误代码
// 如果发生错误，它会打印错误信息（包括文件名和行号）并退出程序
// 注意：在 .cu 文件中，此函数可以保持 static，
// 因为 HANDLE_ERROR 宏在包含 error_handler.h 的任何 .cu 或 .cpp 文件中都会展开为对这个函数的调用。
// 如果其他 .cu 文件需要直接链接到这个函数（不通过宏），则不应是 static。
// 但通过宏使用是常见做法。
/*static*/
void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s in %s at line %d\n", cudaGetErrorString(err), file, line);
        // 在实际的库中，您可能不想直接调用 exit()，而是返回一个错误码或抛出异常
        // 但对于这个概念性代码，exit() 是直接的。
        exit(EXIT_FAILURE);
    }
}