#ifndef NR_LDPC_H
#define NR_LDPC_H



#include <stdint.h>    // for uint8_t, uint16_t, uint32_t, uint64_t

// Constants for 5G NR QC-LDPC code
#define BG1_ROW 46      // number of rows in base graph (BG1)
#define BG1_COL 68      // number of cols in base graph (BG1)

#define BG2_ROW 42      // number of rows in base graph (BG2)
#define BG2_COL 52      // number of cols in base graph (BG2)

#define MAX_Z 384     // Maximum lift size in 5G NR

#define MAX_ITER 20    // Maximum number of iterations
#define ALPHA 0.75f    // Scaling factor for Min-Sum algorithm

// Constants for quantization
#define MAX_LLR 127
#define MIN_LLR -128


// Z_array is the array of Z values for the 5G NR QC-LDPC code
// i_LS ∈ {0,1,2,3,4,5,6,7}
int Z_array[51] = {
        2, 4, 8, 16, 32, 64, 128, 256,  
        3, 6, 12, 24, 48, 96, 192, 384,
        5, 10, 20, 40, 80, 160, 320,
        7, 14, 28, 56, 112, 224,
        9, 18, 36, 72, 144, 288, 
        11, 22, 44, 88, 176, 352,
        13, 26, 52, 104, 208,
        15, 30, 60, 120, 240
};


struct ldpc_params {
    int A;                          // payload size before CRC
    float rate;                     // Code rate
    int bg_index;                   // index of the NR basegraph
    int Z;                          // Lifting factor
    int K;                          // Number of bits per LDPC code block to be encoded
    int N;                          // Number of encoded bits per LDPC code block.
    int H_rows;                     // Number of rows in the base graph
    int H_cols;                     // Number of cols in the base graph.
    int* H;                         // Pointer to the parity check matrix, H.
    int n_nz_in_row;                // Max number of nonzero entries in a row of H
    int n_nz_in_col;                // Max number of nonzero entries in a col of H
    char* h_element_count_per_row;  // An array with all the number of nonzero entires in each row
                                    // of H
    char* h_element_count_per_col;
    int n_iterations;         // Number of iterations for the decoder
    int N_before_puncture;  // Adds 2*Z to N
};


#endif