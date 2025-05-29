#ifndef NR_LDPC_H
#define NR_LDPC_H

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