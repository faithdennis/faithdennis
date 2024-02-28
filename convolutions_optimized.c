#include <omp.h>
#include <x86intrin.h>

// TO DO: Clean up code and make consistent :)

// Computes the dot product of two vectores, vec1 and vec2, both of size n
int32_t naive_dot(uint32_t n, int32_t *vec1, int32_t *vec2) {
  
    int sum = 0;
  
    for (int i = 0; i < n; i++) {
        sum += vec1[i] * vec2[i];
    }
    return sum;
}

// Computes the convolution of two matrices
int naive_convolve(matrix_t *a_matrix, matrix_t *b_matrix, matrix_t **output_matrix) {
    
    int size = b_matrix->rows * b_matrix->cols;
    matrix_t *b_matrix_flipped = malloc(sizeof(matrix_t));
    b_matrix_flipped->data = malloc(sizeof(int32_t)*size);

    for (int i=size-1; i>=0; i--) {
        b_matrix_flipped->data[size-1-i] = b_matrix->data[i];
    }
  
    b_matrix_flipped->rows = b_matrix->rows;
    b_matrix_flipped->cols = b_matrix->cols;

    int output_rows = (a_matrix->rows - b_matrix->rows + 1);
    int output_cols = (a_matrix->cols - b_matrix->cols + 1);
    int output_size = output_rows*output_cols;
    
    *output_matrix = malloc(sizeof(matrix_t));
    (*output_matrix)->data = malloc(sizeof(int)*output_size);

// Checking for error allocating memory on the heap
//    if (!*output_matrix) {
  //      free(b_matrix_flipped->data);
    //    free(b_matrix_flipped);
      //  return -1;
    //}
    
    (*output_matrix)->rows = output_rows;
    (*output_matrix)->cols = output_cols;

    // row, column:
    for (int row=0; row<output_rows; row++) {
        for (int col=0; col<output_cols; col++) {
            int a_index = row*a_matrix->cols + col;
            int sum=0;
            for (int matrix_row=0; matrix_row<b_matrix->rows; matrix_row++) { 
                sum += naive_dot(b_matrix_flipped->cols, &(a_matrix->data[matrix_row*a_matrix->cols + a_index]), &(b_matrix_flipped->data[matrix_row*b_matrix_flipped->cols]));
            }
            (*output_matrix)->data[row*((*output_matrix)->cols) + col] = sum;
        }
    }
   
    free(b_matrix_flipped->data);
    free(b_matrix_flipped); 
    return 0;
}

int32_t parallelized_dot(uint32_t n, int32_t *vec1, int32_t *vec2) {
    int sum = 0;
    int i;
    __m256i sum_vec = _mm256_setzero_si256();
    for (i=0; i<n/8*8; i+=8) {
        __m256i tmp_a = _mm256_loadu_si256((__m256i*) (vec1 + i));
        __m256i tmp_flip_b = _mm256_loadu_si256((__m256i*) (vec2 + i));

        __m256i mul_vector = _mm256_mullo_epi32(tmp_a, tmp_flip_b);
        sum_vec = _mm256_add_epi32(sum_vec, mul_vector);
    }
    int32_t tmp_arr[8];
    _mm256_storeu_si256((__m256i*) tmp_arr, sum_vec);
    for (i; i<n; i++) {
        sum+=vec1[i]*vec2[i];
    }
    sum += tmp_arr[0] + tmp_arr[1] + tmp_arr[2] + tmp_arr[3] + tmp_arr[4] + tmp_arr[5] + tmp_arr[6] + tmp_arr[7];
    return sum;
}

int parallelized_convolve(matrix_t *a_matrix, matrix_t *b_matrix, matrix_t **output_matrix) {
    int size = b_matrix->rows * b_matrix->cols;
    matrix_t *b_matrix_flipped = malloc(sizeof(matrix_t));
    b_matrix_flipped->data = malloc(sizeof(int32_t)*size);

// #pragma omp parallel for
    for(int i=size-1; i>=0; i--) {
        b_matrix_flipped->data[size-1-i] = b_matrix->data[i];
    }
    b_matrix_flipped->rows = b_matrix->rows;
    b_matrix_flipped->cols = b_matrix->cols;

    int output_rows = (a_matrix->rows - b_matrix->rows + 1);
    int output_cols = (a_matrix->cols - b_matrix->cols + 1);
    int output_size = output_rows*output_cols;

    *output_matrix = malloc(sizeof(matrix_t));
    (*output_matrix)->data = malloc(sizeof(int)*output_size);
    (*output_matrix)->rows = output_rows;
    (*output_matrix)->cols = output_cols;
#pragma omp parallel for
    for (int row=0; row<output_rows; row++) {
#pragma omp parallel for
        for (int col=0; col<output_cols; col++) {
            int a_index = row*a_matrix->cols + col;
            int sum=0;
            for (int matrix_row=0; matrix_row<b_matrix->rows; matrix_row++) {
                sum += parallelized_dot(b_matrix_flipped->cols, &(a_matrix->data[matrix_row*a_matrix->cols + a_index]), &(b_matrix_flipped->data[matrix_row*b_matrix_flipped->cols]));
            }
            (*output_matrix)->data[row*((*output_matrix)->cols) + col] = sum;
        }
    }
    free(b_matrix_flipped->data);
    free(b_matrix_flipped);
    return 0;
}

// Executes a task
int execute_task(task_t *task) {
    matrix_t *a_matrix, *b_matrix, *output_matrix;

    if (read_matrix(get_a_matrix_path(task), &a_matrix))
        return -1;
    if (read_matrix(get_b_matrix_path(task), &b_matrix))
        return -1;
    if (convolve(a_matrix, b_matrix, &output_matrix))
        return -1;
    if (write_matrix(get_output_matrix_path(task), output_matrix))
        return -1;
  
// Free memory allocated on the heap
    free(a_matrix->data);
    free(b_matrix->data);
    free(output_matrix->data);
    free(a_matrix);
    free(b_matrix);
    free(output_matrix);
    return 0;
}
