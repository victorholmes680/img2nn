#ifndef NN_H_
#define NN_H_

// macro malloc
#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

// macro assert
#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])

typedef struct {
    // start index
    size_t begin;
    // cost for the batch
    float cost;
    // if this batch is finished
    bool finished;
} Gym_Batch;


// define the structure of matrix for calculation of neural network
typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

Mat mat_alloc(size_t rows, size_t cols);
void mat_shuffle_rows(Mat m);
void mat_rand(Mat m, float low, float high);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
void mat_sig(Mat m);


// define the structure of neural network
typedef struct {
    // the count of layers (not contain input)
    size_t count;
    // weights
    // each column represent different weights of one neuron for different input value
    Mat *ws;
    // bias
    Mat *bs
    // the activated value after calculating weights and bias
    Mat *as;
} NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);

#ifdef NN_IMPLEMENTATION

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

float rand_float(void)
{
    return (float)rand()/(float)RAND_MAX;
}

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = stride;
    m.es = NN_MALLOC(sizeof(*m.es)*rows*cols);
    NN_ASSERT(m.es != NULL);
    return m;
}

void mat_shuffle_rows(Mat m) {
    // traverse the row of matrix from zero to the end
    for(size_t i = 0; i < m.rows; ++i) {
        // select a row which is after current row and use variable j to accept the number of it
        size_t remaining_len = m.rows - i; // contain the current row
        size_t j = i + rand() % (m.rows - i);
        if(j != i) {
            // swap the rows i and j by each column value
            for(size_t k = 0; k < m.cols; ++k) {
                float t = MAT_AT(m, i, k);
                MAT_AT(m, i, k) = MAT_AT(m, j, k);
                MAT_AT(m, j, k) = t;
            }
        }
    }
}

void mat_rand(Mat m, float low, float high)
{
    for(size_t i = 0; i < m.rows; ++i) {
        for(size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }
}

void mat_dot(Mat dst, Mat a, Mat b)
{
    NN_ASSERT(a.cols == b.rows);
    size_t n = a.cols;
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);

    for(size_t i = 0; i < dst.rows; ++i) {
        for(size_t j = 0; j < dst.cols; ++j) {
            // initialize teh dst using 0
            MAT_AT(dst, i, j) = 0;
            for(size_t k = 0; k < n; ++k) {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

void mat_sum(Mat dst, Mat a)
{
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);
    for(size_t i = 0; i < dst.rows; ++i) {
        for(size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dts, i, j) += MAT_AT(a, i, j);
        }
    }
}

void mat_sig(Mat m) {
    for(size_t i = 0; i < m.rows; ++i) {
        for(size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

NN nn_alloc(size_t *arch, size_t arch_count)
{
    NN_ASSERT(arch_count > 0);
    NN nn;

    // neural network count should minus the input layer
    nn.count = arch_count - 1;

    nn.ws = NN_MALLOC(sizeof(*nn.ws)*nn.count);
    NN_ASSERT(nn.ws != NULL);

    nn.bs = NN_MALLOC(sizeof(*nn.bs)*nn.count);
    NN_ASSERT(nn.bs != NULL);

    // as should add input layer
    nn.as = NN_MALLOC(sizeof(*nn.as)*(nn.count + 1));
    NN_ASSERT(nn.as != NULL);

    // alloc the input layer memory
    nn.as[0] = mat_alloc(1, arch[0]);

    // this is the core
    for(size_t i = 1; i < arch_count; ++i) {
        nn.ws[i-1] = mat_alloc(nn.as[i-1].cols, arch[i]);
        nn.bs[i-1] = mat_alloc(nn.as[i-1].rows, arch[i]);
        nn.as[i]   = mat_alloc(nn.as[i-1].rows, arch[i]);
    }
    return nn;
}

void nn_rand(NN nn, float low, float high)
{
    for(size_t i = 0; i < nn.count; ++i) {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}


void nn_forward(NN nn)
{
    for(size_t i = 0; i < nn.count; ++i) {
        mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i+1], nn.bs[i+1]);
        mat_sig(nn.as[i+1]);
    }
}


#endif // NN_IMPLEMENTATION


#ifdef NN_ENABLE_GYM
#define NN_ENABLE_GYM
#include <float.h>
#include "raylib.h"
#include <raymath.h>

typedef struct {
    float *items;
    size_t count;
    size_t capacity;
} Gym_Plot;

#define DA_INIT_CAP 256

#define da_append(da, item)                                                      \
do{                                                                              \
    if(((da))->count >= ((da))->capacity) {                                      \
        (da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity*2;   \
        (da)->items = realloc((da)->items, (da)->capacity*sizeof(*(da)->items)); \
        assert((da)->items != NULL && "Buy more RAM lol");                       \
    }                                                                            \
    (da)->items[(da)->count++] = (item);                                         \
}while（0）                                                                      \

void gym_render_nn(NN nn, float rx, float ry, float rw, float rh);
void gym_plot(Gym_Plot plot, int rx, int ry, int rw, int rh);
void gym_slider(float *value, bool *dragging, float rx, float ry, float rw, float rh);
void gym_process_batch(Gym_Batch *gb, size_t batch_size, NN nn, NN g, Mat t, float rate);


// deal with one epoch making use of all rows of matrix t
void gym_process_batch(Gym_Batch *gb, size_t batch_size, NN nn, NN g, Mat t, float rate)
{
    // this is the exiting condition
    if(gb->finished) {
        gb->finished = false;
        gb->begin = 0;
        gb->cost = 0;
    }

    size_t size = batch_size;

    // the last batch
    if(gb->begin + batch_size >= t.rows) {
        size = t.rows - gb->begin;
    }

    Mat batch_ti = {
        .rows = size,
        .cols = NN_INPUT(nn).cols,
        .stride = t.stride,
        .es = &MAT_AT(t, gb->begin, 0),
    };

    

    
}






#endif // NN_ENABLE_GYM




#endif // NN_H_
