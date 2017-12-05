#include "ftrl.h"

#include <string>
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>

#define TOLERANCE 1e-6f


#if defined USEOMP
#include <omp.h>
#include <cstdio>

#endif

using namespace std;

float sign(float x) {
    if (x < 0) {
        return -1.0f;
    } else {
        return 1.0f;
    }
}

float calculate_sigma(float n, float grad, float alpha) {
    return (sqrt(n + grad * grad) - sqrt(n)) / alpha;
}

float calculate_w(float z, float n, ftrl_params &params) {
    float s = sign(z);
    if (s * z <= params.l1) {
        return 0.0f;
    }

    float w = (s * params.l1 - z) / ((params.beta + sqrt(n)) / params.alpha + params.l2);
    return w;
}

float log_loss(float y, float p) {
    if (y == 1.0f) {
        return -log(fmaxf(p, TOLERANCE));
    } else if (y == 0.0f) {
        return -log(fmaxf(1 - p, 1 - TOLERANCE));
    }
}

float sigmoid(float x) {
    if (x <= -35.0f) {
        return 0.000000000000001f;
    } else if (x >= 35.0f) {
        return 0.999999999999999f;
    }

    return 1.0f / (1.0f + exp(-x));
}

float ftrl_predict(int *values, int len, ftrl_model *model) {
    ftrl_params params = model->params;
    model->w_intercept = calculate_w(model->z_intercept, model->n_intercept, params);
    float wtx = model->w_intercept;

    float *n = model->n;
    float *z = model->z;
    float *w = model->w;

    for (int k = 0; k < len; k++) {
        int i = values[k];
        w[i] = calculate_w(z[i], n[i], params);
        wtx = wtx + w[i];
    }

    return wtx;
}

float ftrl_fit(int *values, int len, float y, ftrl_model *model) {
    ftrl_params params = model->params;
    float wtx = ftrl_predict(values, len, model);
    float p = sigmoid(wtx);
    float grad = p - y;

    float sigma_intercept = calculate_sigma(model->n_intercept, grad, params.alpha);
    model->z_intercept = model->z_intercept + grad - sigma_intercept * model->z_intercept;
    model->n_intercept = model->n_intercept + grad * grad;

    float *n = model->n;
    float *z = model->z;
    float *w = model->w;

    for (int k = 0; k < len; k++) {
        int i = values[k];
        float sigma = calculate_sigma(n[i], grad, params.alpha);
        z[i] = z[i] + grad - sigma * w[i];
        n[i] = n[i] + grad * grad;
    }

    return log_loss(y, p);
}

float ftrl_fit_batch(csr_binary_matrix &X, float *target, int num_examples,
                     ftrl_model *model, bool shuffle) {
    int* values = X.columns;
    int* indptr = X.indptr;

    int *idx = new int[num_examples];
    for (int i = 0; i < num_examples; i++) {
        idx[i] = i;
    }

    if (shuffle) {
        random_shuffle(&idx[0], &idx[num_examples]);
    }

    float loss_total = 0.0f;

    #if defined USEOMP
    #pragma omp parallel for schedule(static) reduction(+: loss_total)
    #endif

    for (int id = 0; id < num_examples; id++) {
        int i = idx[id];

        float y = target[i];
        int *x = &values[indptr[i]];
        int len_x = indptr[i + 1] - indptr[i];

        float loss = ftrl_fit(x, len_x, y, model);
        loss_total = loss_total + loss;
    }

    return loss_total / num_examples;
}


void ftrl_predict_batch(csr_binary_matrix &X, ftrl_model *model, float* result) {
    int n = X.num_examples;
    int* values = X.columns;
    int* indptr = X.indptr;

    #if defined USEOMP
    #pragma omp parallel for schedule(static)
    #endif

    for (int i = 0; i < n; i++) {
        int len_x = indptr[i + 1] - indptr[i];
        int *x = &values[indptr[i]];
        result[i] = ftrl_predict(x, len_x, model);
    }
}


float* zero_float_vector(int size) {
    float* result = new float[size];
    memset(result, 0.0f, size * sizeof(float));
    return result;
}

ftrl_model ftrl_init_model(ftrl_params &params, int num_features) {
    ftrl_model model;

    model.n_intercept = 0.0f;
    model.z_intercept = 0.0f;
    model.w_intercept = 0.0f;

    model.num_features = num_features;
    model.n = zero_float_vector(num_features);
    model.z = zero_float_vector(num_features);
    model.w = zero_float_vector(num_features);

    model.params = params;

    return model;
}

void ftrl_model_cleanup(ftrl_model *model) {
    delete[] model->n;
    delete[] model->z;
    delete[] model->w;
}

void ftrl_save_model(char *path, ftrl_model *model) {
    FILE *f = fopen(path, "wb");

    int n = model->num_features;
    fwrite(&n, sizeof(int), 1, f);

    fwrite(model->n, sizeof(float), n, f);
    fwrite(model->z, sizeof(float), n, f);
    fwrite(model->w, sizeof(float), n, f);

    fwrite(&model->n_intercept, sizeof(int), 1, f);
    fwrite(&model->z_intercept, sizeof(int), 1, f);
    fwrite(&model->w_intercept, sizeof(int), 1, f);

    fwrite(&model->params, sizeof(ftrl_params), 1, f);

    fclose(f);
}

ftrl_model ftrl_load_model(char *path) {
    ftrl_model model;

    FILE *f = fopen(path, "rb");
    int n = 0;
    fread(&n, sizeof(int), 1, f);
    model.num_features = n;

    model.n = new float[n];
    model.z = new float[n];
    model.w = new float[n];

    fread(model.n, sizeof(float), n, f);
    fread(model.z, sizeof(float), n, f);
    fread(model.w, sizeof(float), n, f);

    fread(&model.n_intercept, sizeof(int), 1, f);
    fread(&model.z_intercept, sizeof(int), 1, f);
    fread(&model.w_intercept, sizeof(int), 1, f);

    fread(&model.params, sizeof(ftrl_params), 1, f);

    fclose(f);

    return model;
}
