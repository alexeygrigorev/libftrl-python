#ifndef FTRL_LIBRARY_H
#define FTRL_LIBRARY_H


extern "C" {

struct csr_binary_matrix {
    int *columns;
    int *indptr;
    int num_examples;
};

struct ftrl_params {
    float alpha;
    float beta;
    float l1;
    float l2;
};

struct ftrl_model {
    float n_intercept;
    float z_intercept;
    float w_intercept;

    float *n;
    float *z;
    float *w;

    int num_features;
};

ftrl_model ftrl_init_model(ftrl_params &params, int num_features);

void ftrl_model_cleanup(ftrl_model *model);

float ftrl_fit(int *values, int len, float y, ftrl_params &params, ftrl_model *model);

float ftrl_fit_batch(csr_binary_matrix &X, float* target, int num_examples,
                     ftrl_params &params, ftrl_model *model, bool shuffle);

float ftrl_predict(int *values, int len, ftrl_params &params, ftrl_model *model);

void ftrl_predict_batch(csr_binary_matrix &X, ftrl_params &params, ftrl_model *model,
                        float* result);

void ftrl_save_model(ftrl_model *model, char *path);

ftrl_model ftrl_load_model(char *path);

};

#endif