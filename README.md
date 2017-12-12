
# FTRL-Proximal 

This is an implementation of the FTRL-Proximal algorithm in C with python bindings. FTRL-Proximal is an algorithm for online learning which is quite successful in solving sparse problems. The implementation is based on the algorithm from the ["Ad Click Prediction: a View from the Trenches"](https://research.google.com/pubs/pub41159.html) paper.

Some of the features:

* Uses Open MP to parallelize training, and hence is very fast
* The python code can operate directly on scipy CSR matrices

## Pre-requisites 

Dependensies:

* It needs: `numpy`, `scipy` and open mp
* If you use anaconda, it already has  `numpy`, `scipy`
* to install `GOMP_4.0` for anaconda, use `conda install libgcc`


## Building

    cmake . && make
    mv libftrl.so ftrl/
    python setup.py install

If you don't have `cmake`, it's easy to install:

    mkdir cmake && cd cmake
    wget https://cmake.org/files/v3.10/cmake-3.10.0-Linux-x86_64.sh
    bash cmake-3.10.0-Linux-x86_64.sh --skip-license
    export CMAKE_HOME=`pwd`
    export PATH=$PATH:$CMAKE_HOME/bin


## Example

    import numpy as np
    import scipy.sparse as sp

    from sklearn.metrics import roc_auc_score

    import ftrl

    X = [
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],   
    ]

    X = sp.csr_matrix(X)
    y = np.array([1, 1, 1, 0, 0], dtype='float32')
    
    model = ftrl.FtrlProximal(alpha=1, beta=1, l1=10, l2=0)

    # make 10 passes over the data
    for i in range(10):
        model.fit(X, y)
        y_pred = model.predict(X)
        auc = roc_auc_score(y, y_pred)
        print('%02d: %.5f' % (i + 1, auc))


## Use case 

This library was used for the [Criteo Ad Placement Challenge](https://www.crowdai.org/challenges/nips-17-workshop-criteo-ad-placement-challenge/leaderboards) and showed very competitive performance. You can have a look at the solution here: https://github.com/alexeygrigorev/nips-ad-placement-challenge

In particular, it performed significantly faster than sklearn's Logistic Regression (a wrapper for LIBLINEAR):

- skearn: 1.2 hours to train, auc=0.734
- libftrl-python: 2 minutes to train, auc=0.734
