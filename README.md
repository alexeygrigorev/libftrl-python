
## Pre-requisites 

Dependensies:

* It needs: `numpy`, `scipy` and open mp
* If you use anaconda, it already has  `numpy`, `scipy`
* to install `GOMP_4.0` for anaconda, use `conda install libgcc`


## Building

    # building the library
    mkdir build
    cd build
    cmake ..
    make 

    cd ..

    # moving the library to the python package
    mv build/libftrl.so ftrl/

    # installing the package
    python setup.py install


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

