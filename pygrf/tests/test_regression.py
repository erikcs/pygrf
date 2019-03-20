import numpy as np
import numpy.testing as nt

import pygrf as grf

def test_simple_predict():
    n = 1000
    np.random.seed(42)
    X = np.random.randn(n, 5)
    y = np.random.randn(n)
    X = np.asfortranarray(X)

    f = grf.RegressionForest(seed=1)
    f.fit(X, y)
    f.predict()
    f.predict(X)
