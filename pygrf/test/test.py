# from pygrf import *

import numpy as np
import pandas as pd

p = 6
n = 1000
ticks = 101
Xtest = np.zeros((ticks, p))
xvals = np.linspace(-1, 1, ticks)
Xtest[:, 0] = xvals
X = 2 * np.random.rand(n, p) - 1
Y = (X[:, 0] > 0) + 2 * np.random.randn(n)

# R:
# forest = regression_forest(X, Y, num.trees = 1000, ci.group.size = 4)

#
#
#
# sample_fraction_1 = 0.5
# honesty_fraction_1 = 0.25
#
# n = 16
# k = 10
#
# X = np.random.rand(n, k)
# X = np.random.rand(n)
