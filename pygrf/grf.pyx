import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref

from grfheaders cimport *


# Utils
# -----------------------------------------------------------------------------
cdef class _Serialize:

  cdef ForestSerializer* serializer

  def __cinit__(self):
    self.serializer = new ForestSerializer()

  # Going from C++ to Python
  cdef string serialize(self, const Forest& forest):
    cdef stringstream stream
    cdef string contents

    self.serializer.serialize(stream, forest)
    contents = stream.str()

    return contents

  # Going from Python to C++
  def deserialize(self):
    pass

  def __dealloc__(self):
    del self.serializer


cdef class _Data:

  cdef Data* data

  def __cinit__(self, np.ndarray X, np.ndarray y):
    cdef uint n = X.shape[0]
    cdef uint p = X.shape[1]
    assert len(y) == n
    cdef np.ndarray Xy = np.asfortranarray( np.c_[X, y] )

    self.data = new DefaultData(<double*> Xy.data, n, p + 1)
    self.data.set_outcome_index(p)
    self.data.sort() # ?

  def __dealloc__(self):
      del self.data


cdef class _ForestOptions:

  cdef ForestOptions* options

  def __cinit__(self,
    uint num_trees,
    size_t ci_group_size,
    double sample_fraction,
    uint mtry,
    uint min_node_size,
    bool honesty,
    double honesty_fraction,
    double alpha,
    double imbalance_penalty,
    uint num_threads,
    uint random_seed,
    np.ndarray[dtype=np.intp_t, ndim=1] sample_clusters,
    uint sample_per_cluster):

    self.options = new ForestOptions(
     ForestOptions(<uint> num_trees,
                  <size_t> ci_group_size,
                  <double> sample_fraction,
                  <uint> mtry,
                  <uint> min_node_size,
                  <bool> honesty,
                  <double> honesty_fraction,
                  <double> alpha,
                  <double> imbalance_penalty,
                  <uint> num_threads,
                  <uint> random_seed,
                  <const vector[size_t]&> sample_clusters,
                  <uint> sample_per_cluster))

  def __dealloc__(self):
    del self.options


# Regression
# -----------------------------------------------------------------------------
cdef class _RegressionTrain:

  cdef const ForestTrainer* trainer
  cdef const Forest* forest

  def __cinit__(self):
    self.trainer = new ForestTrainer(ForestTrainers.regression_trainer())

  cdef train(self, Data* data, ForestOptions options):
    self.forest = new Forest(self.trainer.train(data, options))

  def __dealloc__(self):
    del self.trainer
    del self.forest


cdef class RegressionForest:

  cdef bytes serialized

  def __cinit__(self, X, y, sample_fraction=0.5, mtry=None, num_trees=2000, num_threads=None,
    min_node_size=None, honesty=True, honesty_fraction=None, ci_group_size=2,
    alpha=None, imbalance_penalty=None, compute_oob_predictions=True,
    seed=None, clusters=None, samples_per_cluster=None, tune_parameters=False,
    num_fit_trees=10, num_fit_reps=100, num_optimize_reps=1000):

    trainer = _RegressionTrain()
    options = _ForestOptions(num_trees, ci_group_size, sample_fraction, mtry,
                    min_node_size, honesty, honesty_fraction,
                    alpha, imbalance_penalty, num_threads,
                    seed, clusters, samples_per_cluster)
    ser = _Serialize()
    d = _Data(X, y)
    trainer.train(d.data, deref(options.options))
    self.serialized = ser.serialize(deref(trainer.forest))

  def blabla(self):
    return self.serialized


# Quantile
# -----------------------------------------------------------------------------

# Instrumental
# -----------------------------------------------------------------------------
