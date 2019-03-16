import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref

from grfheaders cimport *
cdef int SMAX = 2147483647

# Utils
# -----------------------------------------------------------------------------
cdef class _Serializer:

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
  cdef uint n
  cdef uint p

  def __cinit__(self, double[:, ::1] X, double[::1] y):
    cdef uint n = X.shape[0]
    cdef uint p = X.shape[1]
    assert len(y) == n
    cdef double[::1, :] Xy = np.asfortranarray(np.c_[X, y])

    self.n = n
    self.p = p

    self.data = new DefaultData(&Xy[0, 0], n, p + 1)
    self.data.set_outcome_index(p)
    self.data.sort()

  def __dealloc__(self):
      del self.data


cdef class _ForestOptions:

  cdef ForestOptions* options

  def __cinit__(self,
    uint num_trees=2000,
    size_t ci_group_size=2,
    double sample_fraction=0.5,
    int mtry=-1,
    uint min_node_size=5,
    bool honesty=True,
    double honesty_fraction=0.5,
    double alpha=0.05,
    double imbalance_penalty=0,
    uint num_threads=0,
    uint seed=np.random.randint(0, SMAX, 1)[0],
    np.ndarray[dtype=np.intp_t, ndim=1] clusters=np.empty(0, dtype=int),
    uint samples_per_cluster=0):

    assert 0 < alpha < 0.25
    assert 0 <= sample_fraction <= 1
    assert imbalance_penalty >= 0


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
                  <uint> seed,
                  <const vector[size_t]&> clusters,
                  <uint> samples_per_cluster))

  def __dealloc__(self):
    del self.options


# Regression
# -----------------------------------------------------------------------------
cdef class _RegressionTrainer:

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

  cdef double sample_fraction
  cdef int mtry
  cdef uint num_trees
  cdef uint num_threads
  cdef uint min_node_size
  cdef bool honesty
  cdef double honesty_fraction
  cdef size_t ci_group_size
  cdef double alpha
  cdef double imbalance_penalty
  cdef bool compute_oob_predictions
  cdef uint seed
  cdef bool tune_parameters
  cdef uint num_fit_trees
  cdef uint num_fit_reps
  cdef uint num_optimize_reps

  # TOTUNE: min_node_size, sample_fraction, mtry, imbalance_penalty
  def __cinit__(self,
    sample_fraction=0.5,
    mtry=-1,
    num_trees=2000,
    num_threads=0,
    min_node_size=5,
    honesty=True,
    honesty_fraction=0.5,
    ci_group_size=2,
    alpha=0.05,
    imbalance_penalty=0,
    compute_oob_predictions=True,
    seed=np.random.randint(0, SMAX, 1)[0],
    clusters=np.empty(0, dtype=int),
    samples_per_cluster=0,
    tune_parameters=False,
    num_fit_trees=10,
    num_fit_reps=100,
    num_optimize_reps=1000):

    self.sample_fraction=sample_fraction
    self.mtry = mtry
    self.num_trees = num_trees
    self.num_threads = num_threads
    self.min_node_size = min_node_size
    self.honesty = honesty
    self.honesty_fraction = honesty_fraction
    self.ci_group_size = ci_group_size
    self.alpha = alpha
    self.imbalance_penalty = imbalance_penalty
    self.compute_oob_predictions = compute_oob_predictions
    self.seed = seed
    self.tune_parameters = tune_parameters
    self.num_fit_trees = num_fit_trees
    self.num_fit_reps = num_fit_reps
    self.num_optimize_reps = num_optimize_reps

  def fit(self,
    double[:, ::1] X,
    double[::1] y,
    np.ndarray[dtype=np.intp_t, ndim=1] clusters=np.empty(0, dtype=int),
    uint samples_per_cluster=0):

    d = _Data(X, y)
    _tune_regression_forest(self)

    if self.mtry < 0:
      self.mtry = min(np.ceil(np.sqrt(d.p)) + 20, d.p)

    trainer = _RegressionTrainer()
    options = _ForestOptions(self.num_trees, self.ci_group_size,
                    self.sample_fraction,self.mtry, self.min_node_size,
                    self.honesty, self.honesty_fraction, self.alpha,
                    self.imbalance_penalty, self.num_threads,
                    self.seed, clusters, samples_per_cluster)
    s = _Serializer()
    d = _Data(X, y)
    trainer.train(d.data, deref(options.options))
    self.serialized = s.serialize(deref(trainer.forest))

  def _fit(self):
    pass

  def _tune_regression_forest(self):
    if self.tune_parameters:
      pass

  def blabla(self):
    return self.serialized


# Quantile
# -----------------------------------------------------------------------------

# Instrumental
# -----------------------------------------------------------------------------
