import cython
import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref

from grfheaders cimport *
cdef int SMAX = 2147483647

# Utils
# -----------------------------------------------------------------------------

# (Unused) Going from C++ to Python
@cython.internal
cdef class _Serializer:

  cdef ForestSerializer* serializer

  def __cinit__(self):
    self.serializer = new ForestSerializer()

  cdef string serialize(self, const Forest& forest):
    cdef stringstream stream
    self.serializer.serialize(stream, forest)

    return stream.str()

  def __dealloc__(self):
    del self.serializer


# (Unused) Going from Python to C++
@cython.internal
cdef class _DeSerialiser:

  cdef Forest* forest

  def __cinit__(self):
    pass

  cdef deserialize(self, string serialized):
    cdef stringstream stream
    stream.str(serialized)
    s = _Serializer()
    self.forest = new Forest(s.serializer.deserialize(stream))

  def __dealloc__(self):
    del self.forest


@cython.internal
cdef class _Data:

  cdef Data* data
  cdef uint n
  cdef uint p
  cdef double[::1, :] fdata

  def __cinit__(self, double[::1, :] X, double[::1] y=None):
    cdef uint n = X.shape[0]
    cdef uint p = X.shape[1]
    cdef uint ncols

    if y is not None:
      assert len(y) == n
      self.fdata = np.asfortranarray(np.c_[X, y])
      ncols = p + 1
      self.data = new DefaultData(&self.fdata[0, 0], n, ncols)
      self.data.set_outcome_index(p)
      self.data.sort()
    else:
      self.fdata = np.asfortranarray(X) #TODO: doesn't have to be copied
      ncols = p
      self.data = new DefaultData(&self.fdata[0, 0], n, ncols)

    self.n = n
    self.p = p

  def __dealloc__(self):
      del self.data


@cython.internal
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
    # long[::1] clusters=np.empty(0, dtype=int),
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
@cython.internal
cdef class _RegressionTrainer:

  cdef ForestTrainer* trainer
  cdef Forest* forest

  def __cinit__(self):
    self.trainer = new ForestTrainer(ForestTrainers.regression_trainer())

  cdef train(self, Data* data, const ForestOptions& options):
    self.forest = new Forest(self.trainer.train(data, options))

  def __dealloc__(self):
    del self.trainer
    del self.forest


@cython.internal
cdef class _RegressionPredictor:

  cdef ForestPredictor* predictor
  cdef vector[Prediction]* predictions

  def __cinit__(self, kind, uint num_threads=0):
      self.predictor = new ForestPredictor(
        ForestPredictors.regression_predictor(num_threads))

  cdef predict(self, Data* train_data, Data* data, const Forest& forest,
               bool estimate_variance=False):
    self.predictions = new vector[Prediction](
      self.predictor.predict(forest, train_data, data, estimate_variance))

  cdef predict_oob(self, Data* train_data, Data* data, const Forest& forest,
                   bool estimate_variance=False):
    self.predictions = new vector[Prediction](
       self.predictor.predict_oob(forest, train_data, estimate_variance))


  def bla(self):
   cdef size_t prediction_length = self.predictions.at(0).size()
   cdef size_t N = self.predictions.size()
   cdef double[::1, :] X = np.empty((N, prediction_length))
   cdef vector[double]* pred

   for i in range(N):
     for j in range(prediction_length):
       X[i, j] =  self.predictions.at(i).get_predictions().at(j)

   return X

  def predict_oob(self):
    pass

  @property
  def predictions(self):
    cdef double[:, ::1] X
    return X

  def __dealloc__(self):
    del self.predictor
    del self.predictions


@cython.internal
cdef class _LocLinearRegressionPredictor:
  pass


cdef class RegressionForest:

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
  cdef np.ndarray clusters
  cdef uint samples_per_cluster
  cdef bool tune_parameters
  cdef uint num_fit_trees
  cdef uint num_fit_reps
  cdef uint num_optimize_reps

  cdef _Data traindata
  cdef _RegressionTrainer trainer
  cdef _ForestOptions options

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

    self.sample_fraction = sample_fraction
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
    self.clusters = clusters
    self.samples_per_cluster = samples_per_cluster
    self.tune_parameters = tune_parameters
    self.num_fit_trees = num_fit_trees
    self.num_fit_reps = num_fit_reps
    self.num_optimize_reps = num_optimize_reps

  def fit(self, double[::1, :] X, double[::1] y):
    self.traindata = _Data(X, y)

    if self.mtry < 0:
      self.mtry = min(np.ceil(np.sqrt(self.traindata.p)) + 20, self.traindata.p)

    self._tune_regression_forest()

    self.trainer = _RegressionTrainer()
    self.options = _ForestOptions(self.num_trees, self.ci_group_size,
                    self.sample_fraction,self.mtry, self.min_node_size,
                    self.honesty, self.honesty_fraction, self.alpha,
                    self.imbalance_penalty, self.num_threads,
                    self.seed, self.clusters, self.samples_per_cluster)

    self.trainer.train(self.traindata.data, deref(self.options.options))

    return self

  def _fit(self):
    pass

  # ToTune min_node_size, sample_fraction, mtry, alpha, imbalance_penalty
  def _tune_regression_forest(self):
    if self.tune_parameters:
      pass

  def predict(self, double[::1, :] data=None, uint num_threads=0, bool estimate_variance=False):
    rp = _RegressionPredictor(0)
    d = _Data(data)
    # ds = _DeSerialiser(self.serialized)
    # rp.predict(self._data.data, d.data, deref(ds.forest), False)
    rp.predict(self.traindata.data, d.data, deref(self.trainer.forest), False)

    return rp.bla()



# Quantile
# -----------------------------------------------------------------------------

# Instrumental
# -----------------------------------------------------------------------------
