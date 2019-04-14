from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

ctypedef unsigned int uint


# Misc. lib std
cdef extern from '<iostream>' namespace 'std':
    cdef cppclass ostream:
        pass
    cdef cppclass istream:
        pass


cdef extern from '<sstream>' namespace 'std':
    cdef cppclass stringstream(ostream, istream):
        string str()
        istream str(string)


# Data()
cdef extern from 'grf/core/src/commons/Data.h':
    cdef cppclass Data:
        Data()
        void set_outcome_index(size_t i)
        void sort()

cdef extern from 'grf/core/src/commons/Data.cpp':
    pass


# DefaultData()
cdef extern from 'grf/core/src/commons/DefaultData.h':
    cdef cppclass DefaultData(Data):
        DefaultData(double* data, size_t num_rows, size_t num_cols)

cdef extern from 'grf/core/src/commons/DefaultData.cpp':
    pass


# SparseData()
cdef extern from 'grf/core/src/commons/SparseData.h':
    cdef cppclass SparseData(Data):
        SparseData(double* data, size_t num_rows, size_t num_cols)

cdef extern from 'grf/core/src/commons/SparseData.cpp':
    pass


# Forest()
cdef extern from 'grf/core/src/forest/Forest.h':
    cdef cppclass Forest:
        Forest(Forest&)

        Forest(const vector[shared_ptr[Tree]]& trees,
               size_t num_variables, size_t ci_group_size)

cdef extern from 'grf/core/src/forest/Forest.cpp':
    pass


# ForestSerializer()
cdef extern from 'grf/core/src/serialization/ForestSerializer.h':
    cdef cppclass ForestSerializer:
        void serialize(ostream& stream, const Forest& forest)
        Forest deserialize(istream& stream)

cdef extern from 'grf/core/src/serialization/ForestSerializer.cpp':
    pass


# ForestOptions()
cdef extern from 'grf/core/src/forest/ForestOptions.h':
    cdef cppclass ForestOptions:
        ForestOptions(ForestOptions&)
        ForestOptions(uint num_trees, size_t ci_group_size, double sample_fraction,
                      uint mtry,
                      uint min_node_size, bool honesty, double honesty_fraction,
                      double alpha, double imbalance_penalty, uint num_threads,
                      uint random_seed, const vector[size_t]& sample_clusters,
                      uint sample_per_cluster)

cdef extern from 'grf/core/src/forest/ForestOptions.cpp':
    pass


# ForestTrainer()
cdef extern from 'grf/core/src/forest/ForestTrainer.h':
    cdef cppclass ForestTrainer:
        ForestTrainer(ForestTrainer&)
        ForestTrainer(shared_ptr[RelabelingStrategy] relabeling_strategy,
                      shared_ptr[SplittingRuleFactory] splitting_rule_factory,
                      shared_ptr[OptimizedPredictionStrategy] prediction_strategy)

        const Forest train(const Data* data, const ForestOptions& options) const


cdef extern from 'grf/core/src/forest/ForestTrainer.cpp':
    pass


# ForestTrainers()
cdef extern from 'grf/core/src/forest/ForestTrainers.h':
    cdef cppclass ForestTrainers:
        @staticmethod
        ForestTrainer regression_trainer()

cdef extern from 'grf/core/src/forest/ForestTrainers.cpp':
    pass


# ForestPredictor()
cdef extern from 'grf/core/src/forest/ForestPredictor.h':
    cdef cppclass ForestPredictor:
        ForestPredictor(ForestPredictor&)
        vector[Prediction] predict(const Forest& forest,
                                   Data* train_data,
                                   Data*,
                                   bool estimate_variance) const
        vector[Prediction] predict_oob(const Forest& forest,
                                  Data* train_data,
                                  bool estimate_variance) const

cdef extern from 'grf/core/src/forest/ForestPredictor.cpp':
    pass


# ForestPredictors()
cdef extern from 'grf/core/src/forest/ForestPredictors.h':
    cdef cppclass ForestPredictors:
        @staticmethod
        ForestPredictor regression_predictor(uint num_threads)
        @staticmethod
        ForestPredictor local_linear_predictor(
                  uint num_threads, vector[double] lambdas, bool weighted_penalty,
                  vector[size_t] linear_correction_variables)

cdef extern from 'grf/core/src/forest/ForestPredictors.cpp':
    pass


# Tree()
cdef extern from 'grf/core/src/tree/Tree.h':
    cdef cppclass Tree:
        pass

cdef extern from 'grf/core/src/tree/Tree.cpp':
    pass


# Prediction()
cdef extern from 'grf/core/src/prediction/Prediction.h':
    cdef cppclass Prediction:
        const size_t size() const
        const vector[double]& get_predictions() const
        const vector[double]& get_variance_estimates() const
        const vector[double]& get_error_estimates() const
        const bool contains_variance_estimates() const
        const bool contains_error_estimates() const

cdef extern from 'grf/core/src/prediction/Prediction.cpp':
    pass

# RelabelingStrategy()
cdef extern from 'grf/core/src/relabeling/RelabelingStrategy.h':
    cdef cppclass RelabelingStrategy:
        pass


# SplittingRuleFactory()
cdef extern from 'grf/core/src/splitting/factory/SplittingRuleFactory.h':
    cdef cppclass SplittingRuleFactory:
        pass


# OptimizedPredictionStrategy()
cdef extern from 'grf/core/src/prediction/OptimizedPredictionStrategy.h':
    cdef cppclass OptimizedPredictionStrategy:
        pass
