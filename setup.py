import numpy
import os
import sys

from setuptools import setup
from distutils.core import Extension
from Cython.Build import cythonize


if sys.platform == 'darwin': # Mac
    COMPILE_ARGS = ['-std=c++11', '-stdlib=libc++', '-Wall', '-O2', '-pthread']
    LINK_ARGS = ['-stdlib=libc++']
else: # Linux
    COMPILE_ARGS = ['-std=c++11', '-lstdc++', '-Wall', '-O2', '-pthread']
    LINK_ARGS = ['-lstdc++', '-pthread']

setup_dir = os.path.abspath(os.path.dirname(__file__))
INCLUDE_DIRS = [os.path.join(setup_dir, 'pygrf/grf/core/src')]
INCLUDE_DIRS.append(os.path.join(setup_dir, 'pygrf/grf/core/third_party'))
INCLUDE_DIRS.append(numpy.get_include())

SOURCES = ['pygrf/grf.pyx']

GRF_SOURCES = ['pygrf/grf/core/src/relabeling/NoopRelabelingStrategy.cpp',
               'pygrf/grf/core/src/relabeling/CustomRelabelingStrategy.cpp',
               'pygrf/grf/core/src/relabeling/QuantileRelabelingStrategy.cpp',
               'pygrf/grf/core/src/relabeling/InstrumentalRelabelingStrategy.cpp',
               'pygrf/grf/core/src/prediction/RegressionPredictionStrategy.cpp',
               'pygrf/grf/core/src/prediction/InstrumentalPredictionStrategy.cpp',
               'pygrf/grf/core/src/prediction/PredictionValues.cpp',
               'pygrf/grf/core/src/prediction/ObjectiveBayesDebiaser.cpp',
               'pygrf/grf/core/src/prediction/collector/TreeTraverser.cpp',
               'pygrf/grf/core/src/prediction/collector/SampleWeightComputer.cpp',
               'pygrf/grf/core/src/commons/utility.cpp',
               'pygrf/grf/core/src/commons/SparseData.cpp',
               'pygrf/grf/core/src/tree/TreeOptions.cpp',
               'pygrf/grf/core/src/tree/TreeTrainer.cpp',
               'pygrf/grf/core/src/sampling/RandomSampler.cpp',
               'pygrf/grf/core/src/sampling/SamplingOptions.cpp',
               'pygrf/grf/core/src/splitting/RegressionSplittingRule.cpp',
               'pygrf/grf/core/src/splitting/ProbabilitySplittingRule.cpp',
               'pygrf/grf/core/src/splitting/InstrumentalSplittingRule.cpp',
               'pygrf/grf/core/src/splitting/factory/RegressionSplittingRuleFactory.cpp',
               'pygrf/grf/core/src/splitting/factory/ProbabilitySplittingRuleFactory.cpp',
               'pygrf/grf/core/src/splitting/factory/InstrumentalSplittingRuleFactory.cpp',
               'pygrf/grf/core/src/serialization/TreeSerializer.cpp',
               'pygrf/grf/core/src/serialization/PredictionValuesSerializer.cpp',
               ]

ext = Extension(
        'pygrf.ext',
        language='c++',
        sources=SOURCES + GRF_SOURCES,
        extra_compile_args=COMPILE_ARGS,
        extra_link_args=LINK_ARGS,
        include_dirs=INCLUDE_DIRS
    )

setup(
    name='pygrf',
    version='0.0.1',
    ext_modules=cythonize(ext, compiler_directives={'language_level' : 3})
)
