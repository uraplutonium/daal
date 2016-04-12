/* file: training_parameter.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <jni.h>
#include "neural_networks/training/JTrainingParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingParameter
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingParameter_cInit
  (JNIEnv *env, jobject thisObj, jlong prec)
{
    if (prec == 0)
    {
        return (jlong)(new training::Parameter<double>());
    } else
    {
        return (jlong)(new training::Parameter<float>());
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingParameter
 * Method:    cGetBatchSize
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingParameter_cGetBatchSize
  (JNIEnv *env, jobject thisObj, jlong cParameter, jlong prec)
{
    if (prec == 0)
    {
        return (((training::Parameter<double> *)cParameter))->batchSize;
    } else
    {
        return (((training::Parameter<float> *)cParameter))->batchSize;
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingParameter
 * Method:    cSetBatchSize
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingParameter_cSetBatchSize
  (JNIEnv *env, jobject thisObj, jlong cParameter, jlong prec, jlong batchSize)
{
    if (prec == 0)
    {
        (((training::Parameter<double> *)cParameter))->batchSize = batchSize;
    } else
    {
        (((training::Parameter<float> *)cParameter))->batchSize = batchSize;
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingParameter
 * Method:    cGetNIterations
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingParameter_cGetNIterations
  (JNIEnv *env, jobject thisObj, jlong cParameter, jlong prec)
{
    if (prec == 0)
    {
        return (((training::Parameter<double> *)cParameter))->nIterations;
    } else
    {
        return (((training::Parameter<float> *)cParameter))->nIterations;
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingParameter
 * Method:    cSetNIterations
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingParameter_cSetNIterations
  (JNIEnv *env, jobject thisObj, jlong cParameter, jlong prec, jlong nIterations)
{
    if (prec == 0)
    {
        (((training::Parameter<double> *)cParameter))->nIterations = nIterations;
    } else
    {
        (((training::Parameter<float> *)cParameter))->nIterations = nIterations;
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingParameter
 * Method:    cSetOptimizationSolver
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingParameter_cSetOptimizationSolver
  (JNIEnv *env, jobject thisObj, jlong cParameter, jlong prec, jlong optAddr)
{
    if (prec == 0)
    {
        services::SharedPtr<optimization_solver::sgd::Batch<double> > opt =
            *((services::SharedPtr<optimization_solver::sgd::Batch<double> > *)optAddr);
        (((training::Parameter<double> *)cParameter))->optimizationSolver = opt;
    } else
    {
        services::SharedPtr<optimization_solver::sgd::Batch<float> > opt =
            *((services::SharedPtr<optimization_solver::sgd::Batch<float> > *)optAddr);
        (((training::Parameter<float> *)cParameter))->optimizationSolver = opt;
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingParameter
 * Method:    cGetOptimizationSolver
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingParameter_cGetOptimizationSolver
  (JNIEnv *env, jobject thisObj, jlong cParameter, jlong prec)
{
    if (prec == 0)
    {
        SharedPtr<optimization_solver::sgd::Batch<double> > *opt =
            new SharedPtr<optimization_solver::sgd::Batch<double> >
                ((((training::Parameter<double> *)cParameter))->optimizationSolver);
        return (jlong)opt;
    } else
    {
        SharedPtr<optimization_solver::sgd::Batch<float> > *opt =
            new SharedPtr<optimization_solver::sgd::Batch<float> >
                ((((training::Parameter<float> *)cParameter))->optimizationSolver);
        return (jlong)opt;
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingParameter
 * Method:    cSetObjectiveFunction
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingParameter_cSetObjectiveFunction
  (JNIEnv *env, jobject thisObj, jlong cParameter, jlong prec, jlong objAddr)
{
    if (prec == 0)
    {
        services::SharedPtr<optimization_solver::mse::Batch<double> > obj =
            *((services::SharedPtr<optimization_solver::mse::Batch<double> > *)objAddr);
        (((training::Parameter<double> *)cParameter))->objectiveFunction = obj;
    } else
    {
        services::SharedPtr<optimization_solver::mse::Batch<float> > obj =
            *((services::SharedPtr<optimization_solver::mse::Batch<float> > *)objAddr);
        (((training::Parameter<float> *)cParameter))->objectiveFunction = obj;
    }
}
