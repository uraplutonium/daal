/* file: baseparameter.cpp */
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

#include "daal.h"

#include "optimization_solver/sgd/JBaseParameter.h"

#include "common_defines.i"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms::optimization_solver;

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_BaseParameter
 * Method:    cSetFunction
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_BaseParameter_cSetFunction
(JNIEnv *, jobject, jlong parAddr, jlong cFunction)
{
    sgd::BaseParameter *parameterAddr = (sgd::BaseParameter *)parAddr;
    SharedPtr<optimization_solver::sum_of_functions::Batch> objectiveFunction =
        staticPointerCast<optimization_solver::sum_of_functions::Batch, AlgorithmIface>
        (*(SharedPtr<AlgorithmIface> *)cFunction);
    parameterAddr->function = objectiveFunction;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_BaseParameter
 * Method:    cSetNIterations
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_BaseParameter_cSetNIterations
(JNIEnv *, jobject, jlong parAddr, jlong nIterations)
{
    ((sgd::BaseParameter *)parAddr)->nIterations = nIterations;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_BaseParameter
 * Method:    cGetNIterations
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_BaseParameter_cGetNIterations
(JNIEnv *, jobject, jlong parAddr)
{
    return ((sgd::BaseParameter *)parAddr)->nIterations;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_BaseParameter
 * Method:    cSetAccuracyThreshold
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_BaseParameter_cSetAccuracyThreshold
(JNIEnv *, jobject, jlong parAddr, jdouble accuracyThreshold)
{
    ((sgd::BaseParameter *)parAddr)->accuracyThreshold = accuracyThreshold;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_BaseParameter
 * Method:    cGetAccuracyThreshold
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_BaseParameter_cGetAccuracyThreshold
(JNIEnv *, jobject, jlong parAddr)
{
    return ((sgd::BaseParameter *)parAddr)->accuracyThreshold;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_BaseParameter
 * Method:    cSetBatchIndices
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_BaseParameter_cSetBatchIndices
(JNIEnv *, jobject, jlong parAddr, jlong cBatchIndices)
{
    SharedPtr<SerializationIface> *ntShPtr = (SharedPtr<SerializationIface> *)cBatchIndices;
    ((sgd::BaseParameter *)parAddr)->batchIndices = staticPointerCast<NumericTable, SerializationIface>(*ntShPtr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_BaseParameter
 * Method:    cGetBatchIndices
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_BaseParameter_cGetBatchIndices
(JNIEnv *, jobject, jlong parAddr)
{
    SharedPtr<NumericTable> *ntShPtr = new SharedPtr<NumericTable>();
    *ntShPtr = ((sgd::BaseParameter *)parAddr)->batchIndices;
    return (jlong)ntShPtr;
}


/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_BaseParameter
 * Method:    cSetLearningRateSequence
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_BaseParameter_cSetLearningRateSequence
(JNIEnv *, jobject, jlong parAddr, jlong cLearningRateSequence)
{
    SharedPtr<SerializationIface> *ntShPtr = (SharedPtr<SerializationIface> *)cLearningRateSequence;
    ((sgd::BaseParameter *)parAddr)->learningRateSequence = staticPointerCast<NumericTable, SerializationIface>(*ntShPtr);
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_BaseParameter
 * Method:    cGetLearningRateSequence
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_BaseParameter_cGetLearningRateSequence
(JNIEnv *, jobject, jlong parAddr)
{
    SharedPtr<NumericTable> *ntShPtr = new SharedPtr<NumericTable>();
    *ntShPtr = ((sgd::BaseParameter *)parAddr)->learningRateSequence;
    return (jlong)ntShPtr;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_BaseParameter
 * Method:    cSetSeed
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_BaseParameter_cSetSeed
(JNIEnv *, jobject, jlong parAddr, jint seed)
{
    ((sgd::BaseParameter *)parAddr)->seed = seed;
}

/*
 * Class:     com_intel_daal_algorithms_optimization_solver_sgd_BaseParameter
 * Method:    cGetSeed
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sgd_BaseParameter_cGetSeed
(JNIEnv *, jobject, jlong parAddr)
{
    return ((sgd::BaseParameter *)parAddr)->seed;
}
