/* file: train_batch.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>/* Header for class com_intel_daal_algorithms_brownboost_training_TrainingBatch */

#include "daal.h"
#include "brownboost/training/JTrainingBatch.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::brownboost;

/*
 * Class:     com_intel_daal_algorithms_brownboost_training_TrainingBatch
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_brownboost_training_TrainingBatch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniBatch<brownboost::training::Method, training::Batch, training::defaultDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_brownboost_training_TrainingBatch
 * Method:    cInitParameter
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_brownboost_training_TrainingBatch_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<brownboost::training::Method, training::Batch, training::defaultDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_brownboost_training_TrainingBatch
 * Method:    cInitParameter
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_brownboost_training_TrainingBatch_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<brownboost::training::Method, training::Batch, training::defaultDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_brownboost_training_TrainingBatch
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_brownboost_training_TrainingBatch_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<brownboost::training::Method, training::Batch, training::defaultDense>::getClone(prec, method, algAddr);
}
