/* file: predict_batch.cpp */
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
#include <jni.h>/* Header for class com_intel_daal_algorithms_multinomial_naive_bayes_prediction_PredictionBatch */

#include "daal.h"
#include "multinomial_naive_bayes/prediction/JPredictionBatch.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::multinomial_naive_bayes::prediction;

/*
* Class:     com_intel_daal_algorithms_multinomial_naive_bayes_prediction_PredictionBatch
* Method:    cInit
* Signature: (IIJ)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_prediction_PredictionBatch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method, jlong nClasses)
{
    return jniBatch<multinomial_naive_bayes::prediction::Method, Batch, defaultDense, fastCSR>::newObj(prec, method, nClasses);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_prediction_PredictionBatch_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<multinomial_naive_bayes::prediction::Method, Batch, defaultDense, fastCSR>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_multinomial_naive_bayes_prediction_PredictionBatch
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_prediction_PredictionBatch_cSetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resultAddr)
{
    jniBatch<multinomial_naive_bayes::prediction::Method, Batch, defaultDense, fastCSR>::
        setResult<classifier::prediction::Result>(prec, method, algAddr, resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_multinomial_naive_bayes_prediction_PredictionBatch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_prediction_PredictionBatch_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<multinomial_naive_bayes::prediction::Method, Batch, defaultDense, fastCSR>::getClone(prec, method, algAddr);
}
