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
#include <jni.h>/* Header for class com_intel_daal_algorithms_adaboost_prediction_PredictionBatch */

#include "daal.h"
#include "adaboost/prediction/JPredictionBatch.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();

/*
 * Class:     com_intel_daal_algorithms_adaboost_prediction_PredictionBatch
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_adaboost_prediction_PredictionBatch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniBatch<adaboost::prediction::Method, adaboost::prediction::Batch, adaboost::prediction::defaultDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_adaboost_prediction_PredictionBatch
 * Method:    cInitParameter
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_adaboost_prediction_PredictionBatch_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<adaboost::prediction::Method, adaboost::prediction::Batch, adaboost::prediction::defaultDense>::
        getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_adaboost_prediction_PredictionBatch
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_adaboost_prediction_PredictionBatch_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<adaboost::prediction::Method, adaboost::prediction::Batch, adaboost::prediction::defaultDense>::getClone(prec, method, algAddr);
}
