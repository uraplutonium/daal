/* file: predict_result.cpp */
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

#include <jni.h>

#include "ridge_regression/prediction/JPredictionResult.h"
#include "ridge_regression/prediction/JPredictionResultId.h"
#include "ridge_regression/prediction/JPredictionMethod.h"

#include "common_helpers.h"

#define defaultDenseValue   com_intel_daal_algorithms_ridge_regression_prediction_PredictionMethod_defaultDenseValue
#define predictionId        com_intel_daal_algorithms_ridge_regression_prediction_PredictionResultId_PredictionId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::ridge_regression::prediction;

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_prediction_PredictionResult
 * Method:    cNewResult
 * Signature:()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_prediction_PredictionResult_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<ridge_regression::prediction::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_prediction_PredictionResult
 * Method:    cGetPredictionResult
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_prediction_PredictionResult_cGetPredictionResult
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<ridge_regression::prediction::Result>::get<ridge_regression::prediction::ResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_prediction_PredictionResult
 * Method:    cSetPredictionResult
 * Signature:(JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_ridge_1regression_prediction_PredictionResult_cSetPredictionResult
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<ridge_regression::prediction::Result>::set<ridge_regression::prediction::ResultId, NumericTable>(resAddr, id, ntAddr);
}
