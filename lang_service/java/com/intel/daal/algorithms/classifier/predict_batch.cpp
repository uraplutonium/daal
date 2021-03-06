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
#include <jni.h>/* Header for class com_intel_daal_algorithms_classifier_prediction_PredictionBatch */

#include "daal.h"
#include "classifier/prediction/JPredictionBatch.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::classifier;
using namespace daal::algorithms::classifier::prediction;
using namespace daal::services;

/*
 * Class:     com_intel_daal_algorithms_classifier_prediction_PredictionBatch
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_classifier_prediction_PredictionBatch_cSetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jlong resultAddr)
{
    SerializationIfacePtr *serializableShPtr = (SerializationIfacePtr *)resultAddr;
    classifier::prediction::ResultPtr resultShPtr =
        services::staticPointerCast<classifier::prediction::Result, SerializationIface>(*serializableShPtr);

    SharedPtr<Batch> alg =
        staticPointerCast<Batch, AlgorithmIface>
            (*(SharedPtr<AlgorithmIface> *)algAddr);
    alg->setResult(resultShPtr);
}
