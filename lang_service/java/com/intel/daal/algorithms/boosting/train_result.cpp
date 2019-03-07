/* file: train_result.cpp */
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
#include <jni.h>/* Header for class com_intel_daal_algorithms_boosting_training_TrainingResult */

#include "daal.h"
#include "boosting/training/JTrainingResult.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::services;

#include "train_types.i"

/*
 * Class:     com_intel_daal_algorithms_boosting_training_TrainingResult
 * Method:    cGetMinimum
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_boosting_training_TrainingResult_cGetModel
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    boosting::ModelPtr *m = new boosting::ModelPtr();
    classifier::training::ResultPtr res =
        services::staticPointerCast<classifier::training::Result, SerializationIface>(*((SerializationIfacePtr *)resAddr));

    jlong resModel = 0;
    switch(id)
    {
    case ModelResult:
        *m = services::staticPointerCast<boosting::Model, classifier::Model>(res->get(classifier::training::model));
        break;
    default:
        break;
    }
    return (jlong)m;
}
