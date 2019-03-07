/* file: train_distributedinput.cpp */
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
#include <jni.h>
/* Header for class com_intel_daal_algorithms_classifier_training_TrainingDistributedInput */

#include "daal.h"
#include "multinomial_naive_bayes/training/JTrainingDistributedInput.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::multinomial_naive_bayes;
using namespace daal::algorithms::multinomial_naive_bayes::training;

/*
 * Class:     com_intel_daal_algorithms_multinomial_naive_bayes_training_TrainingDistributedInput
 * Method:    cAddInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_training_TrainingDistributedInput_cAddInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr )
{
    jniInput<DistributedInput>::add<Step2MasterInputId, multinomial_naive_bayes::training::PartialResult>(inputAddr, partialModels, ntAddr);
}
