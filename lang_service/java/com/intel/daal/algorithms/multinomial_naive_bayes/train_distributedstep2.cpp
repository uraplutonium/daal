/* file: train_distributedstep2.cpp */
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
/* Header for class com_intel_daal_algorithms_multinomial_naive_bayes_training_TrainingDistributedStep2Master */

#include "daal.h"
#include "multinomial_naive_bayes/training/JTrainingDistributedStep2Master.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::multinomial_naive_bayes::training;

/*
 * Class:     com_intel_daal_algorithms_multinomial_naive_bayes_training_TrainingDistributedStep2Master
 * Method:    cInit
 * Signature: (IIJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_training_TrainingDistributedStep2Master_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method, jlong nClasses)
{
    return jniDistributed<step2Master, multinomial_naive_bayes::training::Method, Distributed, defaultDense, fastCSR>::newObj(prec, method, nClasses);
}

/*
 * Class:     com_intel_daal_algorithms_multinomial_naive_bayes_training_TrainingDistributedStep2Master
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_training_TrainingDistributedStep2Master_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, multinomial_naive_bayes::training::Method, Distributed, defaultDense, fastCSR>::
        getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_multinomial_naive_bayes_training_TrainingDistributedStep2Master
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_training_TrainingDistributedStep2Master_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, multinomial_naive_bayes::training::Method, Distributed, defaultDense, fastCSR>::
        getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_multinomial_naive_bayes_training_TrainingDistributedStep2Master
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_training_TrainingDistributedStep2Master_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, multinomial_naive_bayes::training::Method, Distributed, defaultDense, fastCSR>::
        getResult(prec, method, algAddr);
}


JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_training_TrainingDistributedStep2Master_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong cObj)
{
    jniDistributed<step2Master, multinomial_naive_bayes::training::Method, Distributed, defaultDense, fastCSR>::
        setResult<multinomial_naive_bayes::training::Result>(prec, method, algAddr, cObj);
}

/*
 * Class:     com_intel_daal_algorithms_multinomial_naive_bayes_training_TrainingDistributedStep2Master
 * Method:    cGetPartialResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_training_TrainingDistributedStep2Master_cGetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, multinomial_naive_bayes::training::Method, Distributed, defaultDense, fastCSR>::
        getPartialResult(prec, method, algAddr);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_training_TrainingDistributedStep2Master_cGetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong cObj)
{
    jniDistributed<step2Master, multinomial_naive_bayes::training::Method, Distributed, defaultDense, fastCSR>::
        setPartialResult<multinomial_naive_bayes::training::PartialResult>(prec, method, algAddr, cObj);
}

/*
 * Class:     com_intel_daal_algorithms_multinomial_naive_bayes_training_TrainingDistributedStep2Master
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_training_TrainingDistributedStep2Master_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, multinomial_naive_bayes::training::Method, Distributed, defaultDense, fastCSR>::
        getClone(prec, method, algAddr);
}
