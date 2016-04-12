/* file: backward_batch.cpp */
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
#include "neural_networks/layers/dropout/JBackwardBatch.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_dropout_BackwardBatch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_dropout_BackwardBatch_cInit
  (JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniBatch<dropout::Method, dropout::backward::Batch, dropout::defaultDense>::
        newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_dropout_BackwardBatch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_dropout_BackwardBatch_cGetInput
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<dropout::Method, dropout::backward::Batch, dropout::defaultDense>::
        getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_dropout_BackwardBatch
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_dropout_BackwardBatch_cInitParameter
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<dropout::Method, dropout::backward::Batch, dropout::defaultDense>::
        getParameterValue(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_dropout_BackwardBatch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_dropout_BackwardBatch_cGetResult
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<dropout::Method, dropout::backward::Batch, dropout::defaultDense>::
        getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_dropout_BackwardBatch
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_dropout_BackwardBatch_cSetResult
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resAddr)
{
    jniBatch<dropout::Method, dropout::backward::Batch, dropout::defaultDense>::
        setResult<dropout::backward::Result>(prec, method, algAddr, resAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_dropout_BackwardBatch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_dropout_BackwardBatch_cClone
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<dropout::Method, dropout::backward::Batch, dropout::defaultDense>::
        getClone(prec, method, algAddr);
}
