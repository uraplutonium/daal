/* file: softmax_cross_forward_batch.cpp */
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
#include "neural_networks/layers/softmax_cross/JSoftmaxCrossForwardBatch.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers::loss;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_softmax_cross_SoftmaxCrossForwardBatch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_softmax_1cross_SoftmaxCrossForwardBatch_cInit
  (JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniBatch<softmax_cross::Method, softmax_cross::forward::Batch, softmax_cross::defaultDense>::
        newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_softmax_cross_SoftmaxCrossForwardBatch
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_softmax_1cross_SoftmaxCrossForwardBatch_cInitParameter
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<softmax_cross::Method, softmax_cross::forward::Batch, softmax_cross::defaultDense>::
        getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_softmax_cross_SoftmaxCrossForwardBatch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_softmax_1cross_SoftmaxCrossForwardBatch_cGetInput
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<softmax_cross::Method, softmax_cross::forward::Batch, softmax_cross::defaultDense>::
        getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_softmax_cross_SoftmaxCrossForwardBatch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_softmax_1cross_SoftmaxCrossForwardBatch_cGetResult
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<softmax_cross::Method, softmax_cross::forward::Batch, softmax_cross::defaultDense>::
        getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_softmax_cross_SoftmaxCrossForwardBatch
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_softmax_1cross_SoftmaxCrossForwardBatch_cSetResult
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resAddr)
{
    jniBatch<softmax_cross::Method, softmax_cross::forward::Batch, softmax_cross::defaultDense>::
        setResult<softmax_cross::forward::Result>(prec, method, algAddr, resAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_softmax_cross_SoftmaxCrossForwardBatch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_softmax_1cross_SoftmaxCrossForwardBatch_cClone
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<softmax_cross::Method, softmax_cross::forward::Batch, softmax_cross::defaultDense>::
        getClone(prec, method, algAddr);
}
