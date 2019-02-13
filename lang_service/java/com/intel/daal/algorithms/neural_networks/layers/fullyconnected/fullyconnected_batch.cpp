/* file: fullyconnected_batch.cpp */
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
#include "neural_networks/layers/fullyconnected/JFullyConnectedBatch.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_fullyconnected_FullyConnectedBatch
 * Method:    cInit
 * Signature: (IIJJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_fullyconnected_FullyConnectedBatch_cInit
  (JNIEnv *env, jobject thisObj, jint prec, jint method, jlong nOutputs)
{
    return jniBatchLayer<fullyconnected::Method, fullyconnected::Batch, fullyconnected::defaultDense>::
        newObj(prec, method, (size_t)nOutputs);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_fullyconnected_FullyConnectedBatch
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_fullyconnected_FullyConnectedBatch_cInitParameter
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatchLayer<fullyconnected::Method, fullyconnected::Batch, fullyconnected::defaultDense>::
        getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_fullyconnected_FullyConnectedBatch
 * Method:    cGetForwardLayer
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_fullyconnected_FullyConnectedBatch_cGetForwardLayer
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatchLayer<fullyconnected::Method, fullyconnected::Batch, fullyconnected::defaultDense>::
        getForwardLayer(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_fullyconnected_FullyConnectedBatch
 * Method:    cGetBackwardLayer
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_fullyconnected_FullyConnectedBatch_cGetBackwardLayer
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatchLayer<fullyconnected::Method, fullyconnected::Batch, fullyconnected::defaultDense>::
        getBackwardLayer(prec, method, algAddr);
}
