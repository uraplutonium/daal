/* file: lcn_batch.cpp */
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
#include "neural_networks/layers/lcn/JLcnBatch.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lcn_LcnBatch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lcn_LcnBatch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniBatchLayer<lcn::Method, lcn::Batch, lcn::defaultDense>::
           newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lcn_LcnBatch
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lcn_LcnBatch_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatchLayer<lcn::Method, lcn::Batch, lcn::defaultDense>::
           getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lcn_LcnBatch
 * Method:    cGetForwardLayer
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lcn_LcnBatch_cGetForwardLayer
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatchLayer<lcn::Method, lcn::Batch, lcn::defaultDense>::
           getForwardLayer(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lcn_LcnBatch
 * Method:    cGetBackwardLayer
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lcn_LcnBatch_cGetBackwardLayer
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatchLayer<lcn::Method, lcn::Batch, lcn::defaultDense>::
           getBackwardLayer(prec, method, algAddr);
}
