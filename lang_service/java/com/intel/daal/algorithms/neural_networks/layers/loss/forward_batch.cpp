/* file: forward_batch.cpp */
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
#include "neural_networks/layers/loss/JForwardBatch.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_loss_ForwardBatch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_loss_ForwardBatch_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    return (jlong) (*((SharedPtr<loss::forward::Batch> *)algAddr))->getLayerInput();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_loss_ForwardBatch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_loss_ForwardBatch_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SharedPtr<SerializationIface> *ptr = new SharedPtr<SerializationIface>;
    *ptr = (*((SharedPtr<loss::forward::Batch> *)algAddr))->getLayerResult();
    return (jlong)ptr;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_loss_ForwardBatch
 * Method:    cDispose
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_ForwardLayer_cDispose
(JNIEnv *env, jobject thisObj, jlong addr)
{
    delete(SharedPtr<loss::forward::Batch> *)addr;
}
