/* file: layer_descriptors.cpp */
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
#include "neural_networks/JLayerDescriptors.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_LayerDescriptors
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_LayerDescriptors_cInit
(JNIEnv *env, jobject thisObj)
{
    return (jlong)(new SharedPtr<Collection<layers::LayerDescriptor> >(new Collection<layers::LayerDescriptor>()));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_LayerDescriptors
 * Method:    cSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_LayerDescriptors_cSize
(JNIEnv *env, jobject thisObj, jlong colAddr)
{
    return (*(SharedPtr<Collection<layers::LayerDescriptor > >*)colAddr)->size();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_LayerDescriptors
 * Method:    cGet
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_LayerDescriptors_cGet
(JNIEnv *env, jobject thisObj, jlong colAddr, jlong index)
{
    layers::LayerDescriptor layerDescriptor = (*(SharedPtr<Collection<layers::LayerDescriptor > >*)colAddr)->get((size_t)index);
    layers::LayerDescriptor *layerDescriptorPtr = new layers::LayerDescriptor(layerDescriptor.index,
                                                                              layerDescriptor.layer,
                                                                              layerDescriptor.nextLayers);
    return (jlong)layerDescriptorPtr;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_LayerDescriptors
 * Method:    cPushBack
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_LayerDescriptors_cPushBack
(JNIEnv *env, jobject thisObj, jlong colAddr, jlong layerAddr)
{
    (*(SharedPtr<Collection<layers::LayerDescriptor > >*)colAddr)->push_back(*((layers::LayerDescriptor *)layerAddr));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_LayerDescriptors
 * Method:    cDispose
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_LayerDescriptors_cDispose
(JNIEnv *env, jobject thisObj, jlong addr)
{
    delete(SharedPtr<Collection<layers::LayerDescriptor> > *)addr;
}
