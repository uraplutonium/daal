/* file: logistic_backward_input.cpp */
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
#include "neural_networks/layers/logistic/JLogisticBackwardInput.h"
#include "neural_networks/layers/logistic/JLogisticLayerDataId.h"

#include "daal.h"

#include "common_helpers.h"

#define auxValueId com_intel_daal_algorithms_neural_networks_layers_logistic_LogisticLayerDataId_auxValueId

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers::logistic;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_logistic_LogisticBackwardInput
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_logistic_LogisticBackwardInput_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if (id == auxValueId)
    {
        jniInput<backward::Input>::set<LayerDataId, Tensor>(inputAddr, auxValue, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_logistic_LogisticBackwardInput
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_logistic_LogisticBackwardInput_cGetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if (id == auxValueId)
    {
        return jniInput<backward::Input>::get<LayerDataId, Tensor>(inputAddr, auxValue);
    }

    return (jlong)0;
}
