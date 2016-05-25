/* file: parameter.cpp */
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
#include "neural_networks/layers/lrn/JParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_Parameter
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_Parameter_cInit
(JNIEnv *env, jobject thisObj)
{
    return (jlong)(new lrn::Parameter);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_Parameter
 * Method:    cGetDimension
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_Parameter_cGetDimension
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    SharedPtr<NumericTable> *ntShPtr = new SharedPtr<NumericTable>();
    *ntShPtr = (((lrn::Parameter *)cParameter))->dimension;
    return (jlong)ntShPtr;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_Parameter
 * Method:    cSetDimension
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_Parameter_cSetDimension
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong dimension)
{
    SharedPtr<SerializationIface> *ntShPtr = (SharedPtr<SerializationIface> *)dimension;
    (((lrn::Parameter *)cParameter))->dimension = staticPointerCast<NumericTable, SerializationIface>(*ntShPtr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_Parameter
 * Method:    cGetkappa
 * Signature: (J)J
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_Parameter_cGetkappa
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((lrn::Parameter *)cParameter))->kappa);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_Parameter
 * Method:    cSetKappa
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_Parameter_cSetKappa
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong kappa)
{
    (((lrn::Parameter *)cParameter))->kappa = (size_t)kappa;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_Parameter
 * Method:    cGetAlpha
 * Signature: (J)J
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_Parameter_cGetAlpha
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((lrn::Parameter *)cParameter))->alpha);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_Parameter
 * Method:    cSetAlpha
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_Parameter_cSetAlpha
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong alpha)
{
    (((lrn::Parameter *)cParameter))->alpha = alpha;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_Parameter
 * Method:    cGetBeta
 * Signature: (J)J
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_Parameter_cGetBeta
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((lrn::Parameter *)cParameter))->beta);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_Parameter
 * Method:    cSetBeta
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_Parameter_cSetBeta
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong beta)
{
    (((lrn::Parameter *)cParameter))->beta = beta;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_Parameter
 * Method:    cGetNAdjust
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_Parameter_cGetNAdjust
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((lrn::Parameter *)cParameter))->nAdjust);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_lrn_Parameter
 * Method:    cSetNAdjust
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_lrn_Parameter_cSetNAdjust
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong nAdjust)
{
    (((lrn::Parameter *)cParameter))->nAdjust = (size_t)nAdjust;
}
