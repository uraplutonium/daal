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
#include "neural_networks/layers/convolution2d/JParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_convolution2d_Parameter
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_convolution2d_Parameter_cInit
(JNIEnv *env, jobject thisObj)
{
    return (jlong)(new convolution2d::Parameter);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_convolution2d_Parameter
 * Method:    cGetGroupDimension
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_convolution2d_Parameter_cGetGroupDimension
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((convolution2d::Parameter *)cParameter))->groupDimension);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_convolution2d_Parameter
 * Method:    cSetGroupDimension
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_convolution2d_Parameter_cSetGroupDimension
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong groupDimension)
{
    (((convolution2d::Parameter *)cParameter))->groupDimension = (size_t)groupDimension;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_convolution2d_Parameter
 * Method:    cgetNKernels
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_convolution2d_Parameter_cgetNKernels
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((convolution2d::Parameter *)cParameter))->nKernels);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_convolution2d_Parameter
 * Method:    csetNKernels
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_convolution2d_Parameter_csetNKernels
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong nKernels)
{
    (((convolution2d::Parameter *)cParameter))->nKernels = (size_t)nKernels;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_convolution2d_Parameter
 * Method:    cgetNGroups
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_convolution2d_Parameter_cgetNGroups
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((convolution2d::Parameter *)cParameter))->nGroups);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_convolution2d_Parameter
 * Method:    csetNGroups
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_convolution2d_Parameter_csetNGroups
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong nGroups)
{
    (((convolution2d::Parameter *)cParameter))->nGroups = (size_t)nGroups;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_convolution2d_Parameter
 * Method:    cSetKernelSize
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_convolution2d_Parameter_cSetKernelSize
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first, jlong second)
{
    (((convolution2d::Parameter *)cParameter))->kernelSize.size[0] = first;
    (((convolution2d::Parameter *)cParameter))->kernelSize.size[1] = second;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_convolution2d_Parameter
 * Method:    cSetStride
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_convolution2d_Parameter_cSetStride
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first, jlong second)
{
    (((convolution2d::Parameter *)cParameter))->stride.size[0] = first;
    (((convolution2d::Parameter *)cParameter))->stride.size[1] = second;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_convolution2d_Parameter
 * Method:    cSetSD
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_convolution2d_Parameter_cSetSD
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first, jlong second)
{
    (((convolution2d::Parameter *)cParameter))->spatialDimensions.size[0] = first;
    (((convolution2d::Parameter *)cParameter))->spatialDimensions.size[1] = second;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_convolution2d_Parameter
 * Method:    cSetPadding
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_convolution2d_Parameter_cSetPadding
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong first, jlong second)
{
    (((convolution2d::Parameter *)cParameter))->padding.size[0] = first;
    (((convolution2d::Parameter *)cParameter))->padding.size[1] = second;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_convolution2d_Parameter
 * Method:    cGetKernelSize
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_convolution2d_Parameter_cGetKernelSize
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(2);
    jlong tmp[2];
    tmp[0] = (jlong)((((convolution2d::Parameter *)cParameter))->kernelSize.size[0]);
    tmp[1] = (jlong)((((convolution2d::Parameter *)cParameter))->kernelSize.size[1]);
    env->SetLongArrayRegion(sizeArray, 0, 2, tmp);

    return sizeArray;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_convolution2d_Parameter
 * Method:    cGetKernelSize
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_convolution2d_Parameter_cGetStride
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(2);
    jlong tmp[2];
    tmp[0] = (jlong)((((convolution2d::Parameter *)cParameter))->stride.size[0]);
    tmp[1] = (jlong)((((convolution2d::Parameter *)cParameter))->stride.size[1]);
    env->SetLongArrayRegion(sizeArray, 0, 2, tmp);

    return sizeArray;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_convolution2d_Parameter
 * Method:    cGetKernelSize
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_convolution2d_Parameter_cGetPadding
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(2);
    jlong tmp[2];
    tmp[0] = (jlong)((((convolution2d::Parameter *)cParameter))->padding.size[0]);
    tmp[1] = (jlong)((((convolution2d::Parameter *)cParameter))->padding.size[1]);
    env->SetLongArrayRegion(sizeArray, 0, 2, tmp);

    return sizeArray;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_convolution2d_Parameter
 * Method:    cGetKernelSize
 * Signature: (J)J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_convolution2d_Parameter_cGetSD
(JNIEnv *env, jobject thisObj, jlong cParameter)
{
    jlongArray sizeArray;
    sizeArray = env->NewLongArray(2);
    jlong tmp[2];
    tmp[0] = (jlong)((((convolution2d::Parameter *)cParameter))->spatialDimensions.size[0]);
    tmp[1] = (jlong)((((convolution2d::Parameter *)cParameter))->spatialDimensions.size[1]);
    env->SetLongArrayRegion(sizeArray, 0, 2, tmp);

    return sizeArray;
}
