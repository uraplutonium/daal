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
#include "svm/JParameter.h"
#include "daal.h"

/*
 * Class:     com_intel_daal_algorithms_svm_Parameter
 * Method:    cSetC
 * Signature:(JD)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svm_Parameter_cSetC
(JNIEnv *env, jobject obj, jlong parAddr, jdouble val)
{
    ((daal::algorithms::svm::Parameter *)parAddr)->C = val;
}

/*
 * Class:     com_intel_daal_algorithms_svm_Parameter
 * Method:    cGetC
 * Signature:(J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_svm_Parameter_cGetC
(JNIEnv *env, jobject obj, jlong parAddr)
{
    return(jdouble)((daal::algorithms::svm::Parameter *)parAddr)->C;
}

/*
 * Class:     com_intel_daal_algorithms_svm_Parameter
 * Method:    cSetAccuracyThreshold
 * Signature:(JD)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svm_Parameter_cSetAccuracyThreshold
(JNIEnv *env, jobject obj, jlong parAddr, jdouble val)
{
    ((daal::algorithms::svm::Parameter *)parAddr)->accuracyThreshold = val;
}

/*
 * Class:     com_intel_daal_algorithms_svm_Parameter
 * Method:    cGetAccuracyThreshold
 * Signature:(J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_svm_Parameter_cGetAccuracyThreshold
(JNIEnv *env, jobject obj, jlong parAddr)
{
    return(jdouble)((daal::algorithms::svm::Parameter *)parAddr)->accuracyThreshold;
}

/*
 * Class:     com_intel_daal_algorithms_svm_Parameter
 * Method:    cSetTau
 * Signature:(JD)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svm_Parameter_cSetTau
(JNIEnv *env, jobject obj, jlong parAddr, jdouble val)
{
    ((daal::algorithms::svm::Parameter *)parAddr)->tau = val;
}

/*
 * Class:     com_intel_daal_algorithms_svm_Parameter
 * Method:    cGetTau
 * Signature:(J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_svm_Parameter_cGetTau
(JNIEnv *env, jobject obj, jlong parAddr)
{
    return(jdouble)((daal::algorithms::svm::Parameter *)parAddr)->tau;
}

/*
 * Class:     com_intel_daal_algorithms_svm_Parameter
 * Method:    cSetMaxIterations
 * Signature:(JJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svm_Parameter_cSetMaxIterations
(JNIEnv *env, jobject obj, jlong parAddr, jlong val)
{
    ((daal::algorithms::svm::Parameter *)parAddr)->maxIterations = val;
}

/*
 * Class:     com_intel_daal_algorithms_svm_Parameter
 * Method:    cGetMaxIterations
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svm_Parameter_cGetMaxIterations
(JNIEnv *env, jobject obj, jlong parAddr)
{
    return(jlong)((daal::algorithms::svm::Parameter *)parAddr)->maxIterations;
}

/*
 * Class:     com_intel_daal_algorithms_svm_Parameter
 * Method:    cSetCacheSize
 * Signature:(JJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svm_Parameter_cSetCacheSize
(JNIEnv *env, jobject obj, jlong parAddr, jlong val)
{
    ((daal::algorithms::svm::Parameter *)parAddr)->cacheSize = val;
}

/*
 * Class:     com_intel_daal_algorithms_svm_Parameter
 * Method:    cGetCacheSize
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svm_Parameter_cGetCacheSize
(JNIEnv *env, jobject obj, jlong parAddr)
{
    return(jlong)((daal::algorithms::svm::Parameter *)parAddr)->cacheSize;
}

/*
 * Class:     com_intel_daal_algorithms_svm_Parameter
 * Method:    cSetDoShrinking
 * Signature:(JZ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svm_Parameter_cSetDoShrinking
(JNIEnv *env, jobject obj, jlong parAddr, jboolean val)
{
    ((daal::algorithms::svm::Parameter *)parAddr)->doShrinking = val;
}

/*
 * Class:     com_intel_daal_algorithms_svm_Parameter
 * Method:    cGetDoShrinking
 * Signature:(J)Z
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_algorithms_svm_Parameter_cGetDoShrinking
(JNIEnv *env, jobject obj, jlong parAddr)
{
    return(jboolean)((daal::algorithms::svm::Parameter *)parAddr)->doShrinking;
}

/*
 * Class:     com_intel_daal_algorithms_svm_Parameter
 * Method:    cSetKernel
 * Signature:(JJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_svm_Parameter_cSetKernel
(JNIEnv *env, jobject obj, jlong parAddr, jlong kernelAddr)
{
    daal::services::SharedPtr<daal::algorithms::kernel_function::KernelIface> kernel =
        daal::services::staticPointerCast<daal::algorithms::kernel_function::KernelIface, daal::algorithms::AlgorithmIface>
            (*(daal::services::SharedPtr<daal::algorithms::AlgorithmIface> *)kernelAddr);
    daal::services::SharedPtr<daal::algorithms::kernel_function::KernelIface>* shKernel =
        new daal::services::SharedPtr<daal::algorithms::kernel_function::KernelIface>(kernel);

    ((daal::algorithms::svm::Parameter *)parAddr)->kernel = *shKernel;
}
