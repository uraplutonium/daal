/* file: parameter.cpp */
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

/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>/* Header for class com_intel_daal_algorithms_adaboost_Batch */

#include "daal.h"
#include "adaboost/JParameter.h"

using namespace daal;
using namespace daal::algorithms;

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_adaboost_Parameter_cSetAccuracyThreshold
(JNIEnv *env, jobject thisObj, jlong parAddr, jdouble acc)
{
    (*(adaboost::Parameter *)parAddr).accuracyThreshold = acc;
}

JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_adaboost_Parameter_cGetAccuracyThreshold
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    return(jdouble)(*(adaboost::Parameter *)parAddr).accuracyThreshold;
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_adaboost_Parameter_cSetMaxIterations
(JNIEnv *env, jobject thisObj, jlong parAddr, jlong nIter)
{
    (*(adaboost::Parameter *)parAddr).maxIterations = nIter;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_adaboost_Parameter_cGetMaxIterations
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    return(jlong)(*(adaboost::Parameter *)parAddr).maxIterations;
}
