/* file: model.cpp */
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

#include "daal.h"
#include "multinomial_naive_bayes/JModel.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_Model_cGetLogP
(JNIEnv *env, jobject thisObj, jlong modelAddr)
{
    NumericTablePtr *nt = new NumericTablePtr();
    multinomial_naive_bayes::ModelPtr res = *(multinomial_naive_bayes::ModelPtr *)modelAddr;

    *nt = res->getLogP();

    return (jlong)nt;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_Model_cGetLogTheta
(JNIEnv *env, jobject thisObj, jlong modelAddr)
{
    NumericTablePtr *nt = new NumericTablePtr();
    multinomial_naive_bayes::ModelPtr res = *(multinomial_naive_bayes::ModelPtr *)modelAddr;

    *nt = res->getLogTheta();

    return (jlong)nt;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_Model_cGetNFeatures
(JNIEnv *env, jobject thisObj, jlong modelAddr)
{
    multinomial_naive_bayes::ModelPtr res = *(multinomial_naive_bayes::ModelPtr *)modelAddr;

    return (jlong)(res->getNFeatures());
}
