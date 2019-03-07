/* file: naivebayesqualitymetricsetbatch.cpp */
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
#include <jni.h>
#include "daal.h"
#include "multinomial_naive_bayes/quality_metric_set/JQualityMetricSetBatch.h"

using namespace daal::algorithms::multinomial_naive_bayes::quality_metric_set;

/*
 * Class:     com_intel_daal_algorithms_multinomial_naive_bayes_quality_metric_set_QualityMetricSetBatch
 * Method:    cInit
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_quality_1metric_1set_QualityMetricSetBatch_cInit
(JNIEnv *, jobject, jlong nClasses)
{
    jlong addr = 0;
    addr = (jlong)(new Batch(nClasses));
    return addr;
}

/*
 * Class:     com_intel_daal_algorithms_multinomial_naive_bayes_quality_metric_set_QualityMetricSetBatch
 * Method:    cInitParameter
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multinomial_1naive_1bayes_quality_1metric_1set_QualityMetricSetBatch_cInitParameter
(JNIEnv *, jobject, jlong parAddr)
{
    jlong addr = 0;
    addr = (jlong) & ((*(Batch *)parAddr).parameter);
    return addr;
}
