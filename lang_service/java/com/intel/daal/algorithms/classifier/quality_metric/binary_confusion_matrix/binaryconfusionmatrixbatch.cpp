/* file: binaryconfusionmatrixbatch.cpp */
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
#include "classifier/quality_metric/binary_confusion_matrix/JBinaryConfusionMatrixBatch.h"
#include "classifier/quality_metric/binary_confusion_matrix/JBinaryConfusionMatrixMethod.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::classifier::quality_metric;
using namespace daal::algorithms::classifier::quality_metric::binary_confusion_matrix;

/*
 * Class:     com_intel_daal_algorithms_classifier_quality_metric_binary_confusion_matrix_BinaryConfusionMatrixBatch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_classifier_quality_1metric_binary_1confusion_1matrix_BinaryConfusionMatrixBatch_cInit
  (JNIEnv *, jobject, jint prec, jint method)
{
    return jniBatch<binary_confusion_matrix::Method, Batch, defaultDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_classifier_quality_metric_binary_confusion_matrix_BinaryConfusionMatrixBatch
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_classifier_quality_1metric_binary_1confusion_1matrix_BinaryConfusionMatrixBatch_cSetResult
  (JNIEnv *, jobject, jlong algAddr, jint prec, jint method, jlong resultAddr)
{
    jniBatch<binary_confusion_matrix::Method, Batch, defaultDense>::
        setResult<binary_confusion_matrix::Result>(prec, method, algAddr, resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_classifier_quality_metric_binary_confusion_matrix_BinaryConfusionMatrixBatch
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_classifier_quality_1metric_binary_1confusion_1matrix_BinaryConfusionMatrixBatch_cInitParameter
  (JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<binary_confusion_matrix::Method, Batch, defaultDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_classifier_quality_metric_binary_confusion_matrix_BinaryConfusionMatrixBatch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_classifier_quality_1metric_binary_1confusion_1matrix_BinaryConfusionMatrixBatch_cGetInput
  (JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<binary_confusion_matrix::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_classifier_quality_metric_binary_confusion_matrix_BinaryConfusionMatrixBatch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_classifier_quality_1metric_binary_1confusion_1matrix_BinaryConfusionMatrixBatch_cGetResult
  (JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<binary_confusion_matrix::Method, Batch, defaultDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_classifier_quality_metric_binary_confusion_matrix_BinaryConfusionMatrixBatch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_classifier_quality_1metric_binary_1confusion_1matrix_BinaryConfusionMatrixBatch_cClone
  (JNIEnv *, jobject, jlong algAddr, jint prec, jint method)
{
    return jniBatch<binary_confusion_matrix::Method, Batch, defaultDense>::getClone(prec, method, algAddr);
}
