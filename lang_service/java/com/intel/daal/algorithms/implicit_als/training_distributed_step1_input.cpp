/* file: training_distributed_step1_input.cpp */
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

#include "implicit_als/training/JDistributedStep1LocalInput.h"

#include "implicit_als_training_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::implicit_als::training;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep1LocalInput
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep1LocalInput_cGetInput
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local, algorithms::implicit_als::training::Method, Distributed, fastCSR>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep1LocalInput
 * Method:    cSetPartialModel
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep1LocalInput_cSetPartialModel
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong partialModelAddr)
{
    jniInput<DistributedInput<step1Local> >::set<PartialModelInputId, algorithms::implicit_als::PartialModel>(inputAddr, id, partialModelAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_DistributedStep1LocalInput
 * Method:    cGetPartialModel
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_DistributedStep1LocalInput_cGetPartialModel
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<DistributedInput<step1Local> >::get<PartialModelInputId, algorithms::implicit_als::PartialModel>(inputAddr, id);
}
