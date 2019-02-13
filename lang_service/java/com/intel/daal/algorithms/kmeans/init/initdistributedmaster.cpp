/* file: initdistributedmaster.cpp */
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
#include "kmeans/init/JInitDistributedStep2Master.h"
#include "kmeans/init/JInitDistributedStep3Master.h"
#include "kmeans/init/JInitDistributedStep5Master.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::kmeans::init;

#define AllMethodsList\
    kmeans::init::Method, Distributed, deterministicDense, randomDense, deterministicCSR, randomCSR

#define PlusPlusMethodsList\
    kmeans::init::Method, Distributed, plusPlusDense, parallelPlusDense, plusPlusCSR, parallelPlusCSR

#define ParallelPlusMethodsList\
    kmeans::init::Method, Distributed, parallelPlusDense, parallelPlusCSR

/*
 * Class:     com_intel_daal_algorithms_kmeans_Distributed
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2Master_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method, jlong nClusters)
{
    return jniDistributed<step2Master, AllMethodsList>::newObj(prec, method, nClusters);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep2Master
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2Master_cSetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resultAddr)
{
    jniDistributed<step2Master, AllMethodsList>::setResult<kmeans::init::Result>(prec, method, algAddr, resultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep2Master
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2Master_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, AllMethodsList>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep2Master
 * Method:    cSetPartialResult
 * Signature: (JIIJZ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2Master_cSetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong partialResultAddr, jboolean initFlag)
{
    jniDistributed<step2Master, AllMethodsList>::
        setPartialResult<kmeans::init::PartialResult>(prec, method, algAddr, partialResultAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep2Master
 * Method:    cGetPartialResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2Master_cGetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, AllMethodsList>::getPartialResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep2Master
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2Master_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, AllMethodsList>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep2Master
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2Master_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, AllMethodsList>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep2Master
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2Master_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, AllMethodsList>::getClone(prec, method, algAddr);
}

/////////////////////////////////////// plusPlus methods ///////////////////////////////////////////////////////
///////////////////////////////////////   step3Master     ///////////////////////////////////////////////////////
/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep3Master
* Method:    cInit
* Signature: (IIJ)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep3Master_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method, jlong nClusters)
{
    return jniDistributed<step3Master, PlusPlusMethodsList>::newObj(prec, method, nClusters);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep3Master
* Method:    cInitParameter
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep3Master_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step3Master, PlusPlusMethodsList>::getParameter(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep3Master
* Method:    cGetInput
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep3Master_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step3Master, PlusPlusMethodsList>::getInput(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep3Master
* Method:    cSetPartialResult
* Signature: (JIIJZ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep3Master_cSetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong partialResultAddr, jboolean initFlag)
{
    jniDistributed<step3Master, PlusPlusMethodsList>::setPartialResult<
        kmeans::init::DistributedStep3MasterPlusPlusPartialResult>(prec, method, algAddr, partialResultAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep3Master
* Method:    cGetPartialResult
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep3Master_cGetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step3Master, PlusPlusMethodsList>::getPartialResult(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep3Master
* Method:    cClone
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep3Master_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step3Master, PlusPlusMethodsList>::getClone(prec, method, algAddr);
}

///////////////////////////////////////   step5Master     ///////////////////////////////////////////////////////
/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep5Master
* Method:    cInit
* Signature: (IIJ)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep5Master_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method, jlong nClusters)
{
    return jniDistributed<step5Master, ParallelPlusMethodsList>::newObj(prec, method, nClusters);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep5Master
* Method:    cInitParameter
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep5Master_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step5Master, ParallelPlusMethodsList>::getParameter(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep5Master
* Method:    cGetInput
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep5Master_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step5Master, ParallelPlusMethodsList>::getInput(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep5Master
* Method:    cGetResult
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep5Master_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step5Master, ParallelPlusMethodsList>::getResult(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep5Master
* Method:    cSetResult
* Signature: (JIIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep5Master_cSetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resultAddr)
{
    jniDistributed<step5Master, ParallelPlusMethodsList>::setResult<kmeans::init::Result>(prec, method, algAddr, resultAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep5Master
* Method:    cGetPartialResult
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep5Master_cGetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step5Master, ParallelPlusMethodsList>::getPartialResult(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep5Master
* Method:    cSetPartialResult
* Signature: (JIIJZ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep5Master_cSetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong partialResultAddr, jboolean initFlag)
{
    jniDistributed<step5Master, ParallelPlusMethodsList>::setPartialResult<
        kmeans::init::DistributedStep5MasterPlusPlusPartialResult>(prec, method, algAddr, partialResultAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep5Master
* Method:    cClone
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep5Master_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step5Master, ParallelPlusMethodsList>::getClone(prec, method, algAddr);
}
