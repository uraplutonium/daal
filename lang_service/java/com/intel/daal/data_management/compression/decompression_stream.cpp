/* file: decompression_stream.cpp */
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

#include "JDecompressionStream.h"
#include "daal.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_compression_Decompression
 * Method:    cDispose
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_compression_DecompressionStream_cDispose
(JNIEnv *env, jobject, jlong strAddr)
{
    delete(DecompressionStream *)strAddr;
}

/*
 * Class:     com_intel_daal_data_1management_compression_Decompression
 * Method:    cDispose
 * Signature:(J)V
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_compression_DecompressionStream_cInit
(JNIEnv *env, jobject, jlong comprAddr, jlong minSize)
{
    jlong strmAddr = 0;
    strmAddr = (jlong)(new DecompressionStream((DecompressorImpl *)comprAddr, minSize));

    if(((DecompressionStream *)strmAddr)->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"),
                      ((DecompressionStream *)strmAddr)->getErrors()->getDescription());
    }
    return strmAddr;
}

/*
 * Class:     com_intel_daal_data_1management_compression_Decompression
 * Method:    cSetInputDataBlock
 * Signature:(J[BJJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_compression_DecompressionStream_cAdd
(JNIEnv *env, jobject, jlong strmAddr, jbyteArray inBlock, jlong size)
{
    jbyte *inBuffer = env->GetByteArrayElements(inBlock, 0);

    DataBlock tmp((byte *)inBuffer, (size_t)size);

    ((CompressionStream *)strmAddr)->push_back(&tmp);

    if(((DecompressionStream *)strmAddr)->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"),
                      ((DecompressionStream *)strmAddr)->getErrors()->getDescription());
    }
}

/*
 * Class:     com_intel_daal_data_1management_compression_Decompression
 * Method:    cIsOutputDataBlockFull
 * Signature:(J)Z
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_compression_DecompressionStream_cGetDecompressedDataSize
(JNIEnv *env, jobject, jlong strmAddr)
{

    if(((DecompressionStream *)strmAddr)->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"),
                      ((DecompressionStream *)strmAddr)->getErrors()->getDescription());
    }
    return(jlong)((DecompressionStream *)strmAddr)->getDecompressedDataSize();
}

/*
 * Class:     com_intel_daal_data_1management_compression_Decompression
 * Method:    cRun
 * Signature:(J[BJJ)I
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_compression_DecompressionStream_cCopyDecompressedArray
(JNIEnv *env, jobject, jlong strmAddr, jbyteArray outBlock, jlong chunkSize)
{
    jbyte *outBuffer = env->GetByteArrayElements(outBlock, 0);
    jlong result = (jlong)((DecompressionStream *)strmAddr)->copyDecompressedArray((byte *)outBuffer, (size_t)chunkSize);
    env->ReleaseByteArrayElements(outBlock, outBuffer, 0);

    if(((DecompressionStream *)strmAddr)->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"),
                      ((DecompressionStream *)strmAddr)->getErrors()->getDescription());
    }

    return result;
}
