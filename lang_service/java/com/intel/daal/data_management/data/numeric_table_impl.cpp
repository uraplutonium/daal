/* file: numeric_table_impl.cpp */
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

#include "JNumericTableImpl.h"
#include "numeric_table.h"
#include "homogen_numeric_table.h"

#include "daal.h"

#include "java_numeric_table.h"
#include "common_defines.i"
#include "common_helpers_functions.h"

using namespace daal::services;
using namespace daal::data_management;

JavaVM* daal::JavaNumericTableBase::globalJavaVM = NULL;
tbb::enumerable_thread_specific<jobject> daal::JavaNumericTableBase::globalDaalContext;

inline static NumericTablePtr *getNIONumericTableObject(JNIEnv *env, jobject thisObj)
{
    // Get a class reference for Java NumericTableFeature
    jclass cls = env->FindClass("com/intel/daal/data_management/data/NumericTable");
    jfieldID objFieldID = env->GetFieldID(cls, "cObject", "J");
    return (NumericTablePtr *)(env->GetLongField(thisObj, objFieldID));
}

/*
 * Class:     daal_NumericTableImpl
 * Method:    allocateDataMemory
 * Signature:()J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cAllocateDataMemory
(JNIEnv *env, jobject thisObj)
{
    // Get a class reference for Java NumericTable
    jclass cls = env->FindClass("com/intel/daal/data_management/data/NumericTable");

    jfieldID objFieldID = env->GetFieldID(cls, "cObject", "J");
    jlong cObj = env->GetLongField(thisObj, objFieldID);

    NumericTable *tbl = ((NumericTablePtr *)cObj)->get();

    DAAL_CHECK_THROW(((HomogenNumericTable<> *)tbl)->allocateDataMemory());
}

/*
 * Class:     daal_NumericTableImpl
 * Method:    freeDataMemory
 * Signature:()V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cFreeDataMemory
(JNIEnv *env, jobject thisObj)
{
    // Get a class reference for Java NumericTable
    jclass cls = env->FindClass("com/intel/daal/data_management/data/NumericTable");

    jfieldID objFieldID = env->GetFieldID(cls, "cObject", "J");
    jlong cObj = env->GetLongField(thisObj, objFieldID);

    NumericTable *tbl = ((NumericTablePtr *)cObj)->get();

    ((HomogenNumericTable<> *)tbl)->freeDataMemory();
}

/*
 * Class:     daal_NumericTableImpl
 * Method:    getNumberOfColumns
 * Signature:()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cGetNumberOfColumns
(JNIEnv *env, jobject thisObj)
{
    // Get a class reference for Java NumericTable
    jclass cls = env->FindClass("com/intel/daal/data_management/data/NumericTable");

    jfieldID objFieldID = env->GetFieldID(cls, "cObject", "J");
    jlong cObj = env->GetLongField(thisObj, objFieldID);

    NumericTable *tbl = ((NumericTablePtr *)cObj)->get();

    jlong nColumns = tbl->getNumberOfColumns();

    return nColumns;
}

/*
 * Class:     daal_NumericTableImpl
 * Method:    getNumberOfRows
 * Signature:()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cGetNumberOfRows
(JNIEnv *env, jobject thisObj)
{
    // Get a class reference for Java NumericTable
    jclass cls = env->FindClass("com/intel/daal/data_management/data/NumericTable");

    jfieldID objFieldID = env->GetFieldID(cls, "cObject", "J");
    jlong cObj = env->GetLongField(thisObj, objFieldID);

    NumericTable *tbl = ((NumericTablePtr *)cObj)->get();

    jlong nRows = tbl->getNumberOfRows();
    return nRows;
}

/*
 * Class:     com_intel_daal_data_1management_data_NumericTableImpl
 * Method:    setNumberOfRows
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cSetNumberOfRows
(JNIEnv *env, jobject thisObj, jlong nRow)
{
    NumericTable *table = getNIONumericTableObject(env, thisObj)->get();
    DAAL_CHECK_THROW(table->resize((size_t)nRow));
}

/*
 * Class:     com_intel_daal_data_1management_data_NumericTableImpl
 * Method:    setNumberOfColumns
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cSetNumberOfColumns
(JNIEnv *env, jobject thisObj, jlong nCol)
{
    NumericTable *table = getNIONumericTableObject(env, thisObj)->get();
    DAAL_CHECK_THROW(table->getDictionary()->setNumberOfFeatures((size_t)nCol));
}

/*
 * Class:     com_intel_daal_data_1management_data_NumericTableImpl
 * Method:    isNormalized
 * Signature:(I)Z
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cIsNormalized
(JNIEnv *env, jobject thisObj, jint flag)
{
    NumericTable *table = getNIONumericTableObject(env, thisObj)->get();
    jboolean checkResult = (jboolean)table->isNormalized((NumericTableIface::NormalizationType)flag);
    return checkResult;
}

/*
 * Class:     com_intel_daal_data_1management_data_NumericTableImpl
 * Method:    setNormalizationFlag
 * Signature:(I)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cSetNormalizationFlag
(JNIEnv *env, jobject thisObj, jint flag)
{
    NumericTable *table = getNIONumericTableObject(env, thisObj)->get();
    jint oldFlag = (jint)table->setNormalizationFlag((NumericTableIface::NormalizationType)flag);
    return oldFlag;
}

/*
 * Class:     com_intel_daal_data_1management_data_NumericTableImpl
 * Method:    getDataLayout
 * Signature:(J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cGetDataLayout
(JNIEnv *env, jobject thisObj, jlong cObject)
{
    NumericTable *table = ((NumericTablePtr *)cObject)->get();
    return table->getDataLayout();
}

/*
 * Class:     com_intel_daal_data_1management_data_NumericTableImpl
 * Method:    getDataMemoryStatus
 * Signature:(J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cGetDataMemoryStatus
(JNIEnv *env, jobject thisObj, jlong cObject)
{
    NumericTable *table = ((NumericTablePtr *)cObject)->get();
    return table->getDataMemoryStatus();
}

/*
 * Class:     com_intel_daal_data_1management_data_NumericTableImpl
 * Method:    cGetNumberOfCategories
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cGetNumberOfCategories
(JNIEnv *env, jobject thisObj, jlong cObject, jint idx)
{
    NumericTable *table = ((NumericTablePtr *)cObject)->get();
    return table->getNumberOfCategories(idx);
}

/*
 * Class:     com_intel_daal_data_management_data_NumericTableImpl
 * Method:    cGetIntValue
 * Signature: (JJJ)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cGetIntValue
(JNIEnv *env, jobject thisObj, jlong cObject, jlong column, jlong row)
{
    NumericTable *table = ((NumericTablePtr *)cObject)->get();
    return table->getValue<int>(column, row);
}

/*
 * Class:     com_intel_daal_data_management_data_NumericTableImpl
 * Method:    cGetFloatValue
 * Signature: (JJJ)F
 */
JNIEXPORT jfloat JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cGetFloatValue
(JNIEnv *env, jobject thisObj, jlong cObject, jlong column, jlong row)
{
    NumericTable *table = ((NumericTablePtr *)cObject)->get();
    return table->getValue<float>(column, row);
}

/*
 * Class:     com_intel_daal_data_management_data_NumericTableImpl
 * Method:    cGetDoubleValue
 * Signature: (JJJ)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cGetDoubleValue
(JNIEnv *env, jobject thisObj, jlong cObject, jlong column, jlong row)
{
    NumericTable *table = ((NumericTablePtr *)cObject)->get();
    return table->getValue<double>(column, row);
}

/*
 * Class:     daal_NumericTableImpl
 * Method:    cNewJavaNumericTable
 * Signature:(JJIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cNewJavaNumericTable
(JNIEnv *env, jobject thisObj, jlong p, jlong n, jint layout, jint featuresEqual, jint tag)
{
    using namespace daal;

    JavaVM *jvm;
    // Get pointer to the Java VM interface function table
    jint status = env->GetJavaVM(&jvm);
    if(status != 0)
    {
        /* printf("Error: Couldn't get Java VM, code = %d\n",(int)status);
        fflush(0); */
        return 0;
    }

    // Create C++ object of the class NumericTable
    NumericTable *tbl = 0;
    switch(tag){
        case SERIALIZATION_JAVANIO_HOMOGEN_NT_ID:
            tbl = new daal::JavaNumericTable<SERIALIZATION_JAVANIO_HOMOGEN_NT_ID>((size_t)p,
                                                                                  (size_t)n,
                                                                                  jvm,
                                                                                  thisObj,
                                                                                  (NumericTableIface::StorageLayout)layout,
                                                                                  (DictionaryIface::FeaturesEqual)featuresEqual);
            break;
        case SERIALIZATION_JAVANIO_AOS_NT_ID:
            tbl = new daal::JavaNumericTable<SERIALIZATION_JAVANIO_AOS_NT_ID>((size_t)p,
                                                                              (size_t)n,
                                                                              jvm,
                                                                              thisObj,
                                                                              (NumericTableIface::StorageLayout)layout,
                                                                              (DictionaryIface::FeaturesEqual)featuresEqual);
            break;
        case SERIALIZATION_JAVANIO_SOA_NT_ID:
            tbl = new daal::JavaNumericTable<SERIALIZATION_JAVANIO_SOA_NT_ID>((size_t)p,
                                                                              (size_t)n,
                                                                              jvm,
                                                                              thisObj,
                                                                              (NumericTableIface::StorageLayout)layout,
                                                                              (DictionaryIface::FeaturesEqual)featuresEqual);
            break;
        case SERIALIZATION_JAVANIO_PACKEDSYMMETRIC_NT_ID:
            tbl = new daal::JavaNumericTable<SERIALIZATION_JAVANIO_PACKEDSYMMETRIC_NT_ID>((size_t)p,
                                                                                          (size_t)n,
                                                                                          jvm,
                                                                                          thisObj,
                                                                                          (NumericTableIface::StorageLayout)layout,
                                                                                          (DictionaryIface::FeaturesEqual)featuresEqual);
            break;
        case SERIALIZATION_JAVANIO_PACKEDTRIANGULAR_NT_ID:
            tbl = new daal::JavaNumericTable<SERIALIZATION_JAVANIO_PACKEDTRIANGULAR_NT_ID>((size_t)p,
                                                                                           (size_t)n,
                                                                                           jvm,
                                                                                           thisObj,
                                                                                           (NumericTableIface::StorageLayout)layout,
                                                                                           (DictionaryIface::FeaturesEqual)featuresEqual);
            break;
        default:
            break;
    }

    if(tbl->getErrors()->size() > 0)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), tbl->getErrors()->getDescription());
    }

    return(jlong)(new SerializationIfacePtr(tbl));
}

/*
 * Class:     com_intel_daal_data_1management_data_NumericTableImpl
 * Method:    cFreeByteBuffer
 * Signature: (Ljava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cFreeByteBuffer
(JNIEnv *env, jobject thisObj, jobject byteBuffer)
{
    daal::byte *buffer = (daal::byte *)(env->GetDirectBufferAddress(byteBuffer));
    daal_free(buffer);
}

/*
 * Class:     com_intel_daal_data_1management_data_NumericTableImpl
 * Method:    cGetCDataDictionary
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cGetCDataDictionary
(JNIEnv *env, jobject thisObj, jlong cTable)
{
    using namespace daal;
    NumericTablePtr *nt = (NumericTablePtr *)cTable;
    NumericTableDictionaryPtr *dict =
        new NumericTableDictionaryPtr((*nt)->getDictionarySharedPtr());
    return (jlong)dict;
}

/*
 * Class:     com_intel_daal_data_1management_data_NumericTableImpl
 * Method:    cSetCDataDictionary
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cSetCDataDictionary
(JNIEnv *env, jobject thisObj, jlong cTable, jlong cDictionary)
{
    using namespace daal;
    NumericTablePtr *nt = (NumericTablePtr *)cTable;
    NumericTableDictionaryPtr *dict = (NumericTableDictionaryPtr*)cDictionary;
    (*nt)->setDictionary(dict->get());
}
