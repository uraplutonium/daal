/* file: keyvaluedatacollection.cpp */
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

#include "daal.h"

#include "JKeyValueDataCollection.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_management_data_KeyValueDataCollection
 * Method:    cNewDataCollection
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_KeyValueDataCollection_cNewDataCollection
  (JNIEnv *env, jobject thisObj)
{
    KeyValueDataCollection *collection = new KeyValueDataCollection();
    return (jlong)(new services::SharedPtr<SerializationIface>(collection));
}

/*
 * Class:     com_intel_daal_data_management_data_KeyValueDataCollection
 * Method:    cGetValue
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_KeyValueDataCollection_cGetValue
  (JNIEnv *env, jobject thisObj, jlong collectionAddr, jint key)
{
    services::SharedPtr<SerializationIface> *collectionShPtr = (services::SharedPtr<SerializationIface> *)collectionAddr;
    KeyValueDataCollection *collection = static_cast<KeyValueDataCollection *>(collectionShPtr->get());
    services::SharedPtr<SerializationIface> *value = new services::SharedPtr<SerializationIface>((*collection)[(size_t)key]);
    return (jlong)value;
}

/*
 * Class:     com_intel_daal_data_management_data_KeyValueDataCollection
 * Method:    cSetValue
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_KeyValueDataCollection_cSetValue
  (JNIEnv *env, jobject thisObj, jlong collectionAddr, jint key, jlong valueAddr)
{
    services::SharedPtr<SerializationIface> *collectionShPtr = (services::SharedPtr<SerializationIface> *)collectionAddr;
    services::SharedPtr<SerializationIface> *valueShPtr = (services::SharedPtr<SerializationIface> *)valueAddr;
    KeyValueDataCollection *collection = static_cast<KeyValueDataCollection *>(collectionShPtr->get());
    (*collection)[(size_t)key] = *valueShPtr;
}

/*
 * Class:     com_intel_daal_data_management_data_KeyValueDataCollection
 * Method:    cSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_KeyValueDataCollection_cSize
  (JNIEnv *env, jobject thisObj, jlong collectionAddr)
{
    services::SharedPtr<SerializationIface> *collectionShPtr = (services::SharedPtr<SerializationIface> *)collectionAddr;
    KeyValueDataCollection *collection = static_cast<KeyValueDataCollection *>(collectionShPtr->get());
    return (jlong)(collection->size());
}

/*
 * Class:     com_intel_daal_data_management_data_KeyValueDataCollection
 * Method:    cGetKeyByIndex
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_KeyValueDataCollection_cGetKeyByIndex
  (JNIEnv *env, jobject thisObj, jlong collectionAddr, jint index)
{
    services::SharedPtr<SerializationIface> *collectionShPtr = (services::SharedPtr<SerializationIface> *)collectionAddr;
    KeyValueDataCollection *collection = static_cast<KeyValueDataCollection *>(collectionShPtr->get());
    size_t key = collection->getKeyByIndex((size_t)index);
    return (jlong)(key);
}

/*
 * Class:     com_intel_daal_data_management_data_KeyValueDataCollection
 * Method:    cGetValueByIndex
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_KeyValueDataCollection_cGetValueByIndex
  (JNIEnv *env, jobject thisObj, jlong collectionAddr, jint index)
{
    services::SharedPtr<SerializationIface> *collectionShPtr = (services::SharedPtr<SerializationIface> *)collectionAddr;
    KeyValueDataCollection *collection = static_cast<KeyValueDataCollection *>(collectionShPtr->get());
    services::SharedPtr<SerializationIface> *valueShPtr = new services::SharedPtr<SerializationIface>(
        collection->getValueByIndex((size_t)index));
    return (jlong)valueShPtr;
}
