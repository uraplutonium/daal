/* file: one_hot_encoder.cpp */
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

#include "JOneHotEncoder.h"

#include "csv_feature_manager.h"

using namespace daal;
using namespace daal::data_management;

/*
 * Class:     com_intel_daal_data_1management_data_1source_OneHotEncoder
 * Method:    cDispose
 * Signature:(JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_1source_OneHotEncoder_cInit
(JNIEnv *env, jobject obj, jlong idx, jlong nCats)
{
    services::SharedPtr<ModifierIface>* ptr = new services::SharedPtr<ModifierIface>(new OneHotEncoder(idx, nCats));
    return (jlong)ptr;
}
