/* file: kdtree_knn_classification_predict_distr.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
* All Rights Reserved.
*
* If this  software was obtained  under the  Intel Simplified  Software License,
* the following terms apply:
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*
*
* If this  software  was obtained  under the  Apache License,  Version  2.0 (the
* "License"), the following terms apply:
*
* You may  not use this  file except  in compliance  with  the License.  You may
* obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
*
*
* Unless  required  by   applicable  law  or  agreed  to  in  writing,  software
* distributed under the License  is distributed  on an  "AS IS"  BASIS,  WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*
* See the   License  for the   specific  language   governing   permissions  and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of the interface for K-Nearest Neighbors (kNN) model-based prediction
//--
*/

#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_predict_types.h"
#include "kdtree_knn_classification_model_impl.h"
#include "serialization_utils.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace prediction
{
namespace interface1
{

using namespace daal::data_management;
using namespace daal::services;

__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep1, SERIALIZATION_K_NEAREST_NEIGHBOR_PREDICTION_DISTRIBUTED_PARTIAL_RESULT_STEP1_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep2, SERIALIZATION_K_NEAREST_NEIGHBOR_PREDICTION_DISTRIBUTED_PARTIAL_RESULT_STEP2_ID);

DistributedPartialResultStep1::DistributedPartialResultStep1() : daal::algorithms::PartialResult(1) {}

NumericTablePtr DistributedPartialResultStep1::get(DistributedPartialResultStep1Id id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep1::set(DistributedPartialResultStep1Id id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

services::Status DistributedPartialResultStep1::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                      int method) const
{
    const kdtree_knn_classification::prediction::DistributedInput<step1Local> * const in
        = static_cast<const kdtree_knn_classification::prediction::DistributedInput<step1Local> *>(input);
    const size_t numberOfRows = in->getNumberOfRows();

    return checkNumericTable(get(kdtree_knn_classification::prediction::keys).get(), keysStr(), 0, 0, 1, numberOfRows);
}

DistributedPartialResultStep2::DistributedPartialResultStep2() : daal::algorithms::PartialResult(3) {}

NumericTablePtr DistributedPartialResultStep2::get(DistributedPartialResultStep2Id id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

data_management::NumericTablePtr DistributedPartialResultStep2::get(DistributedPartialResultStep2PerNodeId id, int key) const
{
    const data_management::KeyValueDataCollectionPtr collection = get(id);
    return staticPointerCast<NumericTable, SerializationIface>((*collection)[key]);
}

data_management::KeyValueDataCollectionPtr DistributedPartialResultStep2::get(DistributedPartialResultStep2PerNodeId id) const
{
    return staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep2::set(DistributedPartialResultStep2Id id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

void DistributedPartialResultStep2::set(DistributedPartialResultStep2PerNodeId id, const data_management::KeyValueDataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

services::Status DistributedPartialResultStep2::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                      int method) const
{
    const kdtree_knn_classification::prediction::DistributedInput<step2Local> * const in
        = static_cast<const kdtree_knn_classification::prediction::DistributedInput<step2Local> *>(input);
    const size_t numberOfRows = in->getNumberOfRows();

    return checkNumericTable(get(kdtree_knn_classification::prediction::prediction).get(), predictionStr(), 0, 0, 1, numberOfRows);
}

} // namespace interface1
} // namespace prediction
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal
