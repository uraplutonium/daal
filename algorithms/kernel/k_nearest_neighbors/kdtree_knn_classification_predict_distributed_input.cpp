/* file: kdtree_knn_classification_predict_distributed_input.cpp */
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
#include "service_defines.h"
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

DistributedInput<step1Local>::DistributedInput() : classifier::prediction::Input() {}

NumericTablePtr DistributedInput<step1Local>::get(NumericTableInputStep1Id id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

SharedPtr<PartialModel> DistributedInput<step1Local>::get(PartialModelInputId id) const
{
    return staticPointerCast<PartialModel, SerializationIface>(Argument::get(id));
}

void DistributedInput<step1Local>::set(NumericTableInputStep1Id id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

void DistributedInput<step1Local>::set(PartialModelInputId id, const SharedPtr<PartialModel> & ptr)
{
    Argument::set(id, ptr);
}

services::Status DistributedInput<step1Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    return checkImpl(parameter);
}

services::Status DistributedInput<step1Local>::checkImpl(const daal::algorithms::Parameter * parameter) const
{
    services::Status s;

    if (parameter)
    {
        DAAL_ASSERT(dynamic_cast<const Parameter *>(parameter));
        const Parameter * const algParameter = static_cast<const Parameter *>(parameter);
        if (algParameter->nClasses < 2) { return s.add(services::ErrorIncorrectNumberOfClasses); }
    }

    NumericTablePtr dataTable = get(data);
    DAAL_CHECK_STATUS(s, checkNumericTable(dataTable.get(), dataStr()));

    const PartialModelConstPtr m = get(partialModel);
    if (!m) { return s.add(services::ErrorNullModel); }

    const size_t trainingDataFeatures = m->getNumberOfFeatures();
    const size_t predictionDataFeatures = dataTable->getNumberOfColumns();
    if (trainingDataFeatures != predictionDataFeatures)
    {
        return s.add(services::Error::create(services::ErrorIncorrectNumberOfColumns, services::ArgumentName, dataStr()));
    }

    return s;
}

DistributedInput<step2Local>::DistributedInput() : daal::algorithms::Input(7)
{
    Argument::set(communicationResponses, KeyValueDataCollectionPtr(new KeyValueDataCollection()));
    Argument::set(communicationInputQueries, KeyValueDataCollectionPtr(new KeyValueDataCollection()));
}

NumericTablePtr DistributedInput<step2Local>::get(NumericTableInputStep2Id id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

int DistributedInput<step2Local>::get(IntInputStep2Id id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id))->getValue<int>(0, 0);
}

data_management::KeyValueDataCollectionPtr DistributedInput<step2Local>::get(NumericTableInputStep2PerNodeId id) const
{
    return staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
}

SharedPtr<PartialModel> DistributedInput<step2Local>::get(PartialModelInputId id) const
{
    return staticPointerCast<PartialModel, SerializationIface>(Argument::get(id));
}

void DistributedInput<step2Local>::set(NumericTableInputStep2Id id, const NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

void DistributedInput<step2Local>::set(IntInputStep2Id id, int value)
{
    typedef HomogenNumericTable<int> HNT;
    SharedPtr<HNT> table(new HNT(1, 1, NumericTable::doAllocate, value));
    Argument::set(id, table);
}

void DistributedInput<step2Local>::set(NumericTableInputStep2PerNodeId id, const data_management::KeyValueDataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

void DistributedInput<step2Local>::set(PartialModelInputId id, const SharedPtr<PartialModel> & ptr)
{
    Argument::set(id, ptr);
}

void DistributedInput<step2Local>::add(NumericTableInputStep2PerNodeId id, size_t key, const data_management::NumericTablePtr & value)
{
    const KeyValueDataCollectionPtr collection = staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
    (*collection)[key] = value;
}

size_t DistributedInput<step2Local>::getNumberOfRows() const
{
    const data_management::NumericTablePtr dataTable = get(arrangedData);
    return dataTable ? dataTable->getNumberOfRows() : 0;
}

size_t DistributedInput<step2Local>::getNumberOfColumns() const
{
    const data_management::NumericTablePtr dataTable = get(arrangedData);
    return dataTable ? dataTable->getNumberOfColumns() : 0;
}

services::Status DistributedInput<step2Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    return checkImpl(parameter);
}

services::Status DistributedInput<step2Local>::checkImpl(const daal::algorithms::Parameter * parameter) const
{
    services::Status s;

    if (parameter)
    {
        DAAL_ASSERT(dynamic_cast<const Parameter *>(parameter));
        const Parameter * const algParameter = static_cast<const Parameter *>(parameter);
        if (algParameter->nClasses < 2) { return s.add(services::ErrorIncorrectNumberOfClasses); }
    }

    NumericTablePtr dataTable = get(arrangedData);
    DAAL_CHECK_STATUS(s, checkNumericTable(dataTable.get(), dataStr()));

    const PartialModelConstPtr m = get(partialModel);
    if (!m) { return s.add(services::ErrorNullModel); }

    const size_t trainingDataFeatures = m->getNumberOfFeatures();
    const size_t predictionDataFeatures = dataTable->getNumberOfColumns();
    if (trainingDataFeatures != predictionDataFeatures)
    {
        return s.add(services::Error::create(services::ErrorIncorrectNumberOfColumns, services::ArgumentName, dataStr()));
    }

    return s;
}

} // namespace interface1
} // namespace prediction
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal
