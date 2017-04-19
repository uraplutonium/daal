/* file: implicit_als_train_init_partial_result.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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

/*
//++
//  Implementation of implicit als algorithm and types methods.
//--
*/

#include "implicit_als_training_init_types.h"
#include "implicit_als_train_init_parameter.h"
#include "serialization_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(PartialResultBase, SERIALIZATION_IMPLICIT_ALS_TRAINING_INIT_PARTIAL_RESULT_BASE_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(PartialResult, SERIALIZATION_IMPLICIT_ALS_TRAINING_INIT_PARTIAL_RESULT_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep2, SERIALIZATION_IMPLICIT_ALS_TRAINING_INIT_DISTRIBUTED_PARTIAL_RESULT_STEP2_ID);


PartialResultBase::PartialResultBase(size_t nElements) : daal::algorithms::PartialResult(nElements) {}

KeyValueDataCollectionPtr PartialResultBase::get(PartialResultBaseId id) const
{
    return services::staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
}

NumericTablePtr PartialResultBase::get(PartialResultBaseId id, size_t key) const
{
    KeyValueDataCollectionPtr collection = get(id);
    NumericTablePtr nt;
    if (collection)
    {
        nt = NumericTable::cast((*collection)[key]);
    }
    return nt;
}

void PartialResultBase::set(PartialResultBaseId id, const KeyValueDataCollectionPtr &ptr)
{
    Argument::set(id, ptr);
}

PartialResult::PartialResult() : PartialResultBase(4) {}

/**
 * Returns a partial result of the implicit ALS initialization algorithm
 * \param[in] id    Identifier of the partial result
 * \return          Partial result that corresponds to the given identifier
 */
services::SharedPtr<PartialModel> PartialResult::get(PartialResultId id) const
{
    return services::staticPointerCast<PartialModel, SerializationIface>(Argument::get(id));
}

/**
 * Sets a partial result of the implicit ALS initialization algorithm
 * \param[in] id    Identifier of the partial result
 * \param[in] ptr   Pointer to the partial result
 */
void PartialResult::set(PartialResultId id, const services::SharedPtr<PartialModel> &ptr)
{
    Argument::set(id, ptr);
}

KeyValueDataCollectionPtr PartialResult::get(PartialResultCollectionId id) const
{
    return services::staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
}

NumericTablePtr PartialResult::get(PartialResultCollectionId id, size_t key) const
{
    KeyValueDataCollectionPtr collection = get(id);
    NumericTablePtr nt;
    if (collection)
    {
        nt = NumericTable::cast((*collection)[key]);
    }
    return nt;
}

void PartialResult::set(PartialResultCollectionId id, const KeyValueDataCollectionPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks a partial result of the implicit ALS initialization algorithm
 * \param[in] input       %Input object for the algorithm
 * \param[in] parameter   %Parameter of the algorithm
 * \param[in] method      Computation method of the algorithm
 */
services::Status PartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    DAAL_CHECK(method == fastCSR, ErrorMethodNotSupported);

    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    const DistributedInput<step1Local> *algInput = static_cast<const DistributedInput<step1Local> *>(input);
    size_t nRows = algInput->get(data)->getNumberOfRows();
    DAAL_CHECK_EX(algParameter->fullNUsers > nRows, ErrorIncorrectParameter, ParameterName, fullNUsersStr());

    PartialModelPtr model = get(partialModel);
    DAAL_CHECK(model, ErrorNullPartialModel);

    size_t nFactors = algParameter->nFactors;
    int unexpectedLayouts = (int)packed_mask;
    services::Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(model->getFactors().get(), factorsStr(), unexpectedLayouts, 0, nFactors, nRows));
    DAAL_CHECK_STATUS(s, checkNumericTable(model->getIndices().get(), indicesStr(), unexpectedLayouts, 0, 1, nRows));

    SharedPtr<HomogenNumericTable<int> > partitionTable = internal::getPartition(algParameter);
    size_t nParts = partitionTable->getNumberOfRows() - 1;
    int *partitionData = partitionTable->getArray();

    KeyValueDataCollectionPtr collection = get(outputOfStep1ForStep2);
    DAAL_CHECK_EX(collection.get(), ErrorNullPartialResult, ArgumentName, outputOfStep1ForStep2Str());
    DAAL_CHECK_EX(collection->size() == nParts, ErrorIncorrectDataCollectionSize, ArgumentName, outputOfStep1ForStep2Str());

    size_t resNCols = NumericTable::cast((*collection)[0])->getNumberOfColumns();
    DAAL_CHECK_EX(resNCols == nRows, ErrorIncorrectNumberOfRows, ArgumentName, outputOfStep1ForStep2Str());
    int expectedLayout = (int)NumericTableIface::csrArray;
    for (size_t i = 0; i < nParts; i++)
    {
        NumericTable *nt = NumericTable::cast((*collection)[i]).get();
        size_t resNRows = partitionData[i + 1] - partitionData[i];
        DAAL_CHECK_STATUS(s, checkNumericTable(nt, outputOfStep1ForStep2Str(), 0, expectedLayout, resNCols, resNRows, false));
    }
    return s;
}

DistributedPartialResultStep2::DistributedPartialResultStep2() : PartialResultBase(3) {}

NumericTablePtr DistributedPartialResultStep2::get(DistributedPartialResultStep2Id id) const
{
    return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void DistributedPartialResultStep2::set(DistributedPartialResultStep2Id id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

services::Status DistributedPartialResultStep2::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    DAAL_CHECK(method == fastCSR, ErrorMethodNotSupported);

    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    const DistributedInput<step2Local> *algInput = static_cast<const DistributedInput<step2Local> *>(input);

    KeyValueDataCollectionPtr collection = algInput->get(inputOfStep2FromStep1);
    size_t nParts = collection->size();
    size_t nRows = NumericTable::cast((*collection)[0])->getNumberOfRows();
    size_t nCols = 0;
    for (size_t i = 0; i < nParts; i++)
    {
        nCols += NumericTable::cast((*collection)[i])->getNumberOfColumns();
    }

    int expectedLayout = (int)NumericTableIface::csrArray;
    return checkNumericTable(get(transposedData).get(), transposedDataStr(), 0, expectedLayout, nCols, nRows, false);
}

}// namespace interface1
}// namespace init
}// namespace training
}// namespace implicit_als
}// namespace algorithms
}// namespace daal
