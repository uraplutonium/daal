/* file: implicit_als_train_init_distributed_input.cpp */
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
DistributedInput<step1Local>::DistributedInput() : implicit_als::training::init::Input() {}

DistributedInput<step2Local>::DistributedInput() : daal::algorithms::Input(1) {}

/**
 * Returns an input object for the implicit ALS initialization algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
KeyValueDataCollectionPtr DistributedInput<step2Local>::get(Step2LocalInputId id) const
{
    return KeyValueDataCollection::cast(Argument::get(id));
}

/**
 * Sets an input object for the implicit ALS initialization algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedInput<step2Local>::set(Step2LocalInputId id, const KeyValueDataCollectionPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the input objects and parameters of the implicit ALS initialization algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step2Local>::check(const daal::algorithms::Parameter *parameter, int method) const
{

    DAAL_CHECK_EX(get(inputOfStep2FromStep1).get(), ErrorNullInputDataCollection, ArgumentName, inputOfStep2FromStep1Str());

    KeyValueDataCollection &collection = *(get(inputOfStep2FromStep1));
    size_t nParts = collection.size();
    DAAL_CHECK_EX(nParts > 0, ErrorNullInputDataCollection, ArgumentName, inputOfStep2FromStep1Str());

    for (size_t i = 0; i < nParts; i++)
    {
        DAAL_CHECK_EX(dynamic_cast<CSRNumericTableIface *>(collection[i].get()), ErrorIncorrectElementInNumericTableCollection, ArgumentName, inputOfStep2FromStep1Str());
    }

    int expectedLayout = (int)NumericTableIface::csrArray;

    services::Status s = checkNumericTable(NumericTable::cast(collection[0]).get(), inputOfStep2FromStep1Str(), 0, expectedLayout);
    if(!s)
    {
        return services::Status(Error::create(ErrorIncorrectElementInNumericTableCollection, ElementInCollection, 0));
    }

    size_t nRows = NumericTable::cast(collection[0])->getNumberOfRows();

    for (size_t i = 1; i < nParts; i++)
    {
        s = checkNumericTable(NumericTable::cast(collection[i]).get(), inputOfStep2FromStep1Str(), 0, expectedLayout, 0, nRows);
        if(!s)
        {
            return services::Status(Error::create(ErrorIncorrectElementInNumericTableCollection, ElementInCollection, i));
        }
    }
    return services::Status();
}

}// namespace interface1
}// namespace init
}// namespace training
}// namespace implicit_als
}// namespace algorithms
}// namespace daal
