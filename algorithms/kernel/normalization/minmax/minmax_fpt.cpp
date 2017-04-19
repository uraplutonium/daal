/* file: minmax_fpt.cpp */
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
//  Implementation of minmax algorithm and types methods.
//--
*/

#include "minmax_types.h"

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace minmax
{
namespace interface1
{
/**
 * Allocates memory to store the result of the minmax normalization algorithm
 * \param[in] input  %Input object for the minmax normalization algorithm
 * \param[in] par    %Parameter of the minmax normalization algorithm
 * \param[in] method Computation method of the minmax normalization algorithm
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, int method)
{
    if(!input) return services::Status(services::ErrorNullInput);

    const Input *algInput = static_cast<const Input *>(input);
    data_management::NumericTablePtr dataTable = algInput->get(data);

    services::Status s;
    DAAL_CHECK_STATUS(s, data_management::checkNumericTable(dataTable.get(), dataStr()));

    size_t nRows = dataTable->getNumberOfRows();
    size_t nColumns = dataTable->getNumberOfColumns();
    data_management::NumericTablePtr normalizedDataTable(new data_management::HomogenNumericTable<algorithmFPType>(
                                                         nColumns, nRows, data_management::NumericTable::doAllocate));
    set(normalizedData, normalizedDataTable);
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, int method);

}// namespace interface1
}// namespace minmax
}// namespace normalization
}// namespace algorithms
}// namespace daal
