/* file: abs.cpp */
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
//  Implementation of abs algorithm and types methods.
//--
*/

#include "abs_types.h"
#include "serialization_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace math
{
namespace abs
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_ABS_RESULT_ID);
/** Default constructor */
Input::Input() : daal::algorithms::Input(1) {};

/**
 * Returns an input object for the absolute value function
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr Input::get(InputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for the absolute value function
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks an input object for the absolute value function
 * \param[in] par     Function parameter
 * \param[in] method  Computation method
 */
Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfInputNumericTables);

    data_management::NumericTablePtr inTable = get(data);
    Status s;
    if(method == fastCSR)
    {
        int expectedLayouts = (int)data_management::NumericTableIface::csrArray;
        s |= data_management::checkNumericTable(inTable.get(), dataStr(), 0, expectedLayouts);
        if(!s) { return s; }
    }
    else
    {
        s |= data_management::checkNumericTable(inTable.get(), dataStr());
    }
    return s;
}

/** Default constructor */
Result::Result() : daal::algorithms::Result(1) {}

/**
 * Returns the result of the absolute value function
 * \param[in] id   Identifier of the result
 * \return         Result that corresponds to the given identifier
 */
data_management::NumericTablePtr Result::get(ResultId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the absolute value function
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Result
 */
void Result::set(ResultId id, const data_management::NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the result of the absolute value function
 * \param[in] in   %Input of the absolute value function
 * \param[in] par     %Parameter of the absolute value function
 * \param[in] method  Computation method of the absolute value function
 */
Status Result::check(const daal::algorithms::Input *in, const daal::algorithms::Parameter *par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfOutputNumericTables);
    if(in == 0) { return Status(ErrorNullInput); }

    data_management::NumericTablePtr dataTable = (static_cast<const Input *>(in))->get(data);
    data_management::NumericTablePtr resultTable = get(value);

    Status s;
    if(method == fastCSR)
    {
        int expectedLayouts = (int)data_management::NumericTableIface::csrArray;
        s |= data_management::checkNumericTable(dataTable.get(), dataStr(), 0, expectedLayouts);
        if(!s) { return s; }

        size_t nDataRows = dataTable->getNumberOfRows();
        size_t nDataColumns = dataTable->getNumberOfColumns();

        s |= data_management::checkNumericTable(resultTable.get(), valueStr(), 0, expectedLayouts, nDataColumns, nDataRows);
        if(!s) { return s; }

        services::SharedPtr<data_management::CSRNumericTableIface> inputTable =
            services::dynamicPointerCast<data_management::CSRNumericTableIface, data_management::NumericTable>(dataTable);

        services::SharedPtr<data_management::CSRNumericTableIface> resTable =
            services::dynamicPointerCast<data_management::CSRNumericTableIface, data_management::NumericTable>(resultTable);

        size_t inSize = inputTable->getDataSize();
        size_t resSize = resTable->getDataSize();

        if(inSize != resSize) { return Status(ErrorIncorrectSizeOfArray); }
    }
    else
    {
        s |= data_management::checkNumericTable(dataTable.get(), dataStr());
        if(!s) { return s; }

        size_t nDataRows = dataTable->getNumberOfRows();
        size_t nDataColumns = dataTable->getNumberOfColumns();

        int unexpectedLayouts = (int)data_management::NumericTableIface::upperPackedSymmetricMatrix |
                                (int)data_management::NumericTableIface::lowerPackedSymmetricMatrix |
                                (int)data_management::NumericTableIface::upperPackedTriangularMatrix |
                                (int)data_management::NumericTableIface::lowerPackedTriangularMatrix |
                                (int)data_management::NumericTableIface::csrArray;
        s |= data_management::checkNumericTable(resultTable.get(), valueStr(), unexpectedLayouts, 0, nDataColumns, nDataRows);
    }
    return s;
}

}// namespace interface1
}// namespace abs
}// namespace math
}// namespace algorithms
}// namespace daal
