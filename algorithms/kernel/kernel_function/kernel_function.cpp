/* file: kernel_function.cpp */
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

/*
//++
//  Implementation of kernel function algorithm and types methods.
//--
*/

#include "kernel_function_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_KERNEL_FUNCTION_RESULT_ID);

ParameterBase::ParameterBase(size_t rowIndexX, size_t rowIndexY, size_t rowIndexResult, ComputationMode computationMode) :
    rowIndexX(rowIndexX), rowIndexY(rowIndexY), rowIndexResult(rowIndexResult), computationMode(computationMode) {}

Input::Input() : daal::algorithms::Input(lastInputId + 1) {}
Input::Input(const Input& other) : daal::algorithms::Input(other){}

/**
* Returns the input object of the kernel function algorithm
* \param[in] id    Identifier of the input object
* \return          %Input object that corresponds to the given identifier
*/
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
* Sets the input object of the kernel function algorithm
* \param[in] id    Identifier of the input object
* \param[in] ptr   Pointer to the input object
*/
void Input::set(InputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

Status Input::checkCSR() const
{
    Status s;
    const int csrLayout = (int)NumericTableIface::csrArray;

    DAAL_CHECK_STATUS(s, checkNumericTable(get(X).get(), XStr(), 0, csrLayout));

    const size_t nFeaturesX = get(X)->getNumberOfColumns();

    return checkNumericTable(get(Y).get(), YStr(), 0, csrLayout, nFeaturesX);
}

Status Input::checkDense() const
{
    Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(X).get(), XStr()));

    const size_t nFeaturesX = get(X)->getNumberOfColumns();

    return checkNumericTable(get(Y).get(), YStr(), 0, 0, nFeaturesX);
}
/**
 * Returns the result of the kernel function algorithm
 * \param[in] id   Identifier of the result
 * \return         Final result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the kernel function algorithm
 * \param[in] id    Identifier of the partial result
 * \param[in] ptr   Pointer to the object
 */
void Result::set(ResultId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
* Checks the result of the kernel function algorithm
* \param[in] input   %Input objects of the algorithm
* \param[in] par     %Parameter of the algorithm
* \param[in] method  Computation method of the algorithm
*/
Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    Status s;
    Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    ParameterBase *algParameter = static_cast<ParameterBase *>(const_cast<daal::algorithms::Parameter *>(par));

    const size_t nRowsX = algInput->get(X)->getNumberOfRows();
    const size_t nRowsY = algInput->get(Y)->getNumberOfRows();

    const int unexpectedLayouts = packed_mask;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(values).get(), valuesStr(), unexpectedLayouts, 0, 0, nRowsX));

    const size_t nVectorsValues = get(values)->getNumberOfRows();

    if(algParameter->rowIndexResult >= nVectorsValues)
    {
        return Status(Error::create(ErrorIncorrectParameter, ParameterName, rowIndexResultStr()));
    }
    if(algParameter->rowIndexX >= nRowsX)
    {
        return Status(Error::create(ErrorIncorrectParameter, ParameterName, rowIndexXStr()));
    }
    if(algParameter->rowIndexY >= nRowsY)
    {
        return Status(Error::create(ErrorIncorrectParameter, ParameterName, rowIndexYStr()));
    }
    return s;
}

}// namespace interface1
}// namespace kernel_function
}// namespace algorithms
}// namespace daal
