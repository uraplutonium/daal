/* file: low_order_moments_partial_result.cpp */
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
//  Implementation of LowOrderMoments classes.
//--
*/

#include "algorithms/moments/low_order_moments_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(PartialResult, SERIALIZATION_MOMENTS_PARTIAL_RESULT_ID);

PartialResult::PartialResult() : daal::algorithms::PartialResult(lastPartialResultId + 1) {}

/**
 * Gets the number of columns in the partial result of the low order %moments algorithm
 * \return Number of columns in the partial result
 */
Status PartialResult::getNumberOfColumns(size_t& nCols) const
{
    NumericTablePtr ntPtr = NumericTable::cast(Argument::get(partialMinimum));
    Status s = checkNumericTable(ntPtr.get(), partialMinimumStr());
    nCols = (s ? ntPtr->getNumberOfColumns() : 0);
    return s;
}

/**
 * Returns the partial result of the low order %moments algorithm
 * \param[in] id   Identifier of the partial result, \ref PartialResultId
 * \return Partial result that corresponds to the given identifier
 */
NumericTablePtr PartialResult::get(PartialResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the partial result of the low order %moments algorithm
 * \param[in] id    Identifier of the partial result
 * \param[in] ptr   Pointer to the partial result
 */
void PartialResult::set(PartialResultId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks correctness of the partial result
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method
 */
services::Status PartialResult::check(const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s;
    int unexpectedLayouts = (int)NumericTableIface::csrArray;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(nObservations).get(), nObservationsStr(), unexpectedLayouts, 0, 1, 1));

    unexpectedLayouts = (int)packed_mask;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(partialMinimum).get(), partialMinimumStr(), unexpectedLayouts));

    size_t nFeatures = get(partialMinimum)->getNumberOfColumns();
    return checkImpl(nFeatures);
}

/**
 * Checks  the correctness of partial result
 * \param[in] input     Pointer to the structure with input objects
 * \param[in] parameter Pointer to the structure of algorithm parameters
 * \param[in] method    Computation method
 */
services::Status PartialResult::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s;
    size_t nFeatures = 0;
    DAAL_CHECK_STATUS(s, static_cast<const InputIface *>(input)->getNumberOfColumns(nFeatures));

    const int unexpectedLayouts = (int)NumericTableIface::csrArray;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(nObservations).get(), nObservationsStr(), unexpectedLayouts, 0, 1, 1));
    return checkImpl(nFeatures);
}

services::Status PartialResult::checkImpl(size_t nFeatures) const
{
    services::Status s;
    const int unexpectedLayouts = (int)packed_mask;
    const char* errorMessages[] = {partialMinimumStr(), partialMaximumStr(), partialSumStr(),
        partialSumSquaresStr(), partialSumSquaresCenteredStr() };

    for(size_t i = 1; i < lastPartialResultId + 1; i++)
        DAAL_CHECK_STATUS(s, checkNumericTable(get((PartialResultId)i).get(), errorMessages[i - 1],
            unexpectedLayouts, 0, nFeatures, 1));
    return s;
}


Parameter::Parameter(EstimatesToCompute  _estimatesToCompute) : estimatesToCompute(_estimatesToCompute)
{}

services::Status Parameter::check() const
{
    return Status();
}

} // namespace interface1
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal
