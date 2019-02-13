/* file: apriori.cpp */
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
//  Implementation of the interface for the association rules algorithm
//--
*/

#include "apriori_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace association_rules
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_ASSOCIATION_RULES_RESULT_ID);

Parameter::Parameter(double minSupport, double minConfidence, size_t nUniqueItems, size_t nTransactions, bool discoverRules,
                        ItemsetsOrder itemsetsOrder, RulesOrder rulesOrder, size_t minSize, size_t maxSize) :
        minSupport(minSupport),
        minConfidence(minConfidence),
        nUniqueItems(nUniqueItems),
        nTransactions(nTransactions),
        discoverRules(discoverRules),
        itemsetsOrder(itemsetsOrder),
        rulesOrder(rulesOrder),
        minItemsetSize(minSize),
        maxItemsetSize(maxSize)
    {}

/**
 * Checks parameters of the association rules algorithm
 */
Status Parameter::check() const
{
    DAAL_CHECK_EX(minSupport >= 0 && minSupport < 1, ErrorIncorrectParameter, ParameterName, minSupportStr());
    DAAL_CHECK_EX(minConfidence >= 0 && minConfidence < 1, ErrorIncorrectParameter, ParameterName, minConfidenceStr());
    DAAL_CHECK_EX(minItemsetSize <= maxItemsetSize, ErrorIncorrectParameter, ParameterName, minItemsetSizeStr());
    return Status();
}

Input::Input() : daal::algorithms::Input(lastInputId + 1) {}

/**
 * Returns the input object of the association rules algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the input object of the association rules algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks parameters of the association rules algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method of the algorithm
 */
Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    DAAL_CHECK(Argument::size() == (lastInputId + 1), ErrorIncorrectNumberOfInputNumericTables);

    const int unexpectedLayouts = (int)NumericTableIface::upperPackedSymmetricMatrix  |
                            (int)NumericTableIface::lowerPackedSymmetricMatrix  |
                            (int)NumericTableIface::upperPackedTriangularMatrix |
                            (int)NumericTableIface::lowerPackedTriangularMatrix;
    const size_t nColumns = 2;
    Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(data).get(), dataStr(), unexpectedLayouts, 0, nColumns));

    const size_t nRows = get(data)->getNumberOfRows();
    const Parameter *algParameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));

    DAAL_CHECK_EX(algParameter->nUniqueItems <= nRows, ErrorIncorrectParameter, ParameterName, nUniqueItemsStr());
    DAAL_CHECK_EX(algParameter->nTransactions <= nRows, ErrorIncorrectParameter, ParameterName, nTransactionsStr());
    DAAL_CHECK_EX((algParameter->maxItemsetSize <= nRows) && (algParameter->nUniqueItems <= 0 || algParameter->maxItemsetSize <= algParameter->nUniqueItems),
        ErrorIncorrectParameter, ParameterName, maxItemsetSizeStr());
    return s;
}


Result::Result() : daal::algorithms::Result((lastResultId + 1)) {}

/**
 * Returns the final result of the association rules algorithm
 * \param[in] id   Identifier of the result
 * \return         Final result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the final result of the association rules algorithm
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the result
 */
void Result::set(ResultId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the result of the association rules algorithm
 * \param[in] input   %Input of algorithm
 * \param[in] par     %Parameter of algorithm
 * \param[in] method  Computation method of the algorithm
 */
Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    Status s;
    const Parameter *algParameter = static_cast<const Parameter *>(par);
    const int unexpectedLayouts = packed_mask;
    DAAL_CHECK_STATUS(s, checkNumericTable(get(largeItemsets).get(), largeItemsetsStr(), unexpectedLayouts, 0, 2, 0, false));
    DAAL_CHECK_STATUS(s, checkNumericTable(get(largeItemsetsSupport).get(), largeItemsetsSupportStr(), unexpectedLayouts, 0, 2, 0, false));

    if(algParameter->discoverRules)
    {
        DAAL_CHECK_STATUS(s, checkNumericTable(get(antecedentItemsets).get(), antecedentItemsetsStr(), unexpectedLayouts, 0, 2, 0, false));
        DAAL_CHECK_STATUS(s, checkNumericTable(get(consequentItemsets).get(), consequentItemsetsStr(), unexpectedLayouts, 0, 2, 0, false));
        DAAL_CHECK_STATUS(s, checkNumericTable(get(confidence).get(), confidenceStr(), unexpectedLayouts, 0, 1, 0, false));
    }

    return s;
}

}// namespace interface1
}// namespace association_rules
}// namespace algorithms
}// namespace daal
