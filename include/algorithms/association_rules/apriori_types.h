/* file: apriori_types.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Association rules parameter structure
//--
*/

#ifndef __APRIORI_TYPES_H__
#define __APRIORI_TYPES_H__

#include "services/daal_defines.h"
#include "algorithms/algorithm.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for the association rules algorithm
 */
namespace association_rules
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__ASSOCIATION_RULES__METHOD"></a>
 * Available methods for finding large itemsets and association rules
 */
enum Method
{
    apriori = 0,         /*!< Apriori method */
    defaultDense = 0     /*!< Apriori default method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__ASSOCIATION_RULES__ITEMSETSFORMAT"></a>
 * Available sort order options for resulting itemsets
 */
enum ItemsetsOrder
{
    itemsetsUnsorted = 0,           /*!< Unsorted */
    itemsetsSortedBySupport = 1     /*!< Sorted by the support value */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__ASSOCIATION_RULES__FORMAT"></a>
 * Available sort order options for resulting association rules
 */
enum RulesOrder
{
    rulesUnsorted = 0,              /*!< Unsorted */
    rulesSortedByConfidence = 1     /*!< Sorted by the confidence value */
};


/**
 * <a name="DAAL-ENUM-ALGORITHMS__ASSOCIATION_RULES__INPUTID"></a>
 * Available identifiers of input objects for the association rules algorithm
 */
enum InputId
{
    data = 0           /*!< %Input data table */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__ASSOCIATION_RULES__RESULTID"></a>
 * Available identifiers of results for the association rules algorithm
 */
enum ResultId
{
    largeItemsets        = 0,       /*!< Large itemsets            */
    largeItemsetsSupport = 1,       /*!< Support of large itemsets */
    antecedentItemsets   = 2,       /*!< Antecedent itemsets       */
    consequentItemsets   = 3,       /*!< Consequent itemsets       */
    confidence           = 4        /*!< Confidence                */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ASSOCIATION_RULES__PARAMETER"></a>
 * \brief Parameters for the association rules compute() method
 *
 * \snippet association_rules/apriori_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    Parameter(double minSupport = 0.01, double minConfidence = 0.6, size_t nUniqueItems = 0, size_t nTransactions = 0,
              bool discoverRules = true, ItemsetsOrder itemsetsOrder = itemsetsUnsorted,
              RulesOrder rulesOrder = rulesUnsorted, size_t minSize = 0, size_t maxSize = 0) :
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

    double         minSupport;          /*!< Minimum support    0.0 <= minSupport    < 1.0 */
    double         minConfidence;       /*!< Minimum confidence 0.0 <= minConfidence < 1.0 */
    size_t         nUniqueItems;        /*!< Number of unique items */
    size_t         nTransactions;       /*!< Number of transactions */
    bool           discoverRules;       /*!< Flag. If true, association rules are built from large itemsets */
    ItemsetsOrder  itemsetsOrder;       /*!< Format of the resulting itemsets */
    RulesOrder     rulesOrder;          /*!< Format of the resulting association rules */
    size_t         minItemsetSize;      /*!< Minimum number of items in a large itemset */
    size_t         maxItemsetSize;      /*!< Maximum number of items in a large itemset.
                                             Set to zero to not limit the upper boundary for the size of large itemsets */
};
/* [Parameter source code] */


/**
 * <a name="DAAL-CLASS-ALGORITHMS__ASSOCIATION_RULES__INPUT"></a>
 * \brief %Input for the association rules algorithm
 */
class Input : public daal::algorithms::Input
{
public:
    Input() : daal::algorithms::Input(1) {}

    virtual ~Input() {}

    /**
     * Returns the input object of the association rules algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the input object of the association rules algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks parameters of the association rules algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method of the algorithm
     */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        services::SharedPtr<data_management::NumericTable> inTable = get(data);

        if(inTable.get() == 0) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        if(inTable->getNumberOfRows() == 0) { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(inTable->getNumberOfColumns() == 0) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }

        Parameter *algParameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));
        if(algParameter->minConfidence < 0 || algParameter->minConfidence > 1 ||
           algParameter->minSupport    < 0 || algParameter->minSupport    > 1 )
        { this->_errors->add(services::ErrorIncorrectParameter); return; }
        return;
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ASSOCIATION_RULES__RESULT"></a>
 * \brief Results obtained with the compute() method of the association rules algorithm in the batch processing mode
 */
class Result : public daal::algorithms::Result
{
public:
    Result() : daal::algorithms::Result(5) {}

    virtual ~Result() {};

    /**
     * Allocates memory for storing Association Rules algorithm results
     * \param[in] input         Pointer to input structure
     * \param[in] parameter     Pointer to parameter structure
     * \param[in] method        Computation method of the algorithm
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        Parameter *algParameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(parameter));

        Argument::set(largeItemsets,
                      services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<size_t>(2, 0, data_management::NumericTableIface::notAllocate)));
        Argument::set(largeItemsetsSupport,
                      services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<size_t>(2, 0, data_management::NumericTableIface::notAllocate)));

        if(algParameter->discoverRules)
        {
            Argument::set(antecedentItemsets,
                          services::SharedPtr<data_management::SerializationIface>(
                              new data_management::HomogenNumericTable<size_t>(2, 0, data_management::NumericTableIface::notAllocate)));
            Argument::set(consequentItemsets,
                          services::SharedPtr<data_management::SerializationIface>(
                              new data_management::HomogenNumericTable<size_t>(2, 0, data_management::NumericTableIface::notAllocate)));
            Argument::set(confidence,
                          services::SharedPtr<data_management::SerializationIface>(
                              new data_management::HomogenNumericTable<algorithmFPType>(1, 0, data_management::NumericTableIface::notAllocate)));
        }
    }

    /**
     * Returns the final result of the association rules algorithm
     * \param[in] id   Identifier of the result
     * \return         Final result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the final result of the association rules algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks the result of the association rules algorithm
     * \param[in] input   %Input of algorithm
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method of the algorithm
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
               int method) const DAAL_C11_OVERRIDE
    {
        Parameter *algParameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));

        services::SharedPtr<data_management::NumericTable> largeItemsetsNT = get(largeItemsets);
        if(largeItemsetsNT.get() == 0) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        if(largeItemsetsNT->getNumberOfColumns() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }

        services::SharedPtr<data_management::NumericTable> largeItemsetsSupportNT = get(largeItemsetsSupport);
        if(largeItemsetsSupportNT.get() == 0) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        if(largeItemsetsSupportNT->getNumberOfColumns() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }

        if(algParameter->discoverRules)
        {
            if(Argument::size() != 5) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

            services::SharedPtr<data_management::NumericTable> leftItemsNT = get(antecedentItemsets);
            if(leftItemsNT.get() == 0) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
            if(leftItemsNT->getNumberOfColumns() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }

            services::SharedPtr<data_management::NumericTable> rightItemsNT = get(consequentItemsets);
            if(rightItemsNT.get() == 0) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
            if(rightItemsNT->getNumberOfColumns() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }

            services::SharedPtr<data_management::NumericTable> confidenceNT = get(confidence);
            if(confidenceNT.get() == 0) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
            if(confidenceNT->getNumberOfColumns() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }
        }
        else
        {
            if(Argument::size() != 5) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }
        }
        return;
    }

    int getSerializationTag() DAAL_C11_OVERRIDE { return SERIALIZATION_ASSOCIATION_RULES_RESULT_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for the serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes the object
    *  \param[in]  arch  Storage for the deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
} // namespace interface1
using interface1::Parameter;
using interface1::Input;
using interface1::Result;

} // namespace association_rules
} // namespace algorithm
} // namespace daal
#endif
