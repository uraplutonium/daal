/* file: sorting_types.h */
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
//  Definition of common types of sorting.
//--
*/

#ifndef __SORTING_TYPES_H__
#define __SORTING_TYPES_H__

#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes to run the sorting algorithms
 */
namespace sorting
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__SORTING__METHOD"></a>
 * Available methods for sorting computation
 */
enum Method
{
    defaultDense = 0      /*!< Default: radix method for sorting a data set */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__SORTING__INPUTID"></a>
 * Available identifiers of input objects for the sorting algorithm
 */
enum InputId
{
    data = 0            /*!< %Input data table */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__SORTING__RESULTID"></a>
 * Available identifiers of results of the sorting algorithm
 */
enum ResultId
{
    sortedData = 0       /*!< observation sorting results */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SORTING__INPUT"></a>
 * \brief %Input objects for the sorting algorithm
 */
class Input : public daal::algorithms::Input
{
public:
    Input() : daal::algorithms::Input(1)
    {}

    virtual ~Input() {}

    /**
     * Returns an input object for the sorting algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the input object of the sorting algorithm
     * \param[in] id    Identifier of the %input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Check the correctness of the %Input object
     * \param[in] method    Algorithm computation method
     * \param[in] par       Pointer to the parameters of the algorithm
     */
    void check(const Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        services::SharedPtr<data_management::NumericTable> inTable = get(data);
        if(!inTable)
        {
            this->_errors->add(services::ErrorNullInputNumericTable);
            return;
        }
        size_t nFeatures = inTable->getNumberOfColumns();
        if(nFeatures == 0) { this->_errors->add(services::ErrorIncorrectSizeOfInputNumericTable); return; }
        if(inTable->getNumberOfRows() == 0)
        {
            this->_errors->add(services::ErrorIncorrectSizeOfInputNumericTable);
            return;
        }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SORTING__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the
 *        sorting algorithm in the batch processing mode
 */
class Result : public daal::algorithms::Result
{
public:
    Result() : daal::algorithms::Result(1)
    {}

    virtual ~Result() {};

    /**
     * Allocates memory to store final results of the sorting algorithms
     * \param[in] input     Input objects for the sorting algorithm
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const int method)
    {
        const Input *in = static_cast<const Input *>(input);

        size_t nFeatures = in->get(data)->getNumberOfColumns();
        size_t nVectors = in->get(data)->getNumberOfRows();

        Argument::set(sortedData, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(nFeatures, nVectors,
                                                                                    data_management::NumericTable::doAllocate)));
    }

    /**
     * Returns the final result of the sorting algorithm
     * \param[in] id   Identifier of the final result, \ref ResultId
     * \return         Final result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the Result object of the sorting algorithm
     * \param[in] id        Identifier of the Result object
     * \param[in] value     Pointer to the Result object
     */
    void set(ResultId id, const services::SharedPtr<data_management::NumericTable> &value)
    {
        Argument::set(id, value);
    }

    /**
     * Checks the correctness of the Result object
     * \param[in] in     Pointer to the object
     * \param[in] par     %Parameter of algorithm
     * \param[in] method Algorithm computation method
     */
    void check(const daal::algorithms::Input *in, const Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        const Input *input = static_cast<const Input *>(in);

        size_t nFeatures = input->get(data)->getNumberOfColumns();
        size_t nVectors  = input->get(data)->getNumberOfRows();

        services::SharedPtr<data_management::NumericTable> outputTable = get(sortedData);

        if(outputTable.get() == 0)                         { this->_errors->add(services::ErrorNullOutputNumericTable); return; }
        if(outputTable->getNumberOfRows() != nVectors)     { this->_errors->add(services::ErrorIncorrectSizeOfOutputNumericTable); return; }
        if(outputTable->getNumberOfColumns() != nFeatures) { this->_errors->add(services::ErrorIncorrectSizeOfOutputNumericTable); return; }
    }

    int getSerializationTag() { return SERIALIZATION_SORTING_RESULT_ID; }

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
using interface1::Input;
using interface1::Result;

} // namespace daal::algorithms::sorting
} // namespace daal::algorithms
} // namespace daal
#endif
