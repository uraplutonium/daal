/* file: softmax_types.h */
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
//  Implementation of the softmax function interface.
//--
*/

#ifndef __SOFTMAX_TYPES_H__
#define __SOFTMAX_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for computing math functions
 */
namespace math
{
/**
 * \brief Contains classes for computing the softmax function
 */
namespace softmax
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__SOFTMAX__METHOD"></a>
 * Computation methods for computing the softmax function
 */
enum Method
{
    defaultDense = 0       /*!< Default: performance-oriented method. */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__SOFTMAX__INPUTID"></a>
 * Available identifiers of input objects for the softmax function
 */
enum InputId
{
    data = 0               /*!< %Input data table */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__SOFTMAX__RESULTID"></a>
 * Available identifiers of the result of the softmax function
 */
enum ResultId
{
    value = 0     /*!< Table to store the result. */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SOFTMAX__INPUT"></a>
 * \brief %Input objects for the softmax function
 */
class Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    Input() : daal::algorithms::Input(1) {};

    virtual ~Input() {}

    /**
     * Returns an input object for the softmax function
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets an input object for the softmax function
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks an input object for the softmax function
     * \param[in] par     Function parameter
     * \param[in] method  Computation method
     */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        services::SharedPtr<data_management::NumericTable> inTable = get(data);

        if(!inTable) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        if(inTable->getNumberOfRows() == 0)    { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }
        if(inTable->getNumberOfColumns() == 0) { this->_errors->add(services::ErrorIncorrectNumberOfColumnsInInputNumericTable); return; }
        if(inTable->getDataMemoryStatus() == data_management::NumericTable::notAllocated)
        {
            this->_errors->add(services::ErrorNullInputNumericTable); return;
        }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SOFTMAX__RESULT"></a>
 * \brief Results obtained with the compute() method of the softmax function in the batch processing mode
 */
class Result : public daal::algorithms::Result
{
public:
    /** Default constructor */
    Result() : daal::algorithms::Result(1) {}

    virtual ~Result() {};

    /**
     * Allocates memory to store the results of the softmax function
     * \param[in] input  %Input object for the the softmax function
     * \param[in] par    %Parameter of the softmax function
     * \param[in] method Computation method of the softmax function
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
    {
        Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));

        if(algInput == 0) { this->_errors->add(services::ErrorNullInput); return; }
        if(algInput->get(data) == 0) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        if(algInput->get(data).get() == 0) { this->_errors->add(services::ErrorNullInputNumericTable); return; }

        size_t nFeatures     = algInput->get(data)->getNumberOfColumns();
        size_t nObservations = algInput->get(data)->getNumberOfRows();
        Argument::set(value, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(nFeatures, nObservations,
                                                                                    data_management::NumericTable::doAllocate)));
    }

    /**
     * Returns the result of the softmax function
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the result of the softmax function
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Result
     */
    void set(ResultId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks the result of the softmax function
     * \param[in] in   %Input of the softmax function
     * \param[in] par     %Parameter of the softmax function
     * \param[in] method  Computation method of the softmax function
     */
    void check(const daal::algorithms::Input *in, const daal::algorithms::Parameter *par,
               int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        if(in == 0) { this->_errors->add(services::ErrorNullInput); return; }

        services::SharedPtr<data_management::NumericTable> dataTable = (static_cast<const Input *>(in))->get(data);
        services::SharedPtr<data_management::NumericTable> resultTable = get(value);

        if(dataTable.get() == 0)                   { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        if(resultTable.get() == 0)                 { this->_errors->add(services::ErrorNullOutputNumericTable); return; }

        if(resultTable->getDataMemoryStatus() == data_management::NumericTable::notAllocated)
        {
            this->_errors->add(services::ErrorNullOutputNumericTable); return;
        }

        size_t nDataRows = dataTable->getNumberOfRows();
        size_t nDataColumns = dataTable->getNumberOfColumns();

        size_t nResultRows = resultTable->getNumberOfRows();
        size_t nResultColumns = resultTable->getNumberOfColumns();

        if(nDataRows != nResultRows) { this->_errors->add(services::ErrorIncorrectNumberOfRowsInOutputNumericTable); return; }
        if(nDataColumns != nResultColumns) { this->_errors->add(services::ErrorIncorrectNumberOfColumnsInOutputNumericTable); return; }
}

    /**
    * Returns the serialization tag of the result
    * \return         Serialization tag of the result
    */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_SOFTMAX_RESULT_ID; }

    /**
    *  Serializes an object
    *  \param[in]  arch  Storage for a serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes an object
    *  \param[in]  arch  Storage for a deserialized object or data structure
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

} // namespace softmax
} // namespace math
} // namespace algorithm
} // namespace daal
#endif
