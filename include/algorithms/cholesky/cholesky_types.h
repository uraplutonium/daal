/* file: cholesky_types.h */
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
//  Implementation of Cholesky algorithm interface.
//--
*/

#ifndef __CHOLESKY_TYPES_H__
#define __CHOLESKY_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for computing Cholesky decomposition
 */
namespace cholesky
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__CHOLESKY__METHOD"></a>
 * Available methods for computing Cholesky decomposition
 */
enum Method
{
    defaultDense = 0       /*!< Default: performance-oriented method. */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__CHOLESKY__INPUTID"></a>
 * Available identifiers of input objects for the Cholesky algorithm
 */
enum InputId
{
    data = 0               /*!< %Input data table */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__CHOLESKY__RESULTID"></a>
 * Available identifiers of results for the Cholesky algorithm
 */
enum ResultId
{
    choleskyFactor = 0     /*!< Table to store the result. Contains the lower triangle matrix L of the decomposition */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__CHOLESKY__INPUT"></a>
 * \brief %Input parameters for the Cholesky algorithm
 */
class Input : public daal::algorithms::Input
{
public:
    Input() : daal::algorithms::Input(1) {}

    virtual ~Input() {}

    /**
     * Returns input NumericTable of the Cholesky algorithm
     * \param[in] id    Identifier of the input numeric table
     * \return          %Input numeric table that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets input for the Cholesky algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks parameters of the Cholesky algorithm
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method of the algorithm
     */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        services::SharedPtr<data_management::NumericTable> inTable = get(data);

        if(inTable.get() == 0)                 { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        if(inTable->getNumberOfRows() == 0)    { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(inTable->getNumberOfColumns() == 0) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }

        data_management::NumericTableIface::StorageLayout iLayout = inTable->getDataLayout();

        if(inTable->getNumberOfColumns() != inTable->getNumberOfRows())
        { this->_errors->add(services::ErrorIncorrectSizeOfInputNumericTable); return; }

        int iLayoutInt = (int) iLayout;
        if(iLayoutInt & data_management::packed_mask)
        {
            if(iLayout == data_management::NumericTableIface::lowerPackedTriangularMatrix ||
               iLayout == data_management::NumericTableIface::upperPackedTriangularMatrix)
            { this->_errors->add(services::ErrorIncorrectTypeOfInputNumericTable); return; }
        }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CHOLESKY__RESULT"></a>
 * \brief Results obtained with the compute() method of the Cholesky algorithm in the batch processing mode
 */
class Result : public daal::algorithms::Result
{
public:
    Result() : daal::algorithms::Result(1) {}

    virtual ~Result() {};

    /**
     * Allocates memory to store the results of Cholesky decomposition
     * \param[in] input  Pointer to the input structure
     * \param[in] par    Pointer to the parameter structure
     * \param[in] method Computation method of the algorithm
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
    {
        Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
        size_t nFeatures   = algInput->get(data)->getNumberOfColumns();
        Argument::set(choleskyFactor, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(nFeatures, nFeatures,
                                                                                    data_management::NumericTable::doAllocate)));
    }

    /**
     * Returns result of the Cholesky algorithm
     * \param[in] id   Identifier of the result
     * \return         Final result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the result of the Cholesky algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks the result of the Cholesky algorithm
     * \param[in] input   %Input of algorithm
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method of the algorithm
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
               int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        services::SharedPtr<data_management::NumericTable> resTable = get(choleskyFactor);

        if(resTable.get() == 0)                 { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        if(resTable->getNumberOfRows() == 0)    { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(resTable->getNumberOfColumns() == 0) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }

        data_management::NumericTableIface::StorageLayout rLayout = resTable->getDataLayout();

        Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));

        if(resTable->getNumberOfColumns() != algInput->get(data)->getNumberOfColumns() ||
           resTable->getNumberOfColumns() != resTable->getNumberOfRows())
        { this->_errors->add(services::ErrorIncorrectSizeOfOutputNumericTable); return; }

        int rLayoutInt = (int) rLayout;
        if(rLayoutInt & data_management::packed_mask)
        {
            if(rLayout != data_management::NumericTableIface::lowerPackedTriangularMatrix)
            { this->_errors->add(services::ErrorIncorrectTypeOfOutputNumericTable); return; }
        }
    }

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_CHOLESKY_RESULT_ID; }

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

} // namespace cholesky
} // namespace algorithm
} // namespace daal
#endif
