/* file: kernel_function_types.h */
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
//  Kernel function parameter structure
//--
*/

#ifndef __KERNEL_FUNCTION_TYPES_H__
#define __KERNEL_FUNCTION_TYPES_H__

#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for computing kernel functions
 */
namespace kernel_function
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__KERNEL_FUNCTION__COMPUTATIONMODE"></a>
 * Mode of computing kernel functions
 */
enum ComputationMode
{
    vectorVector = 0,     /*!< Computes the kernel function for given feature vectors Xi and Yj */
    matrixVector = 1,     /*!< Computes the kernel function for all the vectors in the set X and a given feature vector Yi */
    matrixMatrix = 2,     /*!< Computes the kernel function for all the vectors in the sets X and Y */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KERNEL_FUNCTION___FUNCTION__INPUTID"></a>
 * Available identifiers of input objects of the kernel function algorithm
 */
enum InputId
{
    X = 0,     /*!< %Input left data table */
    Y = 1      /*!< %Input right data table */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KERNEL_FUNCTION___FUNCTION__RESULTID"></a>
 * Available identifiers of results of the kernel function algorithm
 */
enum ResultId
{
    values = 0         /*!< Table to store results */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__PARAMETERBASE"></a>
 * \brief Optional %input objects for the kernel function algorithm
 *
 * \snippet kernel_function/kernel_function_types.h ParameterBase source code
 */
/* [ParameterBase source code] */
struct DAAL_EXPORT ParameterBase : public daal::algorithms::Parameter
{
    ParameterBase(size_t rowIndexX = 0, size_t rowIndexY = 0, size_t rowIndexResult = 0, ComputationMode computationMode = matrixMatrix) :
        rowIndexX(rowIndexX), rowIndexY(rowIndexY), rowIndexResult(rowIndexResult), computationMode(computationMode) {}
    size_t rowIndexX;          /*!< Index of the vector in the set X */
    size_t rowIndexY;          /*!< Index of the vector in the set Y */
    size_t rowIndexResult;     /*!< Index of the result of the kernel function computation */
    ComputationMode computationMode;    /*!< Mode of computing kernel functions */
};
/* [ParameterBase source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__INPUT"></a>
 * \brief %Input objects for the kernel function algorithm
 */
class Input : public daal::algorithms::Input
{
public:
    Input() : daal::algorithms::Input(2) {}

    virtual ~Input() {}

    /**
    * Returns the input object of the kernel function algorithm
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
    * Sets the input object of the kernel function algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the input object
    */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
    * Checks input objects of the kernel function algorithm
    * \param[in] par     %Input objects of the algorithm
    * \param[in] type    Type of the algorithm
    */
    void check(const daal::algorithms::Parameter *par, int type) const DAAL_C11_OVERRIDE
    {
        ParameterBase *algParameter = static_cast<ParameterBase *>(const_cast<daal::algorithms::Parameter *>(par));

        services::SharedPtr<data_management::NumericTable> in1 = get(X);
        if(in1.get() == 0)                 { this->_errors->add(services::ErrorNullInputNumericTable);         return; }
        if(in1->getNumberOfRows() == 0)    { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(in1->getNumberOfColumns() == 0) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures);     return; }

        services::SharedPtr<data_management::NumericTable> in2 = get(Y);
        if(in2.get() == 0)                 { this->_errors->add(services::ErrorNullInputNumericTable);         return; }
        if(in2->getNumberOfRows() == 0)    { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(in2->getNumberOfColumns() == 0) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures);     return; }

        size_t nVectors1 = in1->getNumberOfRows();
        size_t nVectors2 = in2->getNumberOfRows();

        size_t nFeatures1 = in1->getNumberOfColumns();
        size_t nFeatures2 = in2->getNumberOfColumns();

        if(nFeatures1 != nFeatures2) { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__RESULT"></a>
 * \brief Results obtained with the compute() method of the kernel function algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    Result();

    virtual ~Result() {};
    /**
     * Allocates memory to store results of the kernel function algorithm
     * \param[in] input  Pointer to the structure with the input objects
     * \param[in] par    Pointer to the structure of the algorithm parameters
     * \param[in] method       Computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
    {
        const Input *algInput = static_cast<const Input *>(input);
        size_t nVectors1 = algInput->get(X)->getNumberOfRows();
        size_t nVectors2 = algInput->get(Y)->getNumberOfRows();
        Argument::set(values, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(nVectors2, nVectors1,
                                                                                    data_management::NumericTable::doAllocate)));
    }

    /**
     * Returns the result of the kernel function algorithm
     * \param[in] id   Identifier of the result
     * \return         Final result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the result of the kernel function algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the object
     */
    void set(ResultId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
    * Checks the result of the kernel function algorithm
    * \param[in] input   %Input objects of the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] type    Type of the algorithm
    */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
               int type) const DAAL_C11_OVERRIDE
    {
        Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
        ParameterBase *algParameter = static_cast<ParameterBase *>(const_cast<daal::algorithms::Parameter *>(par));

        size_t nVectors1 = algInput->get(X)->getNumberOfRows();
        size_t nVectors2 = algInput->get(Y)->getNumberOfRows();

        services::SharedPtr<data_management::NumericTable> resTable = get(values);

        if(resTable.get() == 0)                 { this->_errors->add(services::ErrorNullOutputNumericTable);        return; }
        if(resTable->getNumberOfRows() == 0)    { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(resTable->getNumberOfColumns() == 0) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures);     return; }
    }

    int getSerializationTag() { return SERIALIZATION_KERNEL_FUNCTION_RESULT_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for a serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes the object
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
using interface1::ParameterBase;
using interface1::Input;
using interface1::Result;

} // namespace kernel_function
} // namespace algorithms
} // namespace daal
#endif
