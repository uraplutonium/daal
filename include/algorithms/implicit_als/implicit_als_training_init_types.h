/* file: implicit_als_training_init_types.h */
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
//  Implementation of the implicit ALS algorithm interface
//--
*/

#ifndef __IMPLICIT_ALS_TRAINING_INIT_TYPES_H__
#define __IMPLICIT_ALS_TRAINING_INIT_TYPES_H__

#include "algorithms/implicit_als/implicit_als_model.h"
#include "algorithms/implicit_als/implicit_als_training_types.h"
#include "data_management/data/csr_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
/**
 * \brief Contains classes for the implicit ALS initialization algorithm
 */
namespace init
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__METHOD"></a>
 * \brief Available methods for initializing the implicit ALS algorithm
 */
enum Method
{
    defaultDense = 0,       /*!< Default: initialization method for input data stored in the dense format */
    fastCSR = 1             /*!< Initialization method for input data stored in the compressed sparse row (CSR) format */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INPUTID"></a>
 * \brief Available identifiers of input objects for the implicit ALS initialization algorithm
 */
enum InputId
{
    data = 0,               /*!< %Input data table that contains ratings */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__PARTIALRESULTID"></a>
 * \brief Available identifiers of partial results of the implicit ALS initialization algorithm
 */
enum PartialResultId
{
    partialModel = 0               /*!< Partial implicit ALS model */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__RESULT_ID"></a>
 * \brief Available identifiers of the results of the implicit ALS initialization algorithm
 */
enum ResultId
{
    model = 0               /*!< Implicit ALS model */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface
 */
namespace interface1
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__PARAMETER"></a>
  * \brief Parameters of the compute() method of the implicit ALS initialization algorithm
 *
 * \snippet implicit_als/implicit_als_training_init_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    Parameter(size_t nFactors = 10, size_t fullNUsers = 0, size_t seed = 777777) :
        nFactors(nFactors), fullNUsers(fullNUsers), seed(seed)
    {}

    size_t nFactors;            /*!< Total number of factors */
    size_t fullNUsers;          /*!< Full number of users */
    size_t seed;                /*!< Seed for generating random numbers in the initialization step */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INPUT"></a>
 * \brief %Input objects for the implicit ALS initialization algorithm
 */
class Input : public daal::algorithms::Input
{
public:
    Input() : daal::algorithms::Input(1) {}

    virtual ~Input() {}

    /**
     * Returns an input object for the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable,
               data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets an input object for the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Returns the number of items, that is, the number of columns in the input data set
     * \return Number of items
     */
    size_t getNumberOfItems() const { return get(data)->getNumberOfRows(); }

    /**
     * Checks the input objects and parameters of the implicit ALS initialization algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        services::SharedPtr<data_management::NumericTable> dataTable = get(data);
        if(!dataTable) { this->_errors->add(services::ErrorNullInputNumericTable); return; }

        if (method == fastCSR)
        {
            if (dynamic_cast<data_management::CSRNumericTableIface *>(dataTable.get()) == NULL)
            { this->_errors->add(services::ErrorIncorrectTypeOfInputNumericTable); return; }
        }

        if(dataTable->getNumberOfColumns() < 1) { this->_errors->add(services::ErrorEmptyInputNumericTable); return; }
        if(dataTable->getNumberOfRows() < 1) { this->_errors->add(services::ErrorEmptyInputNumericTable); return; }

        if(!parameter) { this->_errors->add(services::ErrorNullParameterNotSupported); return; }
        const Parameter *alsParameter = static_cast<const Parameter *>(parameter);
        size_t nFactors = alsParameter->nFactors;
        if(nFactors < 1) { this->_errors->add(services::ErrorIncorrectParameter); return; }
    }
};

/**
 * <a name="DAAL-CLASS-IMPLICIT_ALS__TRAINING__INIT__PARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 *        of the implicit ALS initialization algorithm
 */
class PartialResult : public daal::algorithms::PartialResult
{
public:
    PartialResult() : daal::algorithms::PartialResult(1) {}

    /**
     * Allocates memory for storing partial results of the implicit ALS initialization algorithm
     * \param[in] input         Pointer to the input object structure
     * \param[in] parameter     Pointer to the parameter structure
     * \param[in] method        Computation method of the algorithm
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const Input *algInput = static_cast<const Input *>(input);
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);
        implicit_als::Parameter modelParameter(algParameter->nFactors);

        Argument::set(partialModel, services::SharedPtr<PartialModel>(new PartialModel(
                          modelParameter, algInput->getNumberOfItems(), (algorithmFPType)0.0)));
    }

    /**
     * Returns a partial result of the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Partial result that corresponds to the given identifier
     */
    services::SharedPtr<PartialModel> get(PartialResultId id) const
    {
        return services::staticPointerCast<PartialModel, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets a partial result of the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the partial result
     */
    void set(PartialResultId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks a partial result of the implicit ALS initialization algorithm
     * \param[in] input       %Input object for the algorithm
     * \param[in] parameter   %Parameter of the algorithm
     * \param[in] method      Computation method of the algorithm
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        if (method != fastCSR) { this->_errors->add(services::ErrorMethodNotSupported); return; }
        if (size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfElementsInInputCollection); return; }

        const Parameter *algParameter = static_cast<const Parameter *>(parameter);
        size_t nFactors = algParameter->nFactors;
        if (nFactors < 1) { this->_errors->add(services::ErrorIncorrectParameter); return; }

        const Input *algInput = static_cast<const Input *>(input);
        services::SharedPtr<data_management::NumericTable> dataTable = algInput->get(data);
        size_t nRows = dataTable->getNumberOfRows();

        services::SharedPtr<PartialModel> model = get(partialModel);
        if (!model) { this->_errors->add(services::ErrorNullPartialModel); return; }

        services::SharedPtr<data_management::NumericTable> factors = model->getFactors();
        services::SharedPtr<data_management::NumericTable> indices = model->getIndices();

        if (!factors || !indices) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        if (factors->getNumberOfColumns() != nFactors) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }
        if (indices->getNumberOfColumns() != 1)        { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }
        if (factors->getNumberOfRows() != nRows)
        { this->_errors->add(services::ErrorInconsistentNumberOfRows); return; }
        if (factors->getNumberOfRows() != indices->getNumberOfRows())
        { this->_errors->add(services::ErrorInconsistentNumberOfRows); return; }
    }

     /**
     * Returns a serialization tag, a unique identifier of this class used in serialization
     * \return Serialization tag
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_IMPLICIT_ALS_TRAINING_INIT_PARTIAL_RESULT_ID; }
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
        daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
 * <a name="DAAL-CLASS-IMPLICIT_ALS__TRAINING__INIT__RESULT"></a>
 * \brief Provides methods to access the results obtained with the compute() method
 *        of the implicit ALS initialization algorithm
 */
class Result : public daal::algorithms::implicit_als::training::Result
{
public:

    /**
     * Returns a serialization tag, a unique identifier of this class used in serialization
     * \return Serialization tag
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_IMPLICIT_ALS_TRAINING_INIT_RESULT_ID; }

    /**
     * Returns the result of the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    services::SharedPtr<daal::algorithms::implicit_als::Model> get(ResultId id) const
    {
        return services::staticPointerCast<daal::algorithms::implicit_als::Model,
               data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the result of the implicit ALS initialization algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const services::SharedPtr<daal::algorithms::implicit_als::Model> &ptr)
    {
        Argument::set(id, ptr);
    }

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
        training::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
} // namespace interface1
using interface1::Parameter;
using interface1::Input;
using interface1::PartialResult;
using interface1::Result;

}
}
}
}
}

#endif
