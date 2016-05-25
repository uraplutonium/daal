/* file: linear_regression_training_types.h */
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
//  Implementation of the linear regression algorithm interface
//--
*/

#ifndef __LINEAR_REGRESSION_TRAINING_TYPES_H__
#define __LINEAR_REGRESSION_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/data_serialize.h"
#include "services/daal_defines.h"
#include "algorithms/linear_regression/linear_regression_model.h"
#include "algorithms/linear_regression/linear_regression_ne_model.h"
#include "algorithms/linear_regression/linear_regression_qr_model.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes of the linear regression algorithm
 */
namespace linear_regression
{
/**
 * \brief Contains a class for linear regression model-based training
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__LINEAR_REGRESSION__TRAINING__METHOD"></a>
 * \brief Computation methods for linear regression model-based training
 */
enum Method
{
    defaultDense = 0,  /*!< Normal equations method */
    normEqDense = 0,  /*!< Normal equations method */
    qrDense = 1 /*!< QR decomposition-based method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__LINEAR_REGRESSION__TRAINING__INPUT_ID"></a>
 * \brief Available identifiers of input objects for linear regression model-based training
 */
enum InputId
{
    data = 0,               /*!< %Input data table */
    dependentVariables = 1  /*!< Values of the dependent variable for the input data */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__LINEAR_REGRESSION__TRAINING__MASTER_INPUT_ID"></a>
 * \brief Available identifiers of input objects for linear regression model-based training
 * in the second step of the distributed processing mode
 */
enum Step2MasterInputId
{
    partialModels = 0   /*!< Collection of partial models trained on local nodes */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__LINEAR_REGRESSION__TRAINING__PARTIAL_RESULT_ID"></a>
 * \brief Available identifiers of a partial result of linear regression model-based training
 */
enum PartialResultID
{
    partialModel = 0   /*!< Partial model trained on the available input data */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__LINEAR_REGRESSION__TRAINING__RESULT_ID"></a>
 * \brief Available identifiers of the result of linear regression model-based training
 */
enum ResultId
{
    model = 0   /*!< Linear regression model */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__TRAINING__INPUT_IFACE"></a>
 * \brief Abstract class that specifies the interface of input objects for linear regression model-based training
 */
class InputIface : public daal::algorithms::Input
{
public:
    /** Default constructor */
    InputIface(size_t nElements) : daal::algorithms::Input(nElements) {};

    /**
     * Returns the number of columns in the input data set
     * \return Number of columns in the input data set
     */
    virtual size_t getNFeatures() const = 0;

    /**
     * Returns the number of dependent variables
     * \return Number of dependent variables
     */
    virtual size_t getNDependentVariables() const = 0;

    virtual ~InputIface() {};
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__TRAINING__INPUT"></a>
 * \brief %Input objects for linear regression model-based training
 */
class Input : public InputIface
{
public:
    /** Default constructor */
    Input() : InputIface(2) {};

    virtual ~Input() {};

    /**
     * Returns an input object for linear regression model-based training
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets an input object for linear regression model-based training
     * \param[in] id      Identifier of the input object
     * \param[in] value   Pointer to the object
     */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &value)
    {
        Argument::set(id, value);
    }

    /**
     * Returns the number of columns in the input data set
     * \return Number of columns in the input data set
     */
    size_t getNFeatures() const DAAL_C11_OVERRIDE { return get(data)->getNumberOfColumns(); }

    /**
    * Returns the number of dependent variables
    * \return Number of dependent variables
    */
    size_t getNDependentVariables() const DAAL_C11_OVERRIDE { return get(dependentVariables)->getNumberOfColumns(); }

    /**
    * Checks an input object for the linear regression algorithm
    * \param[in] par     Algorithm parameter
    * \param[in] method  Computation method
    */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        services::SharedPtr<data_management::NumericTable> dataTable = get(data);
        services::SharedPtr<data_management::NumericTable> dependentVariableTable = get(dependentVariables);

        if(!dataTable) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        size_t nRowsInData = dataTable->getNumberOfRows();
        size_t nColumnsInData = dataTable->getNumberOfColumns();
        if(nRowsInData == 0)    { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(nColumnsInData == 0) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;     }
        if(nRowsInData < nColumnsInData)
        { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }

        if(!dependentVariableTable) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        size_t nRowsInDepVariable = dependentVariableTable->getNumberOfRows();
        size_t nColumnsInDepVariable = dependentVariableTable->getNumberOfColumns();
        if(nRowsInDepVariable == 0)    { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(nColumnsInDepVariable == 0) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;     }

        if(nRowsInData != nRowsInDepVariable)
        { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__TRAINING__DISTRIBUTED_INPUT"></a>
 * \brief %Input object for linear regression model-based training in the distributed processing mode
 */
template<ComputeStep step>
class DistributedInput
{};

/**
 * <a name="DAAL-CLASS-LINEAR_REGRESSION__TRAINING__PARTIALRESULT"></a>
 * \brief Provides methods to access a partial result obtained with the compute() method of
 *        linear regression model-based training in the online or distributed processing mode
 */
class PartialResult : public daal::algorithms::PartialResult
{
public:
    PartialResult() : daal::algorithms::PartialResult(1) {};

    /**
    * Returns a partial result of linear regression model-based training
    * \param[in] id    Identifier of the partial result
    * \return          Partial result that corresponds to the given identifier
    */
    services::SharedPtr<daal::algorithms::linear_regression::Model> get(PartialResultID id) const
    {
        return services::staticPointerCast<daal::algorithms::linear_regression::Model,
               data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Returns the number of columns in the input data set
     * \return Number of columns in the input data set
     */
    size_t getNFeatures() const { return get(partialModel)->getNumberOfFeatures(); }

    /**
    * Returns the number of dependent variables
    * \return Number of dependent variables
    */
    size_t getNDependentVariables() const { return get(partialModel)->getNumberOfResponses(); }

    /**
     * Sets an argument of the partial result
     * \param[in] id      Identifier of the argument
     * \param[in] value   Pointer to the argument
     */
    void set(PartialResultID id, const services::SharedPtr<daal::algorithms::linear_regression::Model> &value)
    {
        Argument::set(id, value);
    }

    /**
     * Allocates memory to store a partial result of linear regression model-based training
     * \param[in] input %Input object for the algorithm
     * \param[in] method Method of linear regression model-based training
     * \param[in] parameter %Parameter of linear regression model-based training
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        if(method == qrDense)
        {
            algorithmFPType dummy = 1.0;
            set(partialModel, services::SharedPtr<daal::algorithms::linear_regression::Model>(
                    new ModelQR((static_cast<const InputIface *>(input))->getNFeatures(),
                                (static_cast<const InputIface *>(input))->getNDependentVariables(),
                                *(static_cast<const Parameter *>(parameter)), dummy)));
        }
        else if(method == normEqDense)
        {
            algorithmFPType dummy = 1.0;
            set(partialModel, services::SharedPtr<daal::algorithms::linear_regression::Model>(
                    new ModelNormEq((static_cast<const InputIface *>(input))->getNFeatures(),
                                    (static_cast<const InputIface *>(input))->getNDependentVariables(),
                                    *(static_cast<const Parameter *>(parameter)), dummy)));
        }
    }

    /**
     * Checks a partial result of the linear regression algorithm
     * \param[in] input   %Input object for the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
               int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        services::SharedPtr<daal::algorithms::linear_regression::Model> model = get(training::partialModel);
        if(model->getBeta() == 0) { this->_errors->add(services::ErrorNullOutputNumericTable); return; }
    }

    /**
     * Checks a partial result of the linear regression algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        services::SharedPtr<daal::algorithms::linear_regression::Model> model = get(training::partialModel);
        if(model->getBeta() == 0) { this->_errors->add(services::ErrorNullOutputNumericTable); return; }
    }

    /**
     * Returns the serialization tag of the partial result
     * \return         Serialization tag of the partial result
     */

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_LINEAR_REGRESSION_PARTIAL_RESULT_ID; }
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
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__TRAINING__DISTRIBUTED_INPUT"></a>
 * \brief %Input object for linear regression model-based training in the second step of the
 *  distributed processing mode
 */
template<> class DistributedInput<step2Master> : public InputIface
{
public:
    DistributedInput<step2Master>() : InputIface(1)
    {
        Argument::set(partialModels, services::SharedPtr<data_management::DataCollection>(new data_management::DataCollection()));
    };

    /**
     * Gets an input object for linear regression model-based training
     * in the second step of the distributed processing mode
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::DataCollection> get(Step2MasterInputId id) const
    {
        return services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets an input object for linear regression model-based training
     * in the second step of the distributed processing mode
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   %Input object
     */
    void set(Step2MasterInputId id, const services::SharedPtr<data_management::DataCollection> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     Adds an input object for linear regression model-based training in the second step
     * of the distributed processing mode
     * \param[in] id      Identifier of the input object
     * \param[in] partialResult   %Input object
     */
    void add(Step2MasterInputId id, const services::SharedPtr<PartialResult> &partialResult)
    {
        services::SharedPtr<data_management::DataCollection> collection =
            services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
        collection->push_back(services::staticPointerCast<data_management::SerializationIface, linear_regression::Model>(
                                  partialResult->get(training::partialModel)));
    }

    /**
     * Returns the number of columns in the input data set
     * \return Number of columns in the input data set
     */
    size_t getNFeatures() const DAAL_C11_OVERRIDE
    {
        services::SharedPtr<data_management::DataCollection> partialModelsCollection =
            static_cast<services::SharedPtr<data_management::DataCollection> >(get(partialModels));

        if(partialModelsCollection->size() == 0) { return 0; }

        linear_regression::Model *partialModel
            = static_cast<daal::algorithms::linear_regression::Model *>(((*partialModelsCollection)[0]).get());

        return partialModel->getNumberOfFeatures();
    }

    /**
     * Returns the number of dependent variables
     * \return Number of dependent variables
     */
    size_t getNDependentVariables() const DAAL_C11_OVERRIDE
    {
        services::SharedPtr<data_management::DataCollection> partialModelsCollection =
            static_cast<services::SharedPtr<data_management::DataCollection> >(get(partialModels));

        if(partialModelsCollection->size() == 0) { return 0; }

        linear_regression::Model *partialModel
            = static_cast<daal::algorithms::linear_regression::Model *>(((*partialModelsCollection)[0]).get());

        return partialModel->getNumberOfResponses();
    }

    /**
     * Checks an input object for linear regression model-based training in the second step
     * of the distributed processing mode
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        size_t nBlocks = Argument::size();
        if(nBlocks == 0) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        for(size_t j = 0; j < nBlocks; j++)
        {
            services::SharedPtr<linear_regression::Model> model =
                services::staticPointerCast<linear_regression::Model, data_management::SerializationIface>((*(get(training::partialModels)))[j]);
            if(model->getBeta() == 0) { this->_errors->add(services::ErrorNullOutputNumericTable); return; }
        }
    }
};

/**
 * <a name="DAAL-CLASS-LINEAR_REGRESSION__TRAINING__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of linear regression model-based training
 */
class Result : public daal::algorithms::Result
{
public:
    Result() : daal::algorithms::Result(1) {};

    /**
     * Returns the result of linear regression model-based training
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    services::SharedPtr<daal::algorithms::linear_regression::Model> get(ResultId id) const
    {
        return services::staticPointerCast<daal::algorithms::linear_regression::Model,
               data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the result of linear regression model-based training
     * \param[in] id      Identifier of the result
     * \param[in] value   Result
     */
    void set(ResultId id, const services::SharedPtr<daal::algorithms::linear_regression::Model> &value)
    {
        Argument::set(id, value);
    }

    /**
     * Allocates memory to store the result of linear regression model-based training
     * \param[in] input Pointer to an object containing the input data
     * \param[in] method Computation method for the algorithm
     * \param[in] parameter %Parameter of linear regression model-based training
     */
    template<typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const Parameter *parameter, const int method)
    {
        const Input *in = static_cast<const Input *>(input);

        if(method == qrDense)
        {
            algorithmFPType dummy = 1.0;
            set(model, services::SharedPtr<daal::algorithms::linear_regression::Model>(new ModelQR(in->getNFeatures(),
                                                                                                   in->getNDependentVariables(),
                                                                                                   *parameter, dummy)));
        }
        else if(method == normEqDense)
        {
            algorithmFPType dummy = 1.0;
            set(model, services::SharedPtr<daal::algorithms::linear_regression::Model>(new ModelNormEq(in->getNFeatures(),
                                                                                                       in->getNDependentVariables(),
                                                                                                       *parameter, dummy)));
        }
    }

    /**
     * Allocates memory to store the result of linear regression model-based training
     * \param[in] partialResult Pointer to an object containing the input data
     * \param[in] method        Computation method of the algorithm
     * \param[in] parameter     %Parameter of linear regression model-based training
     */
    template<typename algorithmFPType>
    void allocate(const daal::algorithms::PartialResult *partialResult, const Parameter *parameter, const int method)
    {
        const PartialResult *partialRes = static_cast<const PartialResult *>(partialResult);

        if(method == qrDense)
        {
            algorithmFPType dummy = 1.0;
            set(model, services::SharedPtr<daal::algorithms::linear_regression::Model>(new ModelQR(partialRes->getNFeatures(),
                                                                                                   partialRes->getNDependentVariables(),
                                                                                                   *parameter, dummy)));
        }
        else if(method == normEqDense)
        {
            algorithmFPType dummy = 1.0;
            set(model, services::SharedPtr<daal::algorithms::linear_regression::Model>(new ModelNormEq(partialRes->getNFeatures(),
                                                                                                       partialRes->getNDependentVariables(),
                                                                                                       *parameter, dummy)));
        }
    }

    /**
     * Checks the result of linear regression model-based training
     * \param[in] input   %Input object for the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        services::SharedPtr<daal::algorithms::linear_regression::Model> model = get(training::model);
        if(model->getBeta() == 0) { this->_errors->add(services::ErrorNullOutputNumericTable); return; }
    }

    /**
     * Checks the result of the linear regression model-based training
     * \param[in] pr      %PartialResult of the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     */
    void check(const daal::algorithms::PartialResult *pr, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        services::SharedPtr<daal::algorithms::linear_regression::Model> model = get(training::model);
        if(model->getBeta() == 0) { this->_errors->add(services::ErrorNullOutputNumericTable); return; }
    }

    /**
     * Returns the serialization tag of the linear regression model-based training result
     * \return         Serialization tag of the linear regression model-based training result
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_LINEAR_REGRESSION_TRAINING_RESULT_ID; }

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
using interface1::InputIface;
using interface1::Input;
using interface1::DistributedInput;
using interface1::PartialResult;
using interface1::Result;

} // namespace training
}
}
} // namespace daal
#endif
