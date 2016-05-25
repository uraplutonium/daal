/* file: implicit_als_training_types.h */
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

#ifndef __IMPLICIT_ALS_TRAINING_TYPES_H__
#define __IMPLICIT_ALS_TRAINING_TYPES_H__

#include "algorithms/implicit_als/implicit_als_model.h"
#include "data_management/data/csr_numeric_table.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes of the implicit ALS algorithm
 */
namespace implicit_als
{
/**
 * \brief Contains classes of the implicit ALS training algorithm
 */
namespace training
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__METHOD"></a>
 * Available methods for training the implicit ALS model
 */
enum Method
{
    defaultDense = 0,   /*!< Default: method proposed by Hu, Koren, Volinsky for input data stored in the dense format */
    fastCSR = 1         /*!< Method proposed by Hu, Koren, Volinsky for input data stored in the compressed sparse row (CSR) format */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__NUMERICTABLEINPUTID"></a>
 * Available identifiers of input numeric table objects for the implicit ALS training algorithm
 */
enum NumericTableInputId
{
    data = 0                        /*!< Input data table that contains ratings */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__MODELINPUTID"></a>
 * Available identifiers of input model objects for the implicit ALS training algorithm
 */
enum ModelInputId
{
    inputModel = 1                  /*!< Initial model that contains initialized factors */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__PARTIALMODELINPUTID"></a>
 * Available identifiers of input partial model objects of the implicit ALS training algorithm
 */
enum PartialModelInputId
{
    partialModel = 0                /*!< Partial model that contains factors obtained
                                         in the previous step of the distributed processing mode */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__MASTERINPUTID"></a>
 * Partial results obtained in the previous step and required by the second step
 * of the distributed processing mode
 */
enum MasterInputId
{
    inputOfStep2FromStep1 = 0       /*!< Partial results of the implicit ALS training algorithm computed in the first step
                                         and to be transferred to the second step of the distributed processing mode */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP1ID"></a>
 * Available types of partial results of the implicit ALS training algorithm in the first step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep1Id
{
    outputOfStep1ForStep2 = 0       /*!< Partial results of the implicit ALS training algorithm computed in the first step
                                         and to be transferred to the second step of the distributed processing mode */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP2ID"></a>
 * Available types of partial results of the implicit ALS training algorithm in the second step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep2Id
{
    outputOfStep2ForStep4 = 0       /*!< Partial results of the implicit ALS training algorithm computed in the second step
                                         and to be transferred to the fourth step of the distributed processing mode */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__STEP3LOCALCOLLECTIONINPUTID"></a>
 * Available identifiers of input data collection objects for the implicit ALS training algorithm in the third step
 * of the distributed processing mode
 */
enum Step3LocalCollectionInputId
{
    partialModelBlocksToNode = 1    /*!< Key-value data collection that maps components of a partial model to local nodes:
                                         the i-th element of this collection is a numeric table that contains indices of the factors
                                         that should be transferred to the i-th node */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__STEP3LOCALNUMERICTABLEINPUTID"></a>
 * Available identifiers of input numeric table objects for the implicit ALS training algorithm in the third step
 * of the distributed processing mode
 */
enum Step3LocalNumericTableInputId
{
    offset = 2                      /*!< Pointer to the 1x1 numeric table that holds the global index of the starting row
                                         of the input partial model */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP3ID"></a>
 * Available types of partial results of the implicit ALS training algorithm in the third step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep3Id
{
    outputOfStep3ForStep4 = 0       /*!< Partial results of the implicit ALS training algorithm computed in the third step
                                         and to be transferred to the fourth step of the distributed processing mode */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__STEP4LOCALPARTIALMODELSINPUTID"></a>
 * Available identifiers of input key-value data collection objects for the implicit ALS training algorithm in the fourth step
 * of the distributed processing mode
 */
enum Step4LocalPartialModelsInputId
{
    partialModels = 0               /*!< Key-value data collection that contains partial models consisting of user factors/item factors
                                         computed in the third step of the distributed processing mode.
                                         Each element of the collection contains an object of the PartialModel class. */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__STEP4LOCALNUMERICTABLEINPUTID"></a>
 * Available identifiers of input numeric table objects for the implicit ALS training algorithm in the fourth step
 * of the distributed processing mode
 */
enum Step4LocalNumericTableInputId
{
    partialData = 1,                /*!< Pointer to the CSR numeric table that holds a block of either users or items from the input data set */
    inputOfStep4FromStep2 = 2       /*!< Pointer to the nFactors x nFactors numeric table computed in the second step
                                         of the distributed processing mode */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__PARTIALRESULTID"></a>
 * Available types of partial results of the implicit ALS training algorithm in the fourth step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep4Id
{
    outputOfStep4ForStep1 = 0,      /*!< Partial results of the implicit ALS training algorithm computed in the fourth step
                                         and to be transferred to the first step of the distributed processing mode */
    outputOfStep4ForStep3 = 0,      /*!< Partial results of the implicit ALS training algorithm computed in the fourth step
                                         and to be transferred to the third step of the distributed processing mode */
    outputOfStep4 = 0               /*!< Partial results of the implicit ALS training algorithm computed in the fourth step
                                                     and to be used in implicit ALS PartialModel-based prediction */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__IMPLICIT_ALS__TRAINING__RESULTID"></a>
 * \brief Available identifiers of the results of the implicit ALS training algorithm
 */
enum ResultId
{
    model = 0                       /*!< Implicit ALS model */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INPUT"></a>
 * \brief %Input objects for the implicit ALS training algorithm
 */
class Input : public daal::algorithms::Input
{
public:
    Input() : daal::algorithms::Input(2) {}

    virtual ~Input() {}

    /**
     * Returns the input numeric table object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(NumericTableInputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable,
               data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Returns the input initial model object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<Model> get(ModelInputId id) const
    {
        return services::staticPointerCast<Model, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the input numeric table object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(NumericTableInputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Sets the input initial model object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(ModelInputId id, const services::SharedPtr<Model> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Returns the number of users equal to the number of rows in the input data set
     * \return Number of users
     */
    size_t getNumberOfUsers() const { return get(data)->getNumberOfRows(); }

    /**
     * Returns the number of items equal to the number of columns in the input data set
     * \return Number of items
     */
    size_t getNumberOfItems() const { return get(data)->getNumberOfColumns(); }

    /**
     * Checks the parameters and input objects for the implicit ALS training algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        services::SharedPtr<data_management::NumericTable> dataTable = get(data);
        if(!dataTable) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        if(dataTable->getNumberOfColumns() < 1) { this->_errors->add(services::ErrorEmptyInputNumericTable); return; }
        if(dataTable->getNumberOfRows() < 1) { this->_errors->add(services::ErrorEmptyInputNumericTable); return; }

        if(!parameter) { this->_errors->add(services::ErrorNullParameterNotSupported); return; }
        const Parameter *alsParameter = static_cast<const Parameter *>(parameter);
        size_t nFactors = alsParameter->nFactors;
        if(nFactors < 1) { this->_errors->add(services::ErrorIncorrectParameter); return; }

        size_t nUsers = dataTable->getNumberOfRows();
        size_t nItems = dataTable->getNumberOfColumns();

        services::SharedPtr<Model> model = get(inputModel);
        if(!model) { this->_errors->add(services::ErrorNullModel); return; }

        services::SharedPtr<data_management::NumericTable> usersFactors = model->getUsersFactors();
        if(!usersFactors) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        if(usersFactors->getNumberOfColumns() != nFactors) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }
        if(usersFactors->getNumberOfRows() != nUsers) { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }

        services::SharedPtr<data_management::NumericTable> itemsFactors = model->getItemsFactors();
        if(!itemsFactors) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        if(itemsFactors->getNumberOfColumns() != nFactors) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }
        if(itemsFactors->getNumberOfRows() != nItems) { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDINPUT"></a>
 * \brief %Input objects for the implicit ALS training algorithm in the distributed processing mode
 */
template<ComputeStep step>
class DistributedInput
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDINPUT"></a>
 * \brief %Input objects for the implicit ALS training algorithm in the first step
 * of the distributed processing mode
 */
template<>
class DistributedInput<step1Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput() : daal::algorithms::Input(1) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<PartialModel> get(PartialModelInputId id) const
    {
        return services::staticPointerCast<PartialModel, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets an input object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(PartialModelInputId id, const services::SharedPtr<PartialModel> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks the parameters and input objects for the implicit ALS training algorithm
     * in the first step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        if (method != fastCSR) { this->_errors->add(services::ErrorMethodNotSupported); return; }
        if (size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        const Parameter *alsParameter = static_cast<const Parameter *>(parameter);
        size_t nFactors = alsParameter->nFactors;
        if (nFactors < 1) { this->_errors->add(services::ErrorIncorrectParameter); return; }

        services::SharedPtr<PartialModel> model = get(partialModel);
        if (!model) { this->_errors->add(services::ErrorNullPartialModel); return; }

        services::SharedPtr<data_management::NumericTable> factors = model->getFactors();
        services::SharedPtr<data_management::NumericTable> indices = model->getIndices();

        if (!factors || !indices) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        if (factors->getNumberOfColumns() != nFactors) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }
        if (indices->getNumberOfColumns() != 1)        { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }
        if (factors->getNumberOfRows() != indices->getNumberOfRows())
        { this->_errors->add(services::ErrorInconsistentNumberOfRows); return; }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP1"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the implicit ALS algorithm in the first step of the distributed processing mode
 */
class DistributedPartialResultStep1 : public daal::algorithms::PartialResult
{
public:
    /** Default constructor */
    DistributedPartialResultStep1() : daal::algorithms::PartialResult(1) {}

    virtual ~DistributedPartialResultStep1() {}

    /**
     * Returns a partial result of the implicit ALS training algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(DistributedPartialResultStep1Id id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets a partial result of the implicit ALS training algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep1Id id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Allocates memory to store a partial result of the implicit ALS training algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);
        size_t nFactors = algParameter->nFactors;
        Argument::set(outputOfStep1ForStep2, services::SharedPtr<data_management::SerializationIface>(
                new data_management::HomogenNumericTable<algorithmFPType>(
                        nFactors, nFactors, data_management::NumericTable::doAllocate)));
    }

    /**
     * Checks a partial result of the implicit ALS algorithm
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
               int method) const DAAL_C11_OVERRIDE
    {
        if (method != fastCSR) { this->_errors->add(services::ErrorMethodNotSupported); return; }
        if (size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        const Parameter *alsParameter = static_cast<const Parameter *>(parameter);
        size_t nFactors = alsParameter->nFactors;

        if (nFactors < 1) { this->_errors->add(services::ErrorIncorrectParameter); return; }
        services::SharedPtr<data_management::NumericTable> crossProduct = get(outputOfStep1ForStep2);

        size_t nRows    = crossProduct->getNumberOfRows();
        size_t nColumns = crossProduct->getNumberOfColumns();
        if (nRows != nFactors) { this->_errors->add(services::ErrorIncorrectNumberOfRowsInOutputNumericTable); return; }
        if (nRows != nColumns) { this->_errors->add(services::ErrorIncorrectNumberOfColumnsInOutputNumericTable); return; }

        const DistributedInput<step1Local> *alsInput = static_cast<const DistributedInput<step1Local> *>(input);
        services::SharedPtr<PartialModel> model = alsInput->get(partialModel);
        if (!model) { this->_errors->add(services::ErrorNullPartialModel); return; }
        services::SharedPtr<data_management::NumericTable> factors = model->getFactors();
        size_t nFactorsModel = factors->getNumberOfColumns();
        if (nColumns != nFactorsModel) { this->_errors->add(services::ErrorInconsistentNumberOfColumns); return; }
    }

    /**
     * Returns a serialization tag, a unique identifier of this class used in serialization
     * \return Serialization tag
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_IMPLICIT_ALS_DISTRIBUTED_PARTIAL_RESULT_STEP1_ID; }

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
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDINPUT"></a>
 * \brief %Input objects for the implicit ALS training algorithm in the second step of the
 * distributed processing mode
 */
template<>
class DistributedInput<step2Master> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput() : daal::algorithms::Input(1)
    {
        Argument::set(inputOfStep2FromStep1, services::SharedPtr<data_management::DataCollection>(new data_management::DataCollection()));
    }

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::DataCollection> get(MasterInputId id) const
    {
        return services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets an input object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(MasterInputId id, const services::SharedPtr<data_management::DataCollection> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Adds an input object for the implicit ALS training algorithm in the second step
     * of the distributed processing mode
     * \param[in] id            Identifier of the input object
     * \param[in] partialResult Pointer to the partial result obtained in the previous step of the distributed processing mode
     */
    void add(MasterInputId id, const services::SharedPtr<DistributedPartialResultStep1> &partialResult)
    {
        services::SharedPtr<data_management::DataCollection> collection =
                services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
        if (!collection) { this->_errors->add(services::ErrorNullInputDataCollection); return; }
        collection->push_back(partialResult->get(training::outputOfStep1ForStep2));
    }

    /**
     * Checks the parameters and input objects for the implicit ALS training algorithm in the second step
     * of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        if (method != fastCSR) { this->_errors->add(services::ErrorMethodNotSupported); return; }
        if (size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        services::SharedPtr<data_management::DataCollection> collection = get(inputOfStep2FromStep1);
        if (!collection) { this->_errors->add(services::ErrorNullInputDataCollection); return; }
        size_t nBlocks = collection->size();
        if (nBlocks == 0) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        const Parameter *alsParameter = static_cast<const Parameter *>(parameter);
        size_t nFactors = alsParameter->nFactors;
        if (nFactors < 1) { this->_errors->add(services::ErrorIncorrectParameter); return; }
        for (size_t i = 0; i < nBlocks; i++)
        {
            services::SharedPtr<data_management::NumericTable> nt =
                    services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>((*collection)[i]);
            if(!nt) { this->_errors->add(services::ErrorNullInputNumericTable); return; }

            if (nt->getNumberOfRows() != nFactors)
            { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }
            if (nt->getNumberOfColumns() != nFactors)
            { this->_errors->add(services::ErrorIncorrectNumberOfColumnsInInputNumericTable); return; }
        }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP2"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the implicit ALS algorithm in the second step of the distributed processing mode
 */
class DistributedPartialResultStep2 : public daal::algorithms::PartialResult
{
public:
    /** Default constructor */
    DistributedPartialResultStep2() : daal::algorithms::PartialResult(1) {}

    virtual ~DistributedPartialResultStep2() {}

    /**
     * Returns a partial result of the implicit ALS training algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(DistributedPartialResultStep2Id id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets a partial result of the implicit ALS training algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep2Id id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Allocates memory to store a partial result of the implicit ALS training algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);
        size_t nFactors = algParameter->nFactors;
        Argument::set(outputOfStep2ForStep4, services::SharedPtr<data_management::SerializationIface>(
                new data_management::HomogenNumericTable<algorithmFPType>(
                        nFactors, nFactors, data_management::NumericTable::doAllocate)));
    }

    /**
     * Checks a partial result of the implicit ALS training algorithm
     * \param[in] input     Pointer to the structure of input objects
     * \param[in] parameter Pointer to the structure of algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
               int method) const DAAL_C11_OVERRIDE
    {
        if (method != fastCSR) { this->_errors->add(services::ErrorMethodNotSupported); return; }
        if (size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        const Parameter *algParameter = static_cast<const Parameter *>(parameter);
        size_t nFactors = algParameter->nFactors;
        if (nFactors < 1) { this->_errors->add(services::ErrorIncorrectParameter); return; }

        services::SharedPtr<data_management::NumericTable> nt = get(outputOfStep2ForStep4);
        if (!nt) { this->_errors->add(services::ErrorNullOutputNumericTable); return; }

        if (nt->getNumberOfColumns() != nFactors)
        { this->_errors->add(services::ErrorIncorrectNumberOfColumnsInOutputNumericTable); return; }
        if (nt->getNumberOfRows() != nFactors)
        { this->_errors->add(services::ErrorIncorrectNumberOfRowsInOutputNumericTable); return; }
    }

    /**
     * Returns a serialization tag, a unique identifier of this class used in serialization
     * \return Serialization tag
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_IMPLICIT_ALS_DISTRIBUTED_PARTIAL_RESULT_STEP2_ID; }
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
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDINPUT"></a>
 * \brief %Input objects for the implicit ALS training algorithm in the third step of
 * the distributed processing mode
 */
template<>
class DistributedInput<step3Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput() : daal::algorithms::Input(3) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input partial model object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<PartialModel> get(PartialModelInputId id) const
    {
        return services::staticPointerCast<PartialModel, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Returns an input key-value data collection object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::KeyValueDataCollection> get(Step3LocalCollectionInputId id) const
    {
        return services::staticPointerCast<data_management::KeyValueDataCollection,
                                           data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Returns an input numeric table object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(Step3LocalNumericTableInputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable,
                                           data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets an input partial model object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(PartialModelInputId id, const services::SharedPtr<PartialModel> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Sets an input key-value data collection object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step3LocalCollectionInputId id, const services::SharedPtr<data_management::KeyValueDataCollection> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Sets an input numeric table object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step3LocalNumericTableInputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Returns the number of blocks of data used in distributed computations
     * \return Number of blocks of data
     */
    size_t getNumberOfBlocks() const
    {
        services::SharedPtr<data_management::KeyValueDataCollection> outBlocksCollection = get(partialModelBlocksToNode);
        if (!outBlocksCollection)
        { this->_errors->add(services::ErrorNullInputDataCollection); return 0; }

        return outBlocksCollection->size();
    }

    /**
     * Returns the index of the starting row of the input partial model
     * \return Index of the starting row of the input partial model
     */
    size_t getOffset() const
    {
        services::SharedPtr<data_management::NumericTable> offsetTable = get(offset);
        if (!offsetTable)
        { this->_errors->add(services::ErrorNullInputNumericTable); return 0; }

        data_management::BlockDescriptor<int> block;
        offsetTable->getBlockOfRows(0, 1, data_management::readOnly, block);
        size_t offset = (size_t)((block.getBlockPtr())[0]);
        offsetTable->releaseBlockOfRows(block);
        return offset;
    }

    /**
     * Returns the numeric table that contains the indices of factors that should be transferred to a specified node
     * \param[in] key Index of the node
     * \return Numeric table that contains the indices of factors that should be transferred to a specified node
     */
    services::SharedPtr<data_management::NumericTable> getOutBlockIndices(size_t key) const
    {
        services::SharedPtr<data_management::NumericTable> outBlockIndices;
        services::SharedPtr<data_management::KeyValueDataCollection> outBlocksCollection = get(partialModelBlocksToNode);
        if (!outBlocksCollection)
        { this->_errors->add(services::ErrorNullInputDataCollection); return outBlockIndices; }

        outBlockIndices = services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(
                (*outBlocksCollection)[key]);
        return outBlockIndices;
    }

    /**
     * Checks the parameters and input objects of the implicit ALS training algorithm in the first step of
     * the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        if (method != fastCSR) { this->_errors->add(services::ErrorMethodNotSupported); return; }
        if (size() != 3) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        const Parameter *algParameter = static_cast<const Parameter *>(parameter);
        size_t nFactors = algParameter->nFactors;
        if (nFactors < 1) { this->_errors->add(services::ErrorIncorrectParameter); return; }

        /* Check offset numeric table */
        services::SharedPtr<data_management::NumericTable> nt = get(offset);
        if (!nt) { this->_errors->add(services::ErrorNullOutputNumericTable); return; }

        if (nt->getNumberOfColumns() != 1)
        { this->_errors->add(services::ErrorIncorrectNumberOfColumnsInOutputNumericTable); return; }
        if (nt->getNumberOfRows() != 1)
        { this->_errors->add(services::ErrorIncorrectNumberOfRowsInOutputNumericTable); return; }

        /* Check input partial model */
        services::SharedPtr<PartialModel> model = get(partialModel);
        if (!model) { this->_errors->add(services::ErrorNullPartialModel); return; }
        services::SharedPtr<data_management::NumericTable> factors = model->getFactors();
        services::SharedPtr<data_management::NumericTable> indices = model->getIndices();

        if (!factors || !indices) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        if (factors->getNumberOfColumns() != nFactors) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }
        if (indices->getNumberOfColumns() != 1)        { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }
        size_t nRows = factors->getNumberOfRows();
        if (nRows != indices->getNumberOfRows())
        { this->_errors->add(services::ErrorInconsistentNumberOfRows); return; }

        /* Check input collection */
        services::SharedPtr<data_management::KeyValueDataCollection> collection = get(partialModelBlocksToNode);
        if (!collection) { this->_errors->add(services::ErrorNullInputDataCollection); return; }

        size_t nBlocks = collection->size();
        if (nBlocks == 0) { this->_errors->add(services::ErrorIncorrectNumberOfElementsInInputCollection); return; }

        for (size_t i = 0; i < nBlocks; i++)
        {
            services::SharedPtr<data_management::NumericTable> blockIndices =
                    services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(
                            collection->getValueByIndex((int)i));
            if (!blockIndices)
            { this->_errors->add(services::ErrorNullInputNumericTable); return; }

            if (blockIndices->getNumberOfColumns() != 1)
            { this->_errors->add(services::ErrorIncorrectNumberOfColumnsInInputNumericTable); return; }

            if (blockIndices->getNumberOfRows() > nRows)
            { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }
        }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP3"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the implicit ALS algorithm in the the third step of the distributed processing mode
 */
class DistributedPartialResultStep3 : public daal::algorithms::PartialResult
{
public:
    /** Default constructor */
    DistributedPartialResultStep3() : daal::algorithms::PartialResult(1) {}

    virtual ~DistributedPartialResultStep3() {}

    /**
     * Allocates memory to store a partial result of the implicit ALS training algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const DistributedInput<step3Local> *algInput = static_cast<const DistributedInput<step3Local> *>(input);
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);

        size_t nBlocks = algInput->getNumberOfBlocks();
        size_t offset = algInput->getOffset();

        services::Collection<size_t> _keys;
        data_management::DataCollection _values;
        for (size_t i = 0; i < nBlocks; i++)
        {
            services::SharedPtr<data_management::NumericTable> outBlockIndices = algInput->getOutBlockIndices(i);
            if (!outBlockIndices) { continue; }
            _keys.push_back(i);
            _values.push_back(services::SharedPtr<data_management::SerializationIface>(
                                  new PartialModel(*algParameter, offset, outBlockIndices, (algorithmFPType)0.0)));
        }
        services::SharedPtr<data_management::KeyValueDataCollection> modelsCollection =
            services::SharedPtr<data_management::KeyValueDataCollection> (new data_management::KeyValueDataCollection(_keys, _values));
        Argument::set(outputOfStep3ForStep4, modelsCollection);
    }

    /**
     * Returns a partial result of the implicit ALS training algorithm
     *
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    services::SharedPtr<data_management::KeyValueDataCollection> get(DistributedPartialResultStep3Id id) const
    {
        return services::staticPointerCast<data_management::KeyValueDataCollection,
               data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Returns a partial model obtained with the compute() method of the implicit ALS algorithm in the third step of the
     * distributed processing mode
     *
     * \param[in] id    Identifier of the partial result
     * \param[in] key   Index of the partial model in the key-value data collection
     * \return          Pointer to the partial model object
     */
    services::SharedPtr<PartialModel> get(DistributedPartialResultStep3Id id, size_t key) const
    {
        services::SharedPtr<PartialModel> model;
        services::SharedPtr<data_management::KeyValueDataCollection> collection = get(id);
        if (!collection) { this->_errors->add(services::ErrorNullOutputDataCollection); return model; }
        model = services::staticPointerCast<PartialModel, data_management::SerializationIface>((*collection)[key]);
        return model;
    }

    /**
     * Sets a partial result of the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(DistributedPartialResultStep3Id id, const services::SharedPtr<PartialModel> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks a partial result of the implicit ALS training algorithm
     * \param[in] input     Pointer to the structure of input objects
     * \param[in] parameter Pointer to the structure of algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
               int method) const DAAL_C11_OVERRIDE
    {
        if (method != fastCSR) { this->_errors->add(services::ErrorMethodNotSupported); return; }
        if (size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        const Parameter *algParameter = static_cast<const Parameter *>(parameter);
        size_t nFactors = algParameter->nFactors;
        if (nFactors < 1) { this->_errors->add(services::ErrorIncorrectParameter); return; }

        services::SharedPtr<data_management::KeyValueDataCollection> collection = get(outputOfStep3ForStep4);
        if (!collection) { this->_errors->add(services::ErrorNullOutputDataCollection); return; }

        size_t nBlocks = collection->size();

        for (size_t i = 0; i < nBlocks; i++)
        {
            PartialModel *model = static_cast<PartialModel *>(collection->getValueByIndex((int)i).get());
            if (!model) { this->_errors->add(services::ErrorNullPartialModel); return; }

            services::SharedPtr<data_management::NumericTable> factors = model->getFactors();
            services::SharedPtr<data_management::NumericTable> indices = model->getIndices();

            if (!factors || !indices) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
            if (factors->getNumberOfColumns() != nFactors) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }
            if (indices->getNumberOfColumns() != 1)        { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }
            size_t nRows = factors->getNumberOfRows();
            if (nRows != indices->getNumberOfRows())
            { this->_errors->add(services::ErrorInconsistentNumberOfRows); return; }
        }
    }

    /**
     * Returns a serialization tag, a unique identifier of this class used in serialization
     * \return Serialization tag
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_IMPLICIT_ALS_DISTRIBUTED_PARTIAL_RESULT_STEP3_ID; }

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
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDINPUT"></a>
 * \brief %Input objects for the implicit ALS training algorithm in the fourth step of
 * the distributed processing mode
 */
template<>
class DistributedInput<step4Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput() : daal::algorithms::Input(3) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input key-value data collection object for the implicit ALS training algorithm
     *
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier.
     *                  A key-value data collection contains partial models consisting of user factors/item factors
     *                  computed in the third step of the distributed processing mode
     */
    services::SharedPtr<data_management::KeyValueDataCollection> get(Step4LocalPartialModelsInputId id) const
    {
        return services::staticPointerCast<data_management::KeyValueDataCollection,
                                           data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Returns an input numeric table object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(Step4LocalNumericTableInputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable,
                                           data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets an input key-value data collection object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step4LocalPartialModelsInputId id, const services::SharedPtr<data_management::KeyValueDataCollection> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Sets an input numeric table object for the implicit ALS training algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step4LocalNumericTableInputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Returns the number of rows in the partial matrix of users factors/items factors
     * \return Number of rows in the partial matrix of factors
     */
    size_t getNumberOfRows() const
    {
        services::SharedPtr<data_management::NumericTable> dataTable = get(partialData);
        if (!dataTable) { this->_errors->add(services::ErrorNullInputNumericTable); return 0; }
        return dataTable->getNumberOfRows();
    }

    /**
     * Checks the parameters and input objects for the implicit ALS training algorithm in the first step of
     * the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        if (method != fastCSR) { this->_errors->add(services::ErrorMethodNotSupported); return; }
        if (size() != 3) { this->_errors->add(services::ErrorIncorrectNumberOfElementsInInputCollection); return; }

        const Parameter *algParameter = static_cast<const Parameter *>(parameter);
        size_t nFactors = algParameter->nFactors;
        if (nFactors < 1) { this->_errors->add(services::ErrorIncorrectParameter); return; }

        /* Check input numeric tables */
        services::SharedPtr<data_management::NumericTable> dataTable = get(partialData);
        services::SharedPtr<data_management::NumericTable> crossProduct = get(inputOfStep4FromStep2);
        if (!dataTable || !crossProduct) { this->_errors->add(services::ErrorNullInputNumericTable); return; }

        if (dynamic_cast<data_management::CSRNumericTableIface *>(dataTable.get()) == NULL)
        { this->_errors->add(services::ErrorIncorrectTypeOfInputNumericTable); return; }

        if (dataTable->getNumberOfRows() == 0 || crossProduct->getNumberOfRows() != nFactors)
        { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }

        if (dataTable->getNumberOfColumns() == 0 || crossProduct->getNumberOfColumns() != nFactors)
        { this->_errors->add(services::ErrorIncorrectNumberOfColumnsInInputNumericTable); return; }

        /* Check input data collection */
        services::SharedPtr<data_management::KeyValueDataCollection> collection = get(partialModels);
        if (!collection) { this->_errors->add(services::ErrorNullInputDataCollection); return; }

        size_t nBlocks = collection->size();

        for (size_t i = 0; i < nBlocks; i++)
        {
            PartialModel *model = static_cast<PartialModel *>(collection->getValueByIndex((int)i).get());
            if (!model) { this->_errors->add(services::ErrorNullPartialModel); return; }

            services::SharedPtr<data_management::NumericTable> factors = model->getFactors();
            services::SharedPtr<data_management::NumericTable> indices = model->getIndices();

            if (!factors || !indices) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
            if (factors->getNumberOfColumns() != nFactors) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }
            if (indices->getNumberOfColumns() != 1)        { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }
            if (factors->getNumberOfRows() != indices->getNumberOfRows())
            { this->_errors->add(services::ErrorInconsistentNumberOfRows); return; }
        }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP4"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the implicit ALS algorithm in the the fourth step of the distributed processing mode
 */
class DistributedPartialResultStep4 : public daal::algorithms::PartialResult
{
public:
    /** Default constructor */
    DistributedPartialResultStep4() : daal::algorithms::PartialResult(1) {}

    virtual ~DistributedPartialResultStep4() {}

    /**
     * Allocates memory to store a partial result of the implicit ALS training algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const DistributedInput<step4Local> *algInput = static_cast<const DistributedInput<step4Local> *>(input);
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);

        Argument::set(outputOfStep4ForStep1, services::SharedPtr<data_management::SerializationIface>(
                new PartialModel(*algParameter, algInput->getNumberOfRows(), (algorithmFPType)0.0)));
    }

    /**
     * Returns a partial result of the implicit ALS training algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    services::SharedPtr<PartialModel> get(DistributedPartialResultStep4Id id) const
    {
        return services::staticPointerCast<PartialModel, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets a partial result of the implicit ALS training algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the partial result
     */
    void set(DistributedPartialResultStep4Id id, const services::SharedPtr<PartialModel> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks a partial result of the implicit ALS training algorithm
     * \param[in] input     Pointer to the structure of input objects
     * \param[in] parameter Pointer to the structure of algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
               int method) const DAAL_C11_OVERRIDE
    {
        if (method != fastCSR) { this->_errors->add(services::ErrorMethodNotSupported); return; }
        if (size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfElementsInInputCollection); return; }

        const Parameter *algParameter = static_cast<const Parameter *>(parameter);
        size_t nFactors = algParameter->nFactors;
        if (nFactors < 1) { this->_errors->add(services::ErrorIncorrectParameter); return; }

        services::SharedPtr<PartialModel> model = get(outputOfStep4);
        if (!model) { this->_errors->add(services::ErrorNullPartialModel); return; }

        services::SharedPtr<data_management::NumericTable> factors = model->getFactors();
        services::SharedPtr<data_management::NumericTable> indices = model->getIndices();

        if (!factors || !indices) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        if (factors->getNumberOfColumns() != nFactors) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }
        if (indices->getNumberOfColumns() != 1)        { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }
        if (factors->getNumberOfRows() != indices->getNumberOfRows())
        { this->_errors->add(services::ErrorInconsistentNumberOfRows); return; }
    }

    /**
     * Returns a serialization tag, a unique identifier of this class used in serialization
     * \return Serialization tag
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_IMPLICIT_ALS_DISTRIBUTED_PARTIAL_RESULT_STEP4_ID; }
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
 * <a name="DAAL-CLASS-IMPLICIT_ALS__TRAINING__RESULT"></a>
 * \brief Provides methods to access the results obtained with the compute() method of the implicit ALS training algorithm
 * in the batch processing mode
 */
class Result : public daal::algorithms::Result
{
public:
    /** Default constructor */
    Result() : daal::algorithms::Result(1) {}

    /**
     * Allocates memory to store the results of the implicit ALS training algorithm
     * \param[in] input         Pointer to the input structure
     * \param[in] parameter     Pointer to the parameter structure
     * \param[in] method        Computation method of the algorithm
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const Input *algInput = static_cast<const Input *>(input);
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);

        size_t nUsers = algInput->getNumberOfUsers();
        size_t nItems = algInput->getNumberOfItems();
        Argument::set(model, services::SharedPtr<data_management::SerializationIface>(
                  new Model(nUsers, nItems, *algParameter, (algorithmFPType)0.0)));
    }

    /**
     * Returns the result of the implicit ALS training algorithm
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    services::SharedPtr<daal::algorithms::implicit_als::Model> get(ResultId id) const
    {
        return services::staticPointerCast<daal::algorithms::implicit_als::Model,
               data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the result of the implicit ALS training algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const services::SharedPtr<daal::algorithms::implicit_als::Model> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks the result of the implicit ALS training algorithm
     * \param[in] input       %Input object for the algorithm
     * \param[in] parameter   %Parameter of the algorithm
     * \param[in] method      Computation method of the algorithm
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
               int method) const DAAL_C11_OVERRIDE
    {
        if (size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfElementsInInputCollection); return; }

        const Parameter *algParameter = static_cast<const Parameter *>(parameter);
        size_t nFactors = algParameter->nFactors;
        if (nFactors < 1) { this->_errors->add(services::ErrorIncorrectParameter); return; }

        services::SharedPtr<Model> trainedModel = get(model);
        if (!trainedModel) { this->_errors->add(services::ErrorNullModel); return; }

        services::SharedPtr<data_management::NumericTable> usersFactors = trainedModel->getUsersFactors();
        services::SharedPtr<data_management::NumericTable> itemsFactors = trainedModel->getItemsFactors();

        if (!usersFactors || !itemsFactors) { this->_errors->add(services::ErrorNullOutputNumericTable); return; }
        if (usersFactors->getNumberOfColumns() != nFactors || itemsFactors->getNumberOfColumns() != nFactors)
        { this->_errors->add(services::ErrorIncorrectNumberOfColumnsInOutputNumericTable); return; }

        const Input *algInput = static_cast<const Input *>(input);
        services::SharedPtr<data_management::NumericTable> dataTable = algInput->get(data);

        if (usersFactors->getNumberOfRows() != dataTable->getNumberOfRows())
        { this->_errors->add(services::ErrorIncorrectNumberOfRowsInOutputNumericTable); return; }
        if (itemsFactors->getNumberOfRows() != dataTable->getNumberOfColumns())
        { this->_errors->add(services::ErrorIncorrectNumberOfRowsInOutputNumericTable); return; }
    }

    /**
     * Returns a serialization tag, a unique identifier of this class used in serialization
     * \return Serialization tag
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_IMPLICIT_ALS_TRAINING_RESULT_ID; }

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
} // interface1
using interface1::Input;
using interface1::DistributedInput;
using interface1::DistributedPartialResultStep1;
using interface1::DistributedPartialResultStep2;
using interface1::DistributedPartialResultStep3;
using interface1::DistributedPartialResultStep4;
using interface1::Result;

}
}
}
}

#endif
