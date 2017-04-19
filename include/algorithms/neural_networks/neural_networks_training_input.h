/* file: neural_networks_training_input.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of neural network algorithm interface.
//--
*/

#ifndef __NEURAL_NETWORKS_TRAINING_INPUT_H__
#define __NEURAL_NETWORKS_TRAINING_INPUT_H__

#include "algorithms/algorithm.h"

#include "data_management/data/tensor.h"
#include "data_management/data/data_serialize.h"
#include "data_management/data/data_collection.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/neural_networks_training_model.h"
#include "algorithms/neural_networks/neural_networks_training_partial_result.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for training and prediction using neural network
 */
namespace neural_networks
{
namespace training
{
/**
 * @ingroup neural_networks_training
 * @{
 */
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__TRAINING__INPUTID"></a>
 * \brief Available identifiers of %input objects for the neural network model based training
 */
enum InputId
{
    data        = 0,        /*!< Training data set */
    groundTruth = 1         /*!< Ground-truth results for the training data set */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__TRAINING__INPUTCOLLECTIONID"></a>
 * \brief Available identifiers of %input collection objects for the neural network model based training
 */
enum InputCollectionId
{
    groundTruthCollection = 2   /*!< Data collection of ground-truth results for the training data sets */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__TRAINING__STEP1LOCALINPUTID"></a>
 * \brief Available identifiers of %input objects for the neural network model based training
 */
enum Step1LocalInputId
{
    inputModel  = 3         /*!< Input model */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__TRAINING__STEP2MASTERINPUTID"></a>
 * \brief Partial results from the previous steps in the distributed processing mode required by the second distributed step of the algorithm
 */
enum Step2MasterInputId
{
    partialResults = 0  /*!< Partial results of the neural network training algorithm computed on the first step and to be transferred  to the
                             second step in the distributed processing mode */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__INPUT"></a>
 * \brief Input objects of the neural network training algorithm
 */
class Input : public daal::algorithms::Input
{
public:
    Input(size_t nElements = 3) : daal::algorithms::Input(nElements)
    {

        set(groundTruthCollection, data_management::KeyValueDataCollectionPtr(new data_management::KeyValueDataCollection()));
    }
    Input(const Input& other) : daal::algorithms::Input(other){}

    virtual ~Input() {}

    /**
     * Returns %input object for the neural network training algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::TensorPtr get(InputId id) const
    {
               using namespace daal::data_management;
        if (id == groundTruth)
        {
            KeyValueDataCollectionPtr collection = get(groundTruthCollection);
            if (!collection) { return TensorPtr(); }
            if (collection->size() == 0) { return TensorPtr(); }
            return Tensor::cast(collection->getValueByIndex(0));
        }
        else
        {
            return data_management::Tensor::cast(Argument::get(id));
        }
    }

    /**
     * Returns %input object for the neural network training algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(InputCollectionId id) const
    {
        return services::staticPointerCast<data_management::KeyValueDataCollection, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Returns %input object for the neural network training algorithm
     * \param[in] id    Identifier of the %input object
     * \param[in] key   Key to use to retrieve data
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::TensorPtr get(InputCollectionId id, size_t key) const
    {
        data_management::KeyValueDataCollectionPtr collection = get(id);
        if (!collection) { return data_management::TensorPtr(); }
        return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*collection)[key]);
    }

    /**
     * Sets %input object for the neural network training algorithm
     * \param[in] id      Identifier of the %input object
     * \param[in] value   Pointer to the %input object
     */
    void set(InputId id, const data_management::TensorPtr &value)
    {
        if (id == groundTruth)
        {
            add(groundTruthCollection, 0, value);
        }
        else
        {
            Argument::set(id, value);
        }
    }

    /**
     * Sets %input object for the neural network training algorithm
     * \param[in] id      Identifier of the %input object
     * \param[in] value   Pointer to the %input object
     */
    void set(InputCollectionId id, const data_management::KeyValueDataCollectionPtr &value)
    {
        Argument::set(id, value);
    }

    /**
     * Sets %input object for the neural network training algorithm
     * \param[in] id      Identifier of the %input object
     * \param[in] key     Key to use to retrieve data
     * \param[in] value   Pointer to the %input object
     */
    void add(InputCollectionId id, size_t key, const data_management::TensorPtr &value)
    {
        data_management::KeyValueDataCollectionPtr collection = get(id);
        if (!collection) { return; }
        (*collection)[key] = value;
    }

    /**
     * Checks %input object for the neural network algorithm
     * \param[in] par     Algorithm %parameter
     * \param[in] method  Computatiom method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        using namespace daal::data_management;
        using namespace daal::services;
        const Parameter *param = static_cast<const Parameter *>(par);
        TensorPtr dataTensor = get(data);
        services::Status s;
        DAAL_CHECK_STATUS(s, checkTensor(dataTensor.get(), dataStr()))

        size_t nSamples = dataTensor->getDimensionSize(0);
        DAAL_CHECK_EX(nSamples >= param->optimizationSolver->parameter->batchSize, services::ErrorIncorrectParameter, services::ParameterName, batchSizeStr());

        KeyValueDataCollectionPtr groundTruthTensorCollection = get(groundTruthCollection);

        if (!groundTruthTensorCollection) return services::Status(ErrorNullInputDataCollection);

        if (groundTruthTensorCollection->size() == 0) return services::Status(ErrorIncorrectNumberOfElementsInInputCollection);

        size_t nLastLayers = groundTruthTensorCollection->size();
        for (size_t i = 0; i < nLastLayers; i++)
        {
            size_t layerId = groundTruthTensorCollection->getKeyByIndex((int)i);
            TensorPtr groundTruthTensor = get(groundTruthCollection, layerId);
            if (!groundTruthTensor)
            {
                SharedPtr<Error> error = Error::create(services::ErrorNullTensor, ArgumentName, groundTruthLabelsStr());
                error->addIntDetail(ElementInCollection, (int)layerId);
                return services::Status(error);
            }
            Collection<size_t> expectedDims = groundTruthTensor->getDimensions();
            expectedDims[0] = nSamples;
            DAAL_CHECK_STATUS(s, checkTensor(groundTruthTensor.get(), groundTruthLabelsStr(), &expectedDims))
        }
        return s;
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__DISTRIBUTEDINPUT"></a>
 * \brief Input objects of the neural network training algorithm in the distributed processing mode
 */
template<ComputeStep step>
class DAAL_EXPORT DistributedInput
{};

/**
 * <a name="DAAL-CLASS-NEURAL_NETWORKS__TRAINING__DISTRIBUTEDINPUT_STEP1LOCAL"></a>
 * \brief Input objects of the neural network training algorithm in the distributed processing mode
 */
template<>
class DistributedInput<step1Local> : public Input
{
public:
    DistributedInput(size_t nElements = 4) : Input(nElements) {};
    DistributedInput(const DistributedInput& other) : Input(other){}

    virtual ~DistributedInput() {};

    using Input::set;
    using Input::get;

    /**
     * Returns %input object for the neural network training algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    ModelPtr get(Step1LocalInputId id) const
    {
        return services::staticPointerCast<Model, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets %input object for the neural network training algorithm
     * \param[in] id      Identifier of the %input object
     * \param[in] value   Pointer to the %input object
     */
    void set(Step1LocalInputId id, const ModelPtr &value)
    {
        Argument::set(id, value);
    }

    /**
     * Checks %input object for the neural network algorithm
     * \param[in] par     Algorithm %parameter
     * \param[in] method  Computatiom method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        return Input::check(par, method);
    }
};

/**
 * <a name="DAAL-CLASS-NEURAL_NETWORKS__TRAINING__DISTRIBUTEDINPUT_STEP2MASTER"></a>
 * \brief Input objects of the neural network training algorithm
 */
template<>
class DistributedInput<step2Master> : public daal::algorithms::Input
{
public:
    DistributedInput() : daal::algorithms::Input(1)
    {
        set(partialResults, data_management::KeyValueDataCollectionPtr(new data_management::KeyValueDataCollection()));
    }

    DistributedInput(const DistributedInput& other) : daal::algorithms::Input(other){}

    virtual ~DistributedInput() {};

    /**
     * Returns %input object for the neural network training algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(Step2MasterInputId id) const
    {
        using namespace daal::data_management;
        using namespace daal::services;
        return staticPointerCast<KeyValueDataCollection, SerializationIface>(Argument::get(id));
    }

    /**
     * Sets %input object for the neural network training algorithm
     * \param[in] id      Identifier of the %input object
     * \param[in] value   Pointer to the %input object
     */
    void set(Step2MasterInputId id, const data_management::KeyValueDataCollectionPtr &value)
    {
        Argument::set(id, value);
    }

    /**
     * Adds input object to KeyValueDataCollection of the neural network distributed training algorithm
     * \param[in] id    Identifier of input object
     * \param[in] key   Key to use to retrieve data
     * \param[in] value Pointer to the input object value
     */
    void add(Step2MasterInputId id, size_t key, const PartialResultPtr &value)
    {
        data_management::KeyValueDataCollectionPtr collection = get(id);
        if (!collection) { return; }
        (*collection)[key] = value;
    }

    /**
     * Checks %input object for the neural network algorithm
     * \param[in] par     Algorithm %parameter
     * \param[in] method  Computatiom method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        return services::Status();
    }
};

} // namespace interface1
using interface1::Input;
using interface1::DistributedInput;

/** @} */
}
}
}
} // namespace daal
#endif
