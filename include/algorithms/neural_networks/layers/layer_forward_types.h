/* file: layer_forward_types.h */
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
//  Implementation of neural_networks Network layer.
//--
*/

#ifndef __NEURAL_NETWORKS__LAYERS__FORWARD__TYPES__H__
#define __NEURAL_NETWORKS__LAYERS__FORWARD__TYPES__H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "services/collection.h"
#include "data_management/data/data_collection.h"
#include "algorithms/neural_networks/layers/layer_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for training and prediction using neural network
 */
namespace neural_networks
{
/**
 * \brief Contains classes for neural network layers
 */
namespace layers
{
/**
 * \brief Contains classes for the forward stage of the neural network layer
 */
namespace forward
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FORWARD__INPUTID"></a>
 * Available identifiers of input objects for the layer algorithm
 */
enum InputId
{
    data = 0,       /*!< Input data */
    weights = 1,    /* Weights of the neural network layer */
    biases = 2      /* Biases of the neural network layer */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FORWARD__INPUTLAYERDATAID"></a>
 * Available identifiers of input objects for the layer algorithm
 */
enum InputLayerDataId
{
    inputLayerData = 3
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FORWARD__RESULTID"></a>
 * Available identifiers of results for the layer algorithm
 */
enum ResultId
{
    value = 0     /*!< Table to store the result */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS___RESULTLAYERDATAID"></a>
 * Available identifiers of results for the layer algorithm
 */
enum ResultLayerDataId
{
    resultForBackward = 1     /*!< Data for backward step */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FORWARD__INPUT_IFACE"></a>
 * \brief Abstract class that specifies interface of the input objects for the neural network layer algorithm
 */
class InputIface : public daal::algorithms::Input
{
public:
    /** \brief Constructor
    * \param[in] nElements    Number of elements in Input structure
    */
    InputIface(size_t nElements) : daal::algorithms::Input(nElements) {}
    virtual ~InputIface() {}
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FORWARD__INPUT"></a>
 * \brief %Input objects for layer algorithm
 */
class Input : public InputIface
{
public:
    /**
     * Constructs input objects for the forward layer of neural network
     * \param[in] nElements     Number of input objects for the forward layer
     */
    Input(size_t nElements = 4) : InputIface(nElements)
    {
        Argument::set(inputLayerData, services::SharedPtr<LayerData>(new LayerData()));
    };

    virtual ~Input() {}

    /**
     * Returns input Tensor of the layer algorithm
     * \param[in] id    Identifier of the input tensor
     * \return          %Input tensor that corresponds to the given identifier
     */
    services::SharedPtr<data_management::Tensor> get(forward::InputId id) const
    {
        return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets input for the layer algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const services::SharedPtr<data_management::Tensor> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
    * Returns input InputLayerData of the layer algorithm
    * \param[in] id    Identifier of the input object
    * \return          %Input InputLayerData that corresponds to the given identifier
    */
    services::SharedPtr<LayerData> get(forward::InputLayerDataId id) const
    {
        return services::staticPointerCast<LayerData, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets input for the layer algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputLayerDataId id, const services::SharedPtr<LayerData> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks an input object for the layer algorithm
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method of the algorithm
     */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 4) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        services::SharedPtr<services::Error> error = checkTensor(get(data), "data in Input");
        if (error) { this->_errors->add(error); return; }
    }

    /**
     * Returns the layout of the input object for the layer algorithm
     * \return Layout of the input object for the layer algorithm
     */
    virtual LayerInputLayout getLayout() { return tensorInput; }

    /**
     * Returns dimensions of weights tensor
     * \return Dimensions of weights tensor
     */
    virtual const services::Collection<size_t> getWeightsSizes(const layers::Parameter *parameter) const
    {
        return services::Collection<size_t>();
    }

    /**
     * Returns dimensions of biases tensor
     * \return Dimensions of biases tensor
     */
    virtual const services::Collection<size_t> getBiasesSizes(const layers::Parameter *parameter) const
    {
        return services::Collection<size_t>();
    }

protected:
    services::SharedPtr<services::Error> checkTensor(const services::SharedPtr<data_management::Tensor> &tensor,
                                                     const char *argumentName, const services::Collection<size_t> *dims = NULL) const
    {
        using namespace daal::services;

        SharedPtr<Error> error;
        if (!tensor)
        {
            error = SharedPtr<Error>(new Error(ErrorNullTensor));
        }
        else
        {
            error = tensor->check(dims);
        }
        if (error) { error->addStringDetail(services::ArgumentName, argumentName); }
        return error;
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FORWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method of the layer algorithm
 */
class Result : public daal::algorithms::Result
{
public:
    /** \brief Constructor */
    Result() : daal::algorithms::Result(2) {};

    virtual ~Result() {};

    /**
     * Allocates memory to store the results of layer
     * \param[in] input  Pointer to the input structure
     * \param[in] par    Pointer to the parameter structure
     * \param[in] method Computation method of the algorithm
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method) {};

    /**
    * Returns collection of dimensions of layer output
    * \param[in] inputSize   Collection of input tensor dimensions
    * \param[in] par         Parameters of the algorithm
    * \param[in] method      Method of the algorithm
    * \return    Collection of dimensions of layer output
    */
    virtual const services::Collection<size_t> getValueSize(const services::Collection<size_t> &inputSize,
                                                            const daal::algorithms::Parameter *par, const int method) const = 0;

    /**
    * Returns collection of dimensions of layer output
    * \param[in] inputSize Collection of input tensors dimensions
    * \param[in] par       Parameters of the algorithm
    * \param[in] method    Method of the algorithm
    * \return    Collection of dimensions of layer output
    */
    virtual services::Collection<size_t> getValueSize(const services::Collection< services::Collection<size_t> > &inputSize,
                                                      const daal::algorithms::Parameter *par, const int method)
    {
        return services::Collection<size_t>();
    };

    /**
    * Returns collection of dimensions of layer output
    * \param[in] inputSize   Collection of input tensor dimensions
    * \param[in] par         Parameters of the algorithm
    * \param[in] method      Method of the algorithm
    * \return    Collection of dimensions of layer output
    */
    virtual services::Collection< services::Collection<size_t> > getValueCollectionSize(const services::Collection<size_t> &inputSize,
                                                                                        const daal::algorithms::Parameter *par, const int method)
    {
        return services::Collection< services::Collection<size_t> >();
    };

    /**
     * Returns result of the layer algorithm
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::Tensor> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Returns result of the layer algorithm
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    services::SharedPtr<LayerData> get(ResultLayerDataId id) const
    {
        return services::staticPointerCast<LayerData, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the result of the layer algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const services::SharedPtr<data_management::Tensor> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
    * Sets the result of the layer algorithm
    * \param[in] id    Identifier of the result
    * \param[in] ptr   Pointer to the result
    */
    void set(ResultLayerDataId id, const services::SharedPtr<LayerData> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Returns the serialization tag of the result
     * \return         Serialization tag of the result
     */
    int getSerializationTag() { return SERIALIZATION_NEURAL_NETWORKS_LAYERS_FORWARD_RESULT_ID; }

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

    /**
     * Checks the result object for the layer algorithm
     * \param[in] input         %Input of the algorithm
     * \param[in] parameter     %Parameter of algorithm
     * \param[in] method        Computation method of the algorithm
     */
    virtual void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
                       int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        services::SharedPtr<services::Error> error = checkTensor(get(value), "value in Result");
        if (error) { this->_errors->add(error); return; }

        services::SharedPtr<LayerData> layerData = get(resultForBackward);
        if (!layerData) { this->_errors->add(services::ErrorNullLayerData); return; }
    }

    /**
     * Returns the layout of the result object for the layer algorithm
     * \return Layout of the result object for the layer algorithm
     */
    virtual LayerResultLayout getLayout() { return tensorResult; }

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }

    services::SharedPtr<services::Error> checkTensor(const services::SharedPtr<data_management::Tensor> &tensor,
                                                     const char *argumentName, const services::Collection<size_t> *dims = NULL) const
    {
        using namespace daal::services;

        SharedPtr<Error> error;
        if (!tensor)
        {
            error = SharedPtr<Error>(new Error(ErrorNullTensor));
        }
        else
        {
            error = tensor->check(dims);
        }
        if (error) { error->addStringDetail(services::ArgumentName, argumentName); }
        return error;
    }
};
} // interface1
using interface1::InputIface;
using interface1::Input;
using interface1::Result;
} // forward
} // namespace layer
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
