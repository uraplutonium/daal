/* file: logistic_layer_forward_types.h */
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
//  Implementation of the forward logistic layer.
//--
*/

#ifndef __NEURAL_NETWORKS__LOGISTIC_LAYER_FORWARD_TYPES_H__
#define __NEURAL_NETWORKS__LOGISTIC_LAYER_FORWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_forward_types.h"
#include "algorithms/neural_networks/layers/logistic/logistic_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * \brief Contains classes for the logistic layer
 */
namespace logistic
{
/**
 * \brief Contains classes for the forward logistic layer
 */
namespace forward
{
/**
* \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
*/
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOGISTIC__FORWARD__INPUT"></a>
 * \brief %Input objects for the forward logistic layer
 */
class Input : public layers::forward::Input
{
public:
    /** \brief Default constructor */
    Input() {};

    /**
    * Gets the input of the forward logistic layer
    */
    using layers::forward::Input::get;
    /**
    * Sets the input of the forward logistic layer
    */
    using layers::forward::Input::set;

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

    virtual ~Input() {}
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOGISTIC__FORWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method of the forward logistic layer
 */
class Result : public layers::forward::Result
{
public:
    /** \brief Default constructor */
    Result() : layers::forward::Result() {};
    virtual ~Result() {};

    /**
     * Returns the result of the forward logistic layer
     */
    using layers::forward::Result::get;

    /**
    * Sets the result of the forward logistic layer
    */
    using layers::forward::Result::set;

    /**
     * Returns the result of the forward logistic layer
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::Tensor> get(LayerDataId id) const
    {
        services::SharedPtr<layers::LayerData> layerData =
            services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
        return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
    }

    /**
     * Sets the result of the forward logistic layer
     * \param[in] id      Identifier of the result
     * \param[in] value   Result
     */
    void set(LayerDataId id, const services::SharedPtr<data_management::Tensor> &value)
    {
        services::SharedPtr<layers::LayerData> layerData =
            services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
        (*layerData)[id] = value;
    }

    /**
     * Checks the result of the forward logistic layer
     * \param[in] input   %Input object for the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        layers::forward::Result::check(input, par, method);

        const Input *in = static_cast<const Input *>(input);

        services::SharedPtr<services::Error> error = checkTensor(in->get(layers::forward::data), "data");
        if (error) { this->_errors->add(error); return; }

        error = checkTensor(get(layers::forward::value), "value");
        if (error) { this->_errors->add(error); return; }
    }

    /**
    * Allocates memory to store the result of the forward logistic layer
    * \param[in] input   Pointer to an object containing the input data
    * \param[in] par      %Parameter of the algorithm
    * \param[in] method   Computation method for the algorithm
    */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
    {
        const layers::forward::Input *in = static_cast<const layers::forward::Input * >(input);

        Argument::set(layers::forward::value, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenTensor<algorithmFPType>(in->get(layers::forward::data)->getDimensions(),
                                                                              data_management::Tensor::doAllocate)));
        Argument::set(layers::forward::resultForBackward, services::SharedPtr<LayerData>(new LayerData()));
        allocateLayerData<algorithmFPType>(input, par, method);
    }

    /**
     * Returns dimensions of value tensor
     * \return Dimensions of value tensor
     */
    virtual const services::Collection<size_t> getValueSize(const services::Collection<size_t> &inputSize,
                                                            const daal::algorithms::Parameter *par, const int method) const DAAL_C11_OVERRIDE
    {
        return inputSize;
    }

    /**
    * Allocates memory to store the result of the forward logistic layer
    * \param[in] input   Pointer to an object containing the input data
    * \param[in] par      %Parameter of the algorithm
    * \param[in] method   Computation method for the algorithm
    */
    template <typename algorithmFPType>
    void allocateLayerData(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
    {
        set(auxValue, get(layers::forward::value));
    }

    /**
     * Returns the serialization tag of the result
     * \return         Serialization tag of the result
     */
    int getSerializationTag() { return SERIALIZATION_NEURAL_NETWORKS_LAYERS_LOGISTIC_FORWARD_RESULT_ID; }

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
} // namespace forward
} // namespace logistic
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
