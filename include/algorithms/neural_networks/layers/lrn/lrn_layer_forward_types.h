/* file: lrn_layer_forward_types.h */
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
//  Implementation of the forward local response normalization layer types.
//--
*/

#ifndef __NEURAL_NETWORKS__LRN_LAYER_FORWARD_TYPES_H__
#define __NEURAL_NETWORKS__LRN_LAYER_FORWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_forward_types.h"
#include "algorithms/neural_networks/layers/lrn/lrn_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace lrn
{
/**
 * \brief Contains classes for the forward local response normalization layer
 */
namespace forward
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LRN__FORWARD__INPUT"></a>
 * \brief %Input parameters for the forward local response normalization layer
 */
class Input : public layers::forward::Input
{
public:
    /** Default constructor */
    Input() {};

    /**
     * Returns an input object for the forward local response normalization layer
     */
    using layers::forward::Input::get;

    /**
     * Sets an input object for the forward local response normalization layer
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
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LRN__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of the forward local response normalization layer
 */
class Result : public layers::forward::Result
{
public:
    /** \brief Constructor */
    Result() : layers::forward::Result() {};
    virtual ~Result() {};

    /**
     * Returns the result of the forward local response normalization layer
     */
    using layers::forward::Result::get;

    /**
     * Sets the result of the forward local response normalization layer
     */
    using layers::forward::Result::set;

    /**
     * Returns the result of the forward local response normalization layer
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
     * Sets the result of the forward local response normalization layer
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
     * Checks the result of the forward local response normalization layer
     * \param[in] input   %Input object for the layer
     * \param[in] par     %Parameter of the layer
     * \param[in] method  Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        layers::forward::Result::check(input, par, method);

        services::SharedPtr<data_management::Tensor> dataTable = (static_cast<const layers::forward::Input *>(input))->get(
                                                                     layers::forward::data);
        services::SharedPtr<data_management::Tensor> resultTable = get(layers::forward::value);
        services::SharedPtr<data_management::NumericTable> dimensionTable = (static_cast<const Parameter *>(par))->dimension;

        data_management::BlockDescriptor<int> block;
        dimensionTable->getBlockOfRows(0, 1, data_management::readOnly, block);
        int *dataInt = block.getBlockPtr();
        size_t dim = dataInt[0];
        if(dim >= dataTable->getNumberOfDimensions())
        {
            services::SharedPtr<services::Error> error(new services::Error());
            error->setId(services::ErrorIncorrectParameter);
            error->addStringDetail(services::ArgumentName, "dimension");
            this->_errors->add(error);
        }
        dimensionTable->releaseBlockOfRows(block);

        if(dataTable.get() == 0)                   { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        if(resultTable.get() == 0)                 { this->_errors->add(services::ErrorNullOutputNumericTable); return; }
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
     * Allocates memory to store the result of the forward local response normalization layer
     * \param[in] input Pointer to an object containing the input data
     * \param[in] method Computation method for the layer
     * \param[in] parameter %Parameter of the forward local response normalization layer
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const layers::forward::Input *in = static_cast<const layers::forward::Input * >(input);

        Argument::set(layers::forward::value, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenTensor<algorithmFPType>(in->get(layers::forward::data)->getDimensions(),
                                                                              data_management::Tensor::doAllocate)));
        Argument::set(layers::forward::resultForBackward, services::SharedPtr<LayerData>(new LayerData()));
        allocateLayerData<algorithmFPType>(input, parameter, method);
    }

    /**
      * Allocates memory for storing final results of the relu layer
      * \param[in] input     Pointer to the structure with result objects
      * \param[in] parameter Pointer to the structure of layer parameters
      * \param[in] method    Computation method
      */
    template <typename algorithmFPType>
    void allocateLayerData(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const layers::forward::Input *in = static_cast<const layers::forward::Input * >(input);
        set(lrn::auxData, in->get(layers::forward::data));
        set(lrn::auxSmBeta, services::SharedPtr<data_management::HomogenTensor<algorithmFPType> >(
                new data_management::HomogenTensor<algorithmFPType>(in->get(layers::forward::data)->getDimensions(),
                                                                    data_management::Tensor::doAllocate)));
    }

    /**
    * Returns the serialization tag of the forward local response normalization layer result
    * \return         Serialization tag of the forward local response normalization layer result
    */
    int getSerializationTag() { return SERIALIZATION_NEURAL_NETWORKS_LAYERS_LRN_FORWARD_RESULT_ID; }

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
} // namespace lrn
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
