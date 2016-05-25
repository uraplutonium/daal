/* file: convolution2d_layer_forward_types.h */
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
//  Implementation of forward two-dimensional (2D) convolution layer.
//--
*/

#ifndef __CONVOLUTION2D_LAYER_FORWARD_TYPES_H__
#define __CONVOLUTION2D_LAYER_FORWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_forward_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace convolution2d
{
/**
 * \brief Contains classes for the forward 2D convolution layer
 */
namespace forward
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONVOLUTION2D__FORWARD__INPUT"></a>
 * \brief %Input objects for the forward 2D convolution layer
 */
class Input : public layers::forward::Input
{
public:
    /**
     * Default constructor
     */
    Input() {};

    virtual ~Input() {}

    /**
     * Sets an input object for the forward 2D convolution layer
     */
    using layers::forward::Input::set;
    /**
     * Returns an input object for the forward 2D convolution layer
    */
    using layers::forward::Input::get;

    /**
    * Allocates memory to store the result of forward  2D convolution layer
     * \param[in] parameter %Parameter of forward 2D convolution layer
     * \param[in] method    Computation method for the layer
    */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Parameter *parameter, const int method)
    {
        using daal::services::SharedPtr;
        using daal::data_management::Tensor;
        using daal::data_management::HomogenTensor;

        const Parameter *param =  static_cast<const Parameter * >(parameter);

        if( !get(layers::forward::weights) )
        {
            SharedPtr<Tensor> tensor(new HomogenTensor<float>(getWeightsSizes(param), Tensor::doAllocate));
            param->weightsInitializer->input.set(initializers::data, tensor);
            param->weightsInitializer->compute();
            set(layers::forward::weights, tensor);
        }

        if( !get(layers::forward::biases) )
        {
            SharedPtr<Tensor> tensor(new HomogenTensor<float>(getBiasesSizes(param), Tensor::doAllocate));
            param->biasesInitializer->input.set(initializers::data, tensor);
            param->biasesInitializer->compute();
            set(layers::forward::biases, tensor);
        }
    }

    /**
     * Returns dimensions of weights tensor
     * \return Dimensions of weights tensor
     */
    virtual const services::Collection<size_t> getWeightsSizes(const layers::Parameter *parameter) const DAAL_C11_OVERRIDE
    {
        using daal::services::Collection;

        const Parameter *param =  static_cast<const Parameter *>(parameter);

        const Collection<size_t> &inDims = get(layers::forward::data)->getDimensions();

        Collection<size_t> wDims;

        wDims.push_back(param->nKernels);
        wDims.push_back(inDims[param->groupDimension]);
        wDims.push_back(param->kernelSize.size[0]);
        wDims.push_back(param->kernelSize.size[1]);

        return wDims;
    }

    /**
     * Returns dimensions of biases tensor
     * \return Dimensions of biases tensor
     */
    virtual const services::Collection<size_t> getBiasesSizes(const layers::Parameter *parameter) const DAAL_C11_OVERRIDE
    {
        using daal::services::Collection;

        const Parameter *param =  static_cast<const Parameter *>(parameter);

        Collection<size_t> bDims;
        bDims.push_back(param->nKernels);

        return bDims;
    }

    /**
     * Checks input object of the forward 2D convolution layer
     * \param[in] parameter %Parameter of layer
     * \param[in] method    Computation method of the layer
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        layers::forward::Input::check(parameter, method);
        if( this->_errors->size() > 0 ) { return; }

        const Parameter *algParameter = static_cast<const Parameter *>(parameter);

        services::SharedPtr<data_management::Tensor> dataTensor = get(layers::forward::data);
        services::SharedPtr<data_management::Tensor> wTensor = get(layers::forward::weights);
        services::SharedPtr<data_management::Tensor> bTensor = get(layers::forward::biases);

        if (!data_management::checkTensor(dataTensor.get(), this->_errors.get(), strData())) { return; }

        if( dataTensor->getDimensions().size() < 4 )
        { this->_errors->add( services::ErrorIncorrectNumberOfDimensionsInTensor ); return; }

        if( wTensor )
        {
            services::Collection<size_t> wDims = getWeightsSizes(algParameter);
            if (!data_management::checkTensor(wTensor.get(), this->_errors.get(), strWeights(), &wDims)) { return; }
        }

        if( bTensor )
        {
            services::Collection<size_t> bDims = getBiasesSizes(algParameter);
            if (!data_management::checkTensor(bTensor.get(), this->_errors.get(), strBiases(), &bDims)) { return; }
        }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONVOLUTION2D__RESULT"></a>
 * \brief Results obtained with the compute() method of the forward 2D convolution layer
 *        in the batch processing mode
 */
class Result : public layers::forward::Result
{
public:
    /**
     * Default constructor
     */
    Result() {}

    virtual ~Result() {}

    /**
     * Returns the result of the forward 2D convolution layer
     */
    using layers::forward::Result::get;

    /**
     * Sets the result of the forward 2D convolution layer
     */
    using layers::forward::Result::set;

    /**
     * Returns dimensions of value tensor
     * \return Dimensions of value tensor
     */
    virtual const services::Collection<size_t> getValueSize(const services::Collection<size_t> &inputSize,
                                                            const daal::algorithms::Parameter *parameter, const int method) const DAAL_C11_OVERRIDE
    {
        const Parameter *param =  static_cast<const Parameter * >(parameter);

        size_t c1 =
            (inputSize[param->spatialDimensions.size[0]] + 2 * param->padding.size[0] - param->kernelSize.size[0]) / param->stride.size[0] + 1;
        size_t c2 =
            (inputSize[param->spatialDimensions.size[1]] + 2 * param->padding.size[1] - param->kernelSize.size[1]) / param->stride.size[1] + 1;

        services::Collection<size_t> valueDims;
        for(size_t i = 0; i < inputSize.size(); i++)
        {
            if(i == param->spatialDimensions.size[0]) { valueDims.push_back(c1); }
            else if(i == param->spatialDimensions.size[1]) { valueDims.push_back(c2); }
            else if(i == param->groupDimension) { valueDims.push_back(param->nKernels); }
            else { valueDims.push_back( inputSize[i] ); }
        }

        return valueDims;
    }

    /**
     * Allocates memory to store the result of forward  2D convolution layer
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of forward 2D convolution layer
     * \param[in] method    Computation method for the layer
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        using namespace data_management;
        const Input *in = static_cast<const Input * >(input);

        const services::Collection<size_t> &inDims = in->get(layers::forward::data)->getDimensions();

        if (!get(layers::forward::value))
        {
            set(layers::forward::value, services::SharedPtr<Tensor>(
                          new HomogenTensor<algorithmFPType>(getValueSize(inDims, parameter, method), Tensor::doAllocate)));
        }

        set(layers::forward::resultForBackward, services::SharedPtr<LayerData>(new LayerData()));
        setResultForBackward(input);
    }

    /**
     * Sets the result that is used in backward 2D convolution layer
     * \param[in] input     Pointer to an object containing the input data
     */
    virtual void setResultForBackward(const daal::algorithms::Input *input) DAAL_C11_OVERRIDE
    {
        const Input *in = static_cast<const Input * >(input);
        set(auxData, in->get(layers::forward::data));
        set(auxWeights, in->get(layers::forward::weights));
    }

    /**
     * Returns the result of forward 2D convolution layer
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::Tensor> get(LayerDataId id) const
    {
        services::SharedPtr<layers::LayerData> layerData =
            services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
        if(!layerData)
        {
            this->_errors->add(services::ErrorNullLayerData);
            return services::SharedPtr<data_management::Tensor>();
        }
        return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
    }

    /**
     * Sets the result of forward 2D convolution layer
     * \param[in] id     Identifier of the result
     * \param[in] value  Result
     */
    void set(LayerDataId id, const services::SharedPtr<data_management::Tensor> &value)
    {
        services::SharedPtr<layers::LayerData> layerData =
            services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
        if(!layerData)
        {
            this->_errors->add(services::ErrorNullLayerData);
        }
        (*layerData)[id] = value;
    }

    /**
     * Checks the result of the forward 2D convolution layer
     * \param[in] input   %Input object of the layer
     * \param[in] par     %Parameter of the layer
     * \param[in] method  Computation method of the layer
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        layers::forward::Result::check(input, par, method);
        if( this->_errors->size() > 0 ) { return; }

        services::SharedPtr<services::Error> error;

        const Input     *algInput     = static_cast<const Input *>(input);
        const Parameter *algParameter = static_cast<const Parameter *>(par);

        services::SharedPtr<LayerData> layerData = get(layers::forward::resultForBackward);
        if (!layerData) { this->_errors->add(services::ErrorNullLayerData); return; }

        services::SharedPtr<data_management::Tensor> dataTensor  = algInput->get(layers::forward::data);
        services::SharedPtr<data_management::Tensor> valueTensor = get(layers::forward::value);

        const services::Collection<size_t> &dataDims = dataTensor->getDimensions();
        const services::Collection<size_t>     wDims = algInput->getWeightsSizes(algParameter);
        const services::Collection<size_t>   valDims = getValueSize(dataDims, algParameter, defaultDense);

        if (!data_management::checkTensor(valueTensor.get(), this->_errors.get(), strValue(), &valDims)) { return; }
        if (!data_management::checkTensor(get(auxData).get(), this->_errors.get(), strAuxData(), &dataDims)) { return; }
        if (!data_management::checkTensor(get(auxWeights).get(), this->_errors.get(), strAuxWeights(), &wDims)) { return; }
    }

    /**
     * Returns the serialization tag of the forward 2D convolution layer result
     * \return     Serialization tag of the forward 2D convolution layer result
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_LAYERS_CONVOLUTION2D_FORWARD_RESULT_ID; }

    /**
     * Serializes the object
     * \param[in]  arch  Storage for the serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
     * Deserializes the object
     * \param[in]  arch  Storage for the deserialized object or data structure
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
} // namespace convolution2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
