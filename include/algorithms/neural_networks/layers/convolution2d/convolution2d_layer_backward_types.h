/* file: convolution2d_layer_backward_types.h */
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
//  Implementation of backward two-dimensional (2D) convolution layer.
//--
*/

#ifndef __CONVOLUTION2D_LAYER_BACKWARD_TYPES_H__
#define __CONVOLUTION2D_LAYER_BACKWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_backward_types.h"

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
 * \brief Contains classes for backward 2D convolution layer
 */
namespace backward
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BACKWARD__INPUT"></a>
 * \brief %Input objects for the backward 2D convolution layer
 */
class Input : public layers::backward::Input
{
public:
    /**
     * Default constructor
     */
    Input() {};

    virtual ~Input() {}

    /**
     * Sets an input object for the backward 2D convolution layer
     */
    using layers::backward::Input::get;

    /**
     * Returns an input object for the backward 2D convolution layer
    */
    using layers::backward::Input::set;

    /**
     * Returns an input object for backward 2D convolution layer
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::Tensor> get(LayerDataId id) const
    {
        services::SharedPtr<layers::LayerData> layerData =
            services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::backward::inputFromForward));
        return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
    }

    /**
     * Sets input for the backward 2D convolution layer
     * \param[in] id    Identifier of the input  object
     * \param[in] value Input object to set
     */
    void set(LayerDataId id, const services::SharedPtr<data_management::Tensor> &value)
    {
        services::SharedPtr<layers::LayerData> layerData = get(layers::backward::inputFromForward);
        (*layerData)[id] = value;
    }

    /**
     * Checks an input object of the 2D convolution layer
     * \param[in] parameter %Parameter of layer
     * \param[in] method    Computation method of the layer
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        layers::backward::Input::check(parameter, method);
        if( this->_errors->size() > 0 ) { return; }

        const Parameter *param = static_cast<const Parameter *>(parameter);

        services::SharedPtr<data_management::Tensor> xTensor = get(auxData);

        if (!data_management::checkTensor(xTensor.get(), this->_errors.get(), strAuxData())) { return; }

        const services::Collection<size_t> &xDims = xTensor->getDimensions();
        const services::Collection<size_t> &gDims = get(layers::backward::inputGradient)->getDimensions();

        size_t c1 =
            (xDims[param->spatialDimensions.size[0]] + 2 * param->padding.size[0] - param->kernelSize.size[0]) / param->stride.size[0] + 1;
        size_t c2 =
            (xDims[param->spatialDimensions.size[1]] + 2 * param->padding.size[1] - param->kernelSize.size[1]) / param->stride.size[1] + 1;

        services::Collection<size_t> gradDims;
        for(size_t i = 0; i < xDims.size(); i++)
        {
            if(i == param->spatialDimensions.size[0]) { gradDims.push_back(c1); }
            else if(i == param->spatialDimensions.size[1]) { gradDims.push_back(c2); }
            else if(i == param->groupDimension) { gradDims.push_back(param->nKernels); }
            else { gradDims.push_back( xDims[i] ); }
        }

        services::Collection<size_t> wDims;
        wDims.push_back(param->nKernels);
        wDims.push_back(xDims[param->groupDimension]);
        wDims.push_back(param->kernelSize.size[0]);
        wDims.push_back(param->kernelSize.size[1]);

        if (!data_management::checkTensor(get(layers::backward::inputGradient).get(), this->_errors.get(), strInputGradient(), &gradDims)) { return; }
        if (!data_management::checkTensor(get(auxWeights).get(), this->_errors.get(), strAuxWeights(), &wDims)) { return; }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONVOLUTION2D__RESULT"></a>
 * \brief Results obtained with the compute() method of the backward 2D convolution layer
 */
class Result : public layers::backward::Result
{
public:
    /**
     * Default constructor
     */
    Result() : layers::backward::Result() {}

    virtual ~Result() {}

    /**
     * Returns the result of the backward 2D convolution layer
     */
    using layers::backward::Result::get;

    /**
     * Sets the result of the backward 2D convolution layer
     */
    using layers::backward::Result::set;

    /**
     * Allocates memory to store the result of backward 2D convolution layer
     * \param[in] input     Object containing the input data
     * \param[in] parameter %Parameter of backward 2D convolution layer
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        using daal::data_management::Tensor;
        using daal::data_management::HomogenTensor;

        const Input *in = static_cast<const Input *>(input);
        const Parameter *param =  static_cast<const Parameter * >(parameter);

        services::Collection<size_t> bDims;
        bDims.push_back(param->nKernels);

        services::SharedPtr<Tensor> valueTable = in->get(auxData);
        services::SharedPtr<Tensor> wTable     = in->get(auxWeights);

        if(valueTable == 0 || wTable == 0) { this->_errors->add(services::ErrorNullInputNumericTable); return; }

        if (!get(layers::backward::gradient))
        {
            set(layers::backward::gradient, services::SharedPtr<Tensor>(
                          new HomogenTensor<algorithmFPType>(valueTable->getDimensions(), Tensor::doAllocate)));
        }
        if (!get(layers::backward::weightDerivatives))
        {
            set(layers::backward::weightDerivatives, services::SharedPtr<Tensor>(
                              new HomogenTensor<algorithmFPType>(wTable->getDimensions(), Tensor::doAllocate)));
        }
        if (!get(layers::backward::biasDerivatives))
        {
            set(layers::backward::biasDerivatives, services::SharedPtr<Tensor>(
                              new HomogenTensor<algorithmFPType>(bDims, Tensor::doAllocate)));
        }
    }

    /**
     * Checks the result of the 2D convolution layer
     * \param[in] input   %Input object of the layer
     * \param[in] par     %Parameter of the layer
     * \param[in] method  Computation method of the layer
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        layers::backward::Result::check(input, par, method);
        if( this->_errors->size() > 0 ) { return; }

        const Input *algInput = static_cast<const Input *>(input);
        const Parameter *param = static_cast<const Parameter *>(par);

        if (!data_management::checkTensor(get(layers::backward::gradient).get(), this->_errors.get(), strGradient(),
                                          &(algInput->get(auxData)->getDimensions()))) { return; }
        if (!data_management::checkTensor(get(layers::backward::weightDerivatives).get(), this->_errors.get(), strWeightDerivatives(),
                                          &(algInput->get(auxWeights)->getDimensions()))) { return; }

        services::Collection<size_t> bDims;
        bDims.push_back(param->nKernels);

        if (!data_management::checkTensor(get(layers::backward::biasDerivatives).get(), this->_errors.get(), strBiasDerivatives(), &bDims)) { return; }
    }
};

} // namespace interface1
using interface1::Input;
using interface1::Result;
} // namespace backward
} // namespace convolution2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
