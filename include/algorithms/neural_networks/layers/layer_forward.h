/* file: layer_forward.h */
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
//  Implementation of neural network layer.
//--
*/

#ifndef __NEURAL_NETWORK_LAYER_FORWARD_H__
#define __NEURAL_NETWORK_LAYER_FORWARD_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_forward_types.h"

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
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LAYERIFACE"></a>
 *  \brief Class representing a layer of neural network
 */
class LayerIface : public daal::algorithms::Analysis<batch>
{
public:
    virtual ~LayerIface() {};

    /**
     * Returns the structure that contains results of the layer
     * \return Structure that contains results of the layer
     */
    virtual services::SharedPtr<Result> getLayerResult() = 0;

    /**
     * Returns the structure that contains input objects of the layer
     * \return Structure that contains input objects of the layer
     */
    virtual Input *getLayerInput() = 0;

    /**
     * Returns the structure that contains parameters of the layer
     * \return Structure that contains parameters of the layer
     */
    virtual Parameter *getLayerParameter() = 0;

    /**
     * Returns a pointer to the newly allocated forward neural network layer with a copy of input objects
     * and parameters of this layer
     * \return Pointer to the newly allocated forward layer
     */
    services::SharedPtr<LayerIface> clone() const
    {
        return services::SharedPtr<LayerIface>(cloneImpl());
    }

    /**
     * Allocates memory buffers needed for the computations
     */
    virtual void allocateResult() DAAL_C11_OVERRIDE = 0;

    /**
     * Allocates memory buffers needed for the computations
     */
    virtual void allocateInput() DAAL_C11_OVERRIDE {}

    /**
     * Initializes values of weights and biases if needed
     */
    virtual void initializeInput()
    {
        services::SharedPtr<data_management::Tensor> tensor;

        Parameter *param = getLayerParameter();

        if( !param ) { return; }

        tensor = getLayerInput()->get(weights);
        if( tensor && tensor->getDimensions().size() )
        {
            param->weightsInitializer->input.set(initializers::data, tensor);
            param->weightsInitializer->compute();
        }

        tensor = getLayerInput()->get(biases);
        if( tensor && tensor->getDimensions().size() )
        {
            param->biasesInitializer->input.set(initializers::data, tensor);
            param->biasesInitializer->compute();
        }
    }

protected:
    virtual LayerIface *cloneImpl() const DAAL_C11_OVERRIDE = 0;
};

} // namespace interface1
using interface1::LayerIface;
} // namespace forward
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
