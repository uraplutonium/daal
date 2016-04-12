/* file: layer.h */
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

#ifndef __NEURAL_NETWORK_LAYER_H__
#define __NEURAL_NETWORK_LAYER_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_types.h"
#include "algorithms/neural_networks/layers/layer_forward.h"
#include "algorithms/neural_networks/layers/layer_backward.h"

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
namespace interface1
{
/**
* <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LAYERIFACE"></a>
* \brief Abstract class that specifies the interface of layer
*/
class LayerIface
{
public:
    services::SharedPtr<forward::LayerIface> forwardLayer;   /*!< Forward stage of the layer algorithm */
    services::SharedPtr<backward::LayerIface> backwardLayer; /*!< Backward stage of the layer algorithm */

    virtual ~LayerIface() {};

    /**
     * Returns the structure that contains parameters of the layer
     * \return Structure that contains parameters of the layer
     */
    virtual const layers::Parameter *cloneLayerParameter() = 0;
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LAYERDESCRIPTOR"></a>
* \brief Class defining descriptor for layer on both forward and backward stages and its parameters
*/
class LayerDescriptor
{
public:
    /** \brief Constructor */
    LayerDescriptor() {};

    /** \brief Constructor */
    LayerDescriptor(const size_t index_, const services::SharedPtr<LayerIface> &layer_, const NextLayers &nextLayers_):
        index(index_), layer(layer_), nextLayers(nextLayers_) {};

    size_t index; /*!< Index of the layer in the network */
    services::SharedPtr<LayerIface> layer; /*!< Layer algorithm */
    NextLayers nextLayers; /*!< Layers following the current layer in the network */
};

} // interface1
using interface1::LayerIface;
using interface1::LayerDescriptor;
using interface1::Parameter;
using interface1::NextLayers;

} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
