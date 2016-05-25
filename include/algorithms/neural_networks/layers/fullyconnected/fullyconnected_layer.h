/* file: fullyconnected_layer.h */
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
//  Implementation of fully-connected neural network layer.
//--
*/

#ifndef __NEURAL_NETWORK_FULLYCONNECTED_LAYER_H__
#define __NEURAL_NETWORK_FULLYCONNECTED_LAYER_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/fullyconnected/fullyconnected_layer_types.h"
#include "algorithms/neural_networks/layers/fullyconnected/fullyconnected_layer_forward.h"
#include "algorithms/neural_networks/layers/fullyconnected/fullyconnected_layer_backward.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * \brief Contains classes for neural network fully-connected layer
 */
namespace fullyconnected
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FULLYCONNECTED__BATCH"></a>
 * \brief Computes the result of the forward and backward fully-connected layer of neural network in the batch processing mode
 *
 * \tparam algorithmFPType Data type to use in intermediate computations for the fully-connected layer, double or float
 * \tparam method          Batch fully-connected layer computation method, \ref Method
 *
 * \par References
 *      - \ref interface1::Parameter "Parameter" class
 *      - <a href="DAAL-REF-FULLYCONNECTEDFORWARD-ALGORITHM">Forward fully-connected layer description and usage models</a>
 *      - \ref forward::interface1::Batch  "forward::Batch" class
 *      - <a href="DAAL-REF-FULLYCONNECTEDBACKWARD-ALGORITHM">Backward fully-connected layer description and usage models</a>
 *      - \ref backward::interface1::Batch "backward::Batch" class
 */
template<typename algorithmFPType = float, Method method = defaultDense>
class Batch : public LayerIface
{
public:
    Parameter parameter; /*!< %Parameters of the layer */
    /**
     *  \param[in] nOutputs A number of layer outputs
     *  \param[in] dim      Dimension for which forward propagation step of the fully-connected layer is performed
     */
    Batch(size_t nOutputs, size_t dim = 0): parameter(nOutputs, dim)
    {
        forward::Batch<algorithmFPType, method> *forwardLayerObject = new forward::Batch<algorithmFPType, method>(parameter);
        backward::Batch<algorithmFPType, method> *backwardLayerObject = new backward::Batch<algorithmFPType, method>(parameter);

        LayerIface::forwardLayer = services::SharedPtr<forward::Batch<algorithmFPType, method> >(forwardLayerObject);
        LayerIface::backwardLayer = services::SharedPtr<backward::Batch<algorithmFPType, method> >(backwardLayerObject);
    };
};
} // namespace interface1
using interface1::Batch;

} // namespace fullyconnected
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
