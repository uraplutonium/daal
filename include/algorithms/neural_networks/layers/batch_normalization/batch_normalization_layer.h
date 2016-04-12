/* file: batch_normalization_layer.h */
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
//  Implementation of batch normalization layer.
//--
*/

#ifndef __BATCH_NORMALIZATION_LAYER_H__
#define __BATCH_NORMALIZATION_LAYER_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/batch_normalization/batch_normalization_layer_types.h"
#include "algorithms/neural_networks/layers/batch_normalization/batch_normalization_layer_forward.h"
#include "algorithms/neural_networks/layers/batch_normalization/batch_normalization_layer_backward.h"


namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * \brief Contains classes for batch normalization layer
 */
namespace batch_normalization
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCH_NORMALIZATION__BATCH"></a>
 * \brief Provides methods for the batch normalization layer in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the batch normalization layer, double or float
 * \tparam method           Batch normalization layer computation method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method      Computation methods
 *      - \ref LayerDataId Identifiers of collection in results of forward and in input objects for the backward batch normalization layer
 *
 * \par References
 *      - <a href="DAAL-REF-BATCH_NORMALIZATIONFORWARD-ALGORITHM">Forward batch normalization layer description and usage models</a>
 *      - forward::Batch class
 *      - <a href="DAAL-REF-BATCH_NORMALIZATIONBACKWARD-ALGORITHM">Backward batch normalization layer description and usage models</a>
 *      - backward::Batch class
 *      - \ref interface1::Parameter "Parameter" class
 */
template<typename algorithmFPType = float, Method method = defaultDense>
class Batch : public LayerIface
{
public:
    /** Default constructor */
    Batch()
    {
        forward::Batch<algorithmFPType, method> *forwardLayerObject = new forward::Batch<algorithmFPType, method>();
        backward::Batch<algorithmFPType, method> *backwardLayerObject = new backward::Batch<algorithmFPType, method>();

        delete forwardLayerObject->parameter;
        forwardLayerObject->parameter = &parameter;

        delete backwardLayerObject->parameter;
        backwardLayerObject->parameter = &parameter;

        LayerIface::forwardLayer = services::SharedPtr<forward::Batch<algorithmFPType, method> >(forwardLayerObject);
        LayerIface::backwardLayer = services::SharedPtr<backward::Batch<algorithmFPType, method> >(backwardLayerObject);
    };

    Parameter parameter; /*!< Batch normalization layer parameters */

    /**
     * Returns the structure that contains parameters of the batch normalization layer
     * \return Structure that contains parameters of the batch normalization layer
     */
    virtual const layers::Parameter *cloneLayerParameter()
    {
        Parameter *par = new Parameter(parameter);

        services::SharedPtr<forward::Batch<algorithmFPType, method> > forwardLayerObject =
            services::staticPointerCast<forward::Batch<algorithmFPType, method>, layers::forward::LayerIface>(LayerIface::forwardLayer);

        services::SharedPtr<backward::Batch<algorithmFPType, method> > backwardLayerObject =
            services::staticPointerCast<backward::Batch<algorithmFPType, method>, layers::backward::LayerIface>(LayerIface::backwardLayer);

        forwardLayerObject->parameter = par;
        backwardLayerObject->parameter = par;

        return par;
    };
};
} // namespace interface1
using interface1::Batch;

} // namespace batch_normalization
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
