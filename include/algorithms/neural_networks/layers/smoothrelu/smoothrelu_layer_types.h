/* file: smoothrelu_layer_types.h */
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
//  Implementation of the smooth relu layer types.
//--
*/

#ifndef __NEURAL_NETWORKS__SMOOTHRELU_LAYER_TYPES_H__
#define __NEURAL_NETWORKS__SMOOTHRELU_LAYER_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace smoothrelu
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SMOOTHRELU__METHOD"></a>
 * \brief Computation methods for the smooth relu layer
 */
enum Method
{
    defaultDense = 0, /*!<  Default: performance-oriented method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SMOOTHRELU__LAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward smooth relu layer and results for the forward smooth relu layer
 */
enum LayerDataId
{
    auxData = 2 /*!< Data processed at the forward stage of the layer */
};

} // namespace smoothrelu
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
