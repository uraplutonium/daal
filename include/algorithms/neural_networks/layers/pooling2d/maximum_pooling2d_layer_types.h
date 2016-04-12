/* file: maximum_pooling2d_layer_types.h */
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
//  Implementation of maximum 2D pooling layer.
//--
*/

#ifndef __MAXIMUM_POOLING2D_LAYER_TYPES_H__
#define __MAXIMUM_POOLING2D_LAYER_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/pooling2d/pooling2d_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace maximum_pooling2d
{
/**
 * \brief Computation methods for the maximum 2D pooling layer
 */
enum Method
{
    defaultDense = 0    /*!< Default: performance oriented method */
};

/**
 * \brief Identifiers of input objects for the backward maximum 2D pooling layer
 *        and results for the forward maximum 2D pooling layer
 */
enum LayerDataId
{
    auxMask = 0         /*!< p-dimensional tensor that stores forward maximum pooling layer results */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * \brief Parameters for the maximum 2D pooling layer
 *
 * \snippet neural_networks/layers/pooling2d/maximum_pooling2d_layer_types.h Parameter source code
 */
/* [Parameter source code] */
struct Parameter: public pooling2d::Parameter
{
    /**
     * Constructs the parameters of 2D pooling layer
     * \param[in] firstIndex        Index of the first of two dimensions on which the pooling is performed
     * \param[in] secondIndex       Index of the second of two dimensions on which the pooling is performed
     * \param[in] firstKernelSize   Size of the first dimension of 2D subtensor for which the maximum element is selected
     * \param[in] secondKernelSize  Size of the second dimension of 2D subtensor for which the maximum element is selected
     * \param[in] firstStride       Interval over the first dimension on which the pooling is performed
     * \param[in] secondStride      Interval over the second dimension on which the pooling is performed
     * \param[in] firstPadding      Number of data elements to implicitly add to the the first dimension
     *                              of the 2D subtensor on which the pooling is performed
     * \param[in] secondPadding     Number of data elements to implicitly add to the the second dimension
     *                              of the 2D subtensor on which the pooling is performed
     * \param[in] predictionStage   Flag that specifies whether the layer is used for the prediction stage or not
     */
    Parameter(size_t firstIndex, size_t secondIndex, size_t firstKernelSize = 2, size_t secondKernelSize = 2,
              size_t firstStride = 2, size_t secondStride = 2, size_t firstPadding = 2, size_t secondPadding = 2,
              bool predictionStage = false) :
        layers::pooling2d::Parameter(firstIndex, secondIndex, firstKernelSize, secondKernelSize,
                                     firstStride, secondStride, firstPadding, secondPadding)
    {}
};
/* [Parameter source code] */

} // interface1
using interface1::Parameter;

} // namespace maximum_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal

#endif
