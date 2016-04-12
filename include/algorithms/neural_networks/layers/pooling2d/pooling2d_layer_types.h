/* file: pooling2d_layer_types.h */
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
//  Implementation of 2D pooling layer.
//--
*/

#ifndef __POOLING2D_LAYER_TYPES_H__
#define __POOLING2D_LAYER_TYPES_H__

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
/**
 * \brief Contains classes for the two-dimensional (2D) pooling layer
 */
namespace pooling2d
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * \brief Data structure representing the size of the 2D subtensor
 *        from which the element is computed
 */
struct KernelSize
{
    /**
     * Constructs the structure representing the size of the 2D subtensor
     * from which the element is computed
     * \param[in]  first  Size of the first dimension of the 2D subtensor
     * \param[in]  second Size of the second dimension of the 2D subtensor
     */
    KernelSize(size_t first = 2, size_t second = 2) { size[0] = first; size[1] = second; }
    size_t size[2];
};

/**
 * \brief Data structure representing the intervals on which the subtensors for pooling are computed
 */
struct Stride
{
    /**
     * Constructs the structure representing the intervals on which the subtensors for pooling are computed
     * \param[in]  first  Interval over the first dimension on which the pooling is performed
     * \param[in]  second Interval over the second dimension on which the pooling is performed
     */
    Stride(size_t first = 2, size_t second = 2) { size[0] = first; size[1] = second; }
    size_t size[2];
};

/**
 * \brief Data structure representing the number of data elements to implicitly add
 *        to each side of the 2D subtensor on which pooling is performed
 */
struct Padding
{
    /**
     * Constructs the structure representing the number of data elements to implicitly add
     * to each side of the 2D subtensor on which pooling is performed
     * \param[in]  first  Number of data elements to add to the the first dimension of the 2D subtensor
     * \param[in]  second Number of data elements to add to the the second dimension of the 2D subtensor
     */
    Padding(size_t first = 2, size_t second = 2) { size[0] = first; size[1] = second; }
    size_t size[2];
};

/**
 * \brief Data structure representing the indices of the two dimensions on which pooling is performed
 */
struct SpatialDimensions
{
    /**
     * Constructs the structure representing the indices of the two dimensions on which pooling is performed
     * \param[in]  first  The first dimension index
     * \param[in]  second The second dimension index
     */
    SpatialDimensions(size_t first = 2, size_t second = 3) { size[0] = first; size[1] = second; }
    size_t size[2];
};

/**
 * \brief Parameters for the forward and backward two-dimensional pooling layers
 *
 * \snippet neural_networks/layers/pooling2d/pooling2d_layer_types.h Parameter source code
 */
/* [Parameter source code] */
struct Parameter: public layers::Parameter
{
    /**
     * Constructs the parameters of 2D pooling layer
     * \param[in] firstIndex        Index of the first of two dimensions on which the pooling is performed
     * \param[in] secondIndex       Index of the second of two dimensions on which the pooling is performed
     * \param[in] firstKernelSize   Size of the first dimension of 2D subtensor for which the element is computed
     * \param[in] secondKernelSize  Size of the second dimension of 2D subtensor for which the element is computed
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
        indices(firstIndex, secondIndex), kernelSize(firstKernelSize, secondKernelSize),
        stride(firstStride, secondStride), padding(firstPadding, secondPadding), predictionStage(predictionStage)
    {}

    Stride stride;              /*!< Data structure representing the intervals on which the subtensors for pooling are computed */
    Padding padding;            /*!< Data structure representing the number of data elements to implicitly add
                                     to each size of the 2D subtensor on which pooling is performed */
    KernelSize kernelSize;      /*!< Data structure representing the size of the 2D subtensor
                                     from which the element is computed */
    SpatialDimensions indices;  /*!< Indices of the two dimensions on which pooling is performed */
    bool predictionStage;       /*!< Flag that specifies whether the layer is used at the prediction stage or not */
};
/* [Parameter source code] */

} // interface1
using interface1::Parameter;
using interface1::KernelSize;
using interface1::Stride;
using interface1::Padding;
using interface1::SpatialDimensions;

} // namespace pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal

#endif
