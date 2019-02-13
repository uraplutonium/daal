/* file: average_pooling2d_layer.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of average pooling2d calculation algorithm and types methods.
//--
*/

#include "average_pooling2d_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace average_pooling2d
{
namespace interface1
{
/**
 * Constructs the parameters of average 2D pooling layer
 * \param[in] firstIndex        Index of the first of two dimensions on which the pooling is performed
 * \param[in] secondIndex       Index of the second of two dimensions on which the pooling is performed
 * \param[in] firstKernelSize   Size of the first dimension of 2D subtensor for which the average element is computed
 * \param[in] secondKernelSize  Size of the second dimension of 2D subtensor for which the average element is computed
 * \param[in] firstStride       Interval over the first dimension on which the pooling is performed
 * \param[in] secondStride      Interval over the second dimension on which the pooling is performed
 * \param[in] firstPadding      Number of data elements to implicitly add to the the first dimension
 *                              of the 2D subtensor on which the pooling is performed
 * \param[in] secondPadding     Number of data elements to implicitly add to the the second dimension
 *                              of the 2D subtensor on which the pooling is performed
 */
Parameter::Parameter(size_t firstIndex, size_t secondIndex, size_t firstKernelSize, size_t secondKernelSize,
          size_t firstStride, size_t secondStride, size_t firstPadding, size_t secondPadding) :
    layers::pooling2d::Parameter(firstIndex, secondIndex, firstKernelSize, secondKernelSize,
                                 firstStride, secondStride, firstPadding, secondPadding)
{}

}// namespace interface1
}// namespace average_pooling2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
