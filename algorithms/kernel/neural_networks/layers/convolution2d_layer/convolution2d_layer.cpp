/* file: convolution2d_layer.cpp */
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
//  Implementation of convolution2d calculation algorithm and types methods.
//--
*/

#include "convolution2d_layer_types.h"

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
namespace interface1
{
/**
 *  Default constructor
 */
Parameter::Parameter() : groupDimension(1), indices(2, 3), kernelSizes(2, 2), strides(2, 2), paddings(0, 0), nKernels(1), nGroups(1) {};

}// namespace interface1
}// namespace convolution2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
