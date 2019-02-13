/* file: transposed_conv2d_layer_backward_dense_default_batch_fpt_cpu.cpp */
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

//++
//  Implementation of transposed convolution 2d calculation functions.
//--


#include "transposed_conv2d_layer_backward_batch_container.h"
#include "transposed_conv2d_layer_backward_kernel.h"
#include "transposed_conv2d_layer_backward_impl.i"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace transposed_conv2d
{

namespace backward
{
namespace interface1
{
template class neural_networks::layers::transposed_conv2d::backward::BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // interface1
namespace internal
{
template class TransposedConv2dKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // internal
} // backward

}
}
}
}
}
