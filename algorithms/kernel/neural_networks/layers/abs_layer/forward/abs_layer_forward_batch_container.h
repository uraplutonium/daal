/* file: abs_layer_forward_batch_container.h */
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
//  Implementation of abs layer container.
//--
*/

#ifndef __ABS_LAYER_FORWARD_BATCH_CONTAINER_H__
#define __ABS_LAYER_FORWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/abs/abs_layer.h"
#include "abs_layer_forward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace abs
{
namespace forward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::AbsKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    abs::forward::Input *input = static_cast<abs::forward::Input *>(_in);
    abs::forward::Result *result = static_cast<abs::forward::Result *>(_res);

    daal::algorithms::Parameter *par = NULL;
    daal::services::Environment::env &env = *_env;

    Tensor *inputTensor  = input->get(layers::forward::data).get();
    Tensor *resultTensor = result->get(layers::forward::value).get();

    __DAAL_CALL_KERNEL(env, internal::AbsKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *inputTensor, *resultTensor);
}
} // namespace interface1
} // namespace forward

} // namespace abs
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
