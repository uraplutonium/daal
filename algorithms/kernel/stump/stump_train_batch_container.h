/* file: stump_train_batch_container.h */
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
//  Implementation of Decision Stump algorithm container -- a class that contains
//  Friedman Decision Stump kernels for supported architectures.
//--
*/

#ifndef __STUMP_TRAIN_BATCH_CONTAINER_H__
#define __STUMP_TRAIN_BATCH_CONTAINER_H__

#include "stump_training_batch.h"
#include "stump_train_kernel.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace training
{

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::StumpTrainKernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    classifier::training::Input *input = static_cast<classifier::training::Input *>(_in);
    stump::training::Result *result = static_cast<stump::training::Result *>(_res);
    size_t n = input->size();
    NumericTable *a[3];
    a[0] = static_cast<NumericTable *>(input->get(classifier::training::data).get());
    a[1] = static_cast<NumericTable *>(input->get(classifier::training::labels).get());
    a[2] = static_cast<NumericTable *>(input->get(classifier::training::weights).get());
    stump::Model *r = static_cast<stump::Model *>(result->get(classifier::training::model).get());

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::StumpTrainKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, n, a, r, NULL);
}

} // namespace daal::algorithm::stump::training
}
}
} // namespace daal

#endif
