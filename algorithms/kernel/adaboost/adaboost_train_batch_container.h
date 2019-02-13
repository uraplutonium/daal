/* file: adaboost_train_batch_container.h */
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
//  Implementation of Ada Boost algorithm container -- a class that contains
//  Freund Ada Boost kernels for supported architectures.
//--
*/

#ifndef __ADABOOST_TRAIN_BATCH_CONTAINER_H__
#define __ADABOOST_TRAIN_BATCH_CONTAINER_H__

#include "adaboost_training_batch.h"
#include "adaboost_train_kernel.h"
#include "kernel.h"

namespace daal
{
namespace algorithms
{
namespace adaboost
{
namespace training
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::AdaBoostTrainKernel, method, algorithmFPType);
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
    adaboost::training::Result *result = static_cast<adaboost::training::Result *>(_res);

    size_t n = input->size();

    NumericTablePtr a[2];
    a[0] = services::staticPointerCast<NumericTable>(input->get(classifier::training::data));
    a[1] = services::staticPointerCast<NumericTable>(input->get(classifier::training::labels));

    adaboost::Model *r = static_cast<adaboost::Model *>(result->get(classifier::training::model).get());
    adaboost::Parameter *par = static_cast<adaboost::Parameter *>(_par);

    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::AdaBoostTrainKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, n, a, r, par);
}

} // namespace daal::algorithms::adaboost::training
}
}
} // namespace daal

#endif // __ADABOOST_TRAINING_BATCH_CONTAINER_H__
