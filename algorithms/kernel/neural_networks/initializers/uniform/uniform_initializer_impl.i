/* file: uniform_initializer_impl.i */
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
//  Implementation of uniform algorithm
//--
*/

#include "service_rng.h"

#ifndef __UNIFORM_INITIALIZER_IMPL_I__
#define __UNIFORM_INITIALIZER_IMPL_I__

using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace uniform
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void UniformKernel<algorithmFPType, method, cpu>::compute(const initializers::Input *input,
    const uniform::Parameter *parameter, initializers::Result *result)
{
    IntRng<int,cpu> rng(parameter->seed);

    SharedPtr<Tensor> resultTable  = result->get(initializers::value);

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTable->getSubtensor(0, 0, 0, resultTable->getDimensions()[0], writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    size_t size = resultBlock.getSize();

    int* buff = (int*)services::daal_malloc(sizeof(size_t) * size);
    if(!buff) {this->_errors->add(services::ErrorMemoryAllocationFailed); return;}

    int max_int = 2147483647;
    algorithmFPType a = (algorithmFPType)(parameter->a);
    algorithmFPType b = (algorithmFPType)(parameter->b);
    algorithmFPType mul = 1.0/max_int*(b-a);

    rng.uniform(size, 0, max_int, buff);

    for(size_t i=0; i<size; i++)
    {
        resultArray[i] = buff[i]*mul+a;
    }

    services::daal_free(buff);
    resultTable->releaseSubtensor(resultBlock);
}

} // internal
} // namespace uniform
} // namespace initializers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
