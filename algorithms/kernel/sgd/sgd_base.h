/* file: sgd_base.h */
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

//++
//  Declaration of template function that calculate sgd.
//--


#ifndef __SGD_BASE_H__
#define __SGD_BASE_H__

#include "sgd_batch.h"
#include "kernel.h"
#include "numeric_table.h"
#include "service_rng.h"
#include "service_math.h"
#include "service_micro_table.h"

using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sgd
{
namespace internal
{
/**
 *  \brief Kernel for sgd calculation
 *  in case floating point type of intermediate calculations
 *  and method of calculations are different
 */
template<typename algorithmFPType, CpuType cpu>
class SGDBaseKernel : public Kernel
{
protected:
    algorithmFPType vectorNorm(algorithmFPType *vec, size_t nElements)
    {
        algorithmFPType norm = 0;
        for(size_t i = 0; i < nElements; i++)
        {
            norm += vec[i] * vec[i];
        }
        return sSqrt<cpu>(norm); // change to sqNorm
    }

    void getRandom(int minVal, int maxVal, int *randomValue, int nRandomValues, size_t seed)
    {
        daal::internal::IntRng <int, cpu> rng((int)seed);
        rng.uniform(nRandomValues, minVal, maxVal, randomValue);
    }

};

/**
 *  \brief Kernel for sgd calculation
 *  in case floating point type of intermediate calculations
 *  and method of calculations are different
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class SGDKernel : public SGDBaseKernel<algorithmFPType, cpu>
{};


} // namespace daal::internal
} // namespace sgd
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

#endif
