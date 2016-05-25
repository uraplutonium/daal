/* file: linear_regression_group_of_betas_dense_default_batch_impl.i */
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
//  Declaration of template class that computes linear regression quality metrics.
//--
*/

#ifndef __LINEAR_REGRESSION_GROUP_OF_BETAS_DEFAULT_IMPL_I__
#define __LINEAR_REGRESSION_GROUP_OF_BETAS_DEFAULT_IMPL_I__

#include "service_memory.h"
#include "service_micro_table.h"
#include "service_lapack.h"
#include "threading.h"
#include "service_numeric_table.h"
#include "linear_regression_group_of_betas_dense_default_batch_kernel.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace quality_metric
{
namespace group_of_betas
{
namespace internal
{

template<Method method, typename algorithmFPType, CpuType cpu>
void GroupOfBetasKernel<method, algorithmFPType, cpu>::compute(
    const NumericTable* y, const NumericTable* z, const NumericTable* zReduced,
    size_t numBeta, size_t numBetaReduced, algorithmFPType accuracyThreshold,
    NumericTable* out[])
{
    const auto n = y->getNumberOfRows();
    const auto k = y->getNumberOfColumns();

    SmartPtr<cpu> aResSS0(k * sizeof(algorithmFPType));
    if(!aResSS0.get())
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed);
        return;
    }

    ReadRows<algorithmFPType, cpu, NumericTable> yBD(*y, n);
    ReadRows<algorithmFPType, cpu, NumericTable> zBD(*z, n);
    ReadRows<algorithmFPType, cpu, NumericTable> zReducedBD(*zReduced, n);

    WriteRows<algorithmFPType, cpu, NumericTable> ermBD(*out[expectedMeans], 1);
    WriteRows<algorithmFPType, cpu, NumericTable> resSSBD(*out[resSS], 1);

    const algorithmFPType divN = 1./algorithmFPType(n);
    //Compute ERM, resSS, resSS0
    {
        algorithmFPType *pErm = ermBD.get();
        algorithmFPType *pResSS = resSSBD.get();
        algorithmFPType *pResSS0 = (algorithmFPType *)aResSS0.get();

        for(size_t j = 0; j < k; pErm[j] = 0, pResSS[j] = 0, pResSS0[j] = 0, ++j);

        const algorithmFPType* pz = zBD.get();
        const algorithmFPType* py = yBD.get();
        const algorithmFPType* pz0 = zReducedBD.get();

        for(size_t i = 0; i < n; ++i)
        {
            for(size_t j = 0; j < k; ++j, ++py, ++pz, ++pz0)
            {
                pErm[j] += *py;
                pResSS[j] += (*py - *pz)*(*py - *pz);
                pResSS0[j] += (*py - *pz0)*(*py - *pz0);
            }
        }
        for(size_t j = 0; j < k; pErm[j] *= divN, ++j);
    }

    //Compute ERV, regSS, tSS
    {
        WriteRows<algorithmFPType, cpu, NumericTable> tSSBD(*out[tSS], 1);
        algorithmFPType *pTSS = tSSBD.get();

        WriteRows<algorithmFPType, cpu, NumericTable> regSSBD(*out[regSS], 1);
        algorithmFPType *pRegSS = regSSBD.get();
        for(size_t j = 0; j < k; pRegSS[j] = 0, pTSS[j] = 0, ++j);

        const algorithmFPType *pErm = ermBD.get();
        const algorithmFPType* py = yBD.get();
        const algorithmFPType* pz = zBD.get();
        for(size_t i = 0; i < n; ++i)
        {
            for(size_t j = 0; j < k; ++j, ++py, ++pz)
            {
                pTSS[j] += (*py - pErm[j])*(*py - pErm[j]);
                pRegSS[j] += (*pz - pErm[j])*(*pz - pErm[j]);
            }
        }

        WriteRows<algorithmFPType, cpu, NumericTable> ervBD(*out[expectedVariance], 1);
        algorithmFPType *pErv = ervBD.get();
        WriteRows<algorithmFPType, cpu, NumericTable> detBD(*out[determinationCoeff], 1);
        algorithmFPType *pDet = detBD.get();
        WriteRows<algorithmFPType, cpu, NumericTable> fBD(*out[fStatistics], 1);
        algorithmFPType *pF = fBD.get();

        const algorithmFPType *pResSS = resSSBD.get();
        const algorithmFPType *pResSS0 = (algorithmFPType *)aResSS0.get();
        const algorithmFPType divN_1 = 1./algorithmFPType(n - 1);
        const algorithmFPType multF = algorithmFPType(n - numBeta)/algorithmFPType(numBeta - numBetaReduced);
        for(size_t j = 0; j < k; ++j)
        {
            pErv[j] = pTSS[j]*divN_1;
            pRegSS[j] *= divN;
            pDet[j] = pRegSS[j]/pTSS[j];
            const algorithmFPType div = (pResSS[j] < accuracyThreshold ? accuracyThreshold : pResSS[j]);
            pF[j] = multF*(pResSS0[j] - pResSS[j])/div;
        }
    }
}

}
}
}
}
}
}

#endif
