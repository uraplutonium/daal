/* file: covariance_dense_batch_impl.i */
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
//  Covariance matrix computation algorithm implementation in batch mode
//--
*/

#ifndef __COVARIANCE_CSR_BATCH_IMPL_I__
#define __COVARIANCE_CSR_BATCH_IMPL_I__

#include "covariance_kernel.h"
#include "covariance_impl.i"

#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void CovarianceDenseBatchKernel<algorithmFPType, method, cpu>::compute(
            SharedPtr<NumericTable> &dataTable, SharedPtr<NumericTable> &covTable,
            SharedPtr<NumericTable> &meanTable, const Parameter *parameter)
{
    algorithmFPType nObservationsValue = 0.0;
    SharedPtr<NumericTable> nObservationsTable(
        new daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>(&nObservationsValue, 1, 1));
    bool isOnline = false;
    updateDensePartialResults<algorithmFPType, method, cpu>(dataTable,
        covTable, meanTable, nObservationsTable, isOnline, this->_errors);
    finalizeCovariance<algorithmFPType, cpu>(covTable, meanTable, nObservationsTable, parameter, this->_errors);
}

}
}
}
}

#endif
