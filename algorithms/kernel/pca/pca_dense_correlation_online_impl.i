/* file: pca_dense_correlation_online_impl.i */
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
//  Functuons that are used in PCA algorithm
//--
*/

#ifndef __PCA_DENSE_CORRELATION_ONLINE_IMPL_I__
#define __PCA_DENSE_CORRELATION_ONLINE_IMPL_I__

#include "service_math.h"
#include "service_memory.h"
#include "service_numeric_table.h"
#include "pca_dense_correlation_online_kernel.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{

template <typename algorithmFPType, CpuType cpu>
void PCACorrelationKernel<online, algorithmFPType, cpu>::compute(const services::SharedPtr<data_management::NumericTable> data,
                                                                 PartialResult<correlationDense> *partialResult,
                                                                 OnlineParameter<algorithmFPType, correlationDense> *parameter)
{
    parameter->covariance->input.set(covariance::data, data);
    parameter->covariance->parameter.outputMatrixType = covariance::correlationMatrix;

    parameter->covariance->compute();
    this->_errors->add(parameter->covariance->getErrors()->getErrors());
}

template <typename algorithmFPType, CpuType cpu>
void PCACorrelationKernel<online, algorithmFPType, cpu>::finalize(PartialResult<correlationDense> *partialResult,
                                                                  OnlineParameter<algorithmFPType, correlationDense> *parameter,
                                                                  services::SharedPtr<data_management::NumericTable> eigenvectors,
                                                                  services::SharedPtr<data_management::NumericTable> eigenvalues)
{
    parameter->covariance->finalizeCompute();
    this->_errors->add(parameter->covariance->getErrors()->getErrors());

    services::SharedPtr<data_management::NumericTable> correlation = parameter->covariance->getResult()->get(covariance::covariance);

    this->computeCorrelationEigenvalues(correlation, eigenvectors, eigenvalues);
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
