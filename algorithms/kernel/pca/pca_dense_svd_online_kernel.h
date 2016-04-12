/* file: pca_dense_svd_online_kernel.h */
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
//  Declaration of template structs that calculate PCA SVD.
//--
*/

#ifndef __PCA_DENSE_SVD_ONLINE_KERNEL_H__
#define __PCA_DENSE_SVD_ONLINE_KERNEL_H__

#include "pca_online.h"
#include "pca_types.h"
#include "svd/svd_dense_default_kernel.h"

#include "pca_dense_svd_base.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{

template <typename interm, CpuType cpu>
class PCASVDOnlineKernel : public PCASVDKernelBase<interm, cpu>
{
public:
    PCASVDOnlineKernel() {};

    void compute(const services::SharedPtr<data_management::NumericTable> &data,
                 services::SharedPtr<data_management::NumericTable> &nObservations,
                 services::SharedPtr<data_management::NumericTable> &auxiliaryTable,
                 services::SharedPtr<data_management::NumericTable> &sumSVD,
                 services::SharedPtr<data_management::NumericTable> &sumSquaresSVD);

    void finalizeMerge(const services::SharedPtr<data_management::NumericTable> &nObservationsTable,
                       services::SharedPtr<data_management::NumericTable> &eigenvalues,
                       services::SharedPtr<data_management::NumericTable> &eigenvectors,
                       services::SharedPtr<data_management::DataCollection> &rTables);

    ~PCASVDOnlineKernel() {}

protected:
    void normalizeDataset();

    void decompose();

    services::SharedPtr<data_management::NumericTable> _data;
    services::SharedPtr<data_management::NumericTable> _auxiliaryTable;
    services::SharedPtr<data_management::NumericTable> _sumSquaresSVD;
    services::SharedPtr<data_management::NumericTable> _sumSVD;

    services::SharedPtr<data_management::NumericTable> _normalizedData;

    size_t _nObservations;
    size_t _nOldObservations;
    size_t _nFeatures;
};

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal
#endif
