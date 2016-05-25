/* file: pca_dense_svd_batch_impl.i */
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

#ifndef __PCA_DENSE_SVD_BATCH_IMPL_I__
#define __PCA_DENSE_SVD_BATCH_IMPL_I__

#include "service_math.h"
#include "service_memory.h"
#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{

template <typename interm, CpuType cpu>
void PCASVDBatchKernel<interm, cpu>::compute(const services::SharedPtr<data_management::NumericTable> &data,
                                             services::SharedPtr<data_management::NumericTable> &eigenvalues,
                                             services::SharedPtr<data_management::NumericTable> &eigenvectors)
{
    _data = data;
    _eigenvalues = eigenvalues;
    _eigenvectors = eigenvectors;

    _nObservations = _data->getNumberOfRows();
    _nFeatures = _data->getNumberOfColumns();

    normalizeDataset();

    if (this->_errors->size() != 0) { return; }

    decompose();

    if (this->_errors->size() != 0) { return; }

    this->scaleSingularValues(_eigenvalues.get(), _nObservations);
}

namespace
{

template<typename interm>
void extractMean(size_t nFeatures, size_t nVectors, interm *data, interm *sums, interm *normalizedData)
{
    for (size_t i = 0; i < nVectors; i++)
    {
        for (size_t j = 0; j < nFeatures; j++)
        {
            normalizedData[i * nFeatures + j] = data[i * nFeatures + j] - sums[j] / nVectors;
        }
    }
}

template<typename interm, CpuType cpu>
void diviseByVariance(size_t nFeatures, size_t nVectors, interm *ssq, interm *normalizedData)
{
    for (size_t i = 0; i < nVectors; i++)
    {
        for (size_t j = 0; j < nFeatures; j++)
        {
            if (ssq[j] != 0)
            {
                normalizedData[i * nFeatures + j] /= sSqrt<cpu>(ssq[j] / (nVectors - 1));
            }
        }
    }
}

template <typename interm>
void setToZero(const size_t size, interm *data)
{
    for (size_t i = 0; i < size; i++)
    {
        data[i] = 0;
    }
}

template<typename interm>
void computeSums(const size_t nFeatures, const size_t nVectors, const interm *data, interm *sums)
{
    for(size_t i = 0; i < nVectors; i++)
    {
        for(size_t j = 0; j < nFeatures; j++)
        {
            sums[j] += data[i * nFeatures + j];
        }
    }
}

template <typename interm>
void computeSumsOfSquares(const size_t nFeatures, const size_t nVectors, const interm *normalizedData, interm *ssq)
{
    for(size_t i = 0; i < nVectors; i++)
    {
        for(size_t j = 0; j < nFeatures; j++)
        {
            ssq[j] += normalizedData[i * nFeatures + j] * normalizedData[i * nFeatures + j];
        }
    }
}

}

template <typename interm, CpuType cpu>
void PCASVDBatchKernel<interm, cpu>::normalizeDataset()
{
    using data_management::NumericTable;
    using data_management::HomogenNumericTable;
    using daal::internal::HomogenNumericTableCPU;

    if(this->_type == normalizedDataset)
    {
        _normalizedData = _data;
        return;
    }

    BlockDescriptor<interm> block;
    _data->getBlockOfRows(0, _nObservations, data_management::readOnly, block);
    interm *dataArray = block.getBlockPtr();

    HomogenNumericTableCPU<interm, cpu> *normalized = new HomogenNumericTableCPU<interm, cpu>(_nFeatures, _nObservations);
    normalized->assign(0);

    _normalizedData = services::SharedPtr<HomogenNumericTable<interm> >(normalized);

    BlockDescriptor<interm> normalizedBlock;
    _normalizedData->getBlockOfRows(0, _nObservations, data_management::readOnly, normalizedBlock);
    interm *normalizedDataArray = normalizedBlock.getBlockPtr();

    interm *sums = (interm *)daal::services::daal_malloc(_nFeatures * sizeof(interm));
    interm *ssq = (interm *)daal::services::daal_malloc(_nFeatures * sizeof(interm));

    setToZero(_nFeatures, sums);
    setToZero(_nFeatures, ssq);

    computeSums<interm>(_nFeatures, _nObservations, dataArray, sums);

    extractMean<interm>(_nFeatures, _nObservations, dataArray, sums, normalizedDataArray);

    computeSumsOfSquares<interm>(_nFeatures, _nObservations, normalizedDataArray, ssq);

    diviseByVariance<interm, cpu>(_nFeatures, _nObservations, ssq, normalizedDataArray);

    daal::services::daal_free(ssq);
    daal::services::daal_free(sums);

    _data->releaseBlockOfRows(block);
    _normalizedData->releaseBlockOfRows(normalizedBlock);
}

template <typename interm, CpuType cpu>
void PCASVDBatchKernel<interm, cpu>::decompose()
{
    const NumericTable *normalizedDataTable = _normalizedData.get();
    const NumericTable *const *svdInputs = &normalizedDataTable;

    NumericTable *svdResults[3];
    svdResults[0] = _eigenvalues.get();
    svdResults[1] = 0;
    svdResults[2] = _eigenvectors.get();

    svd::Parameter params;
    params.leftSingularMatrix = svd::notRequired;

    daal::algorithms::svd::internal::SVDBatchKernel<interm, svd::defaultDense, cpu> svdKernel;
    svdKernel.compute(1, svdInputs, 3, svdResults, &params);

    if(svdKernel.getErrorCollection()->size() > 0) { this->_errors->add(svdKernel.getErrorCollection()); }
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
