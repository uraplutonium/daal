/* file: covariance_impl.i */
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
//  Covariance matrix computation algorithm implementation
//--
*/

#ifndef __COVARIANCE_IMPL_I__
#define __COVARIANCE_IMPL_I__

#include "numeric_table.h"
#include "csr_numeric_table.h"
#include "service_memory.h"
#include "service_math.h"
#include "service_blas.h"
#include "service_spblas.h"
#include "service_stat.h"
#include "threading.h"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu>
void getTableData(ReadWriteMode rwMode, SharedPtr<NumericTable> &numericTable,
            BlockDescriptor<algorithmFPType> &bd, algorithmFPType **dataArray)
{
    size_t nRows = numericTable->getNumberOfRows();
    numericTable->getBlockOfRows(0, nRows, rwMode, bd);
    *dataArray = bd.getBlockPtr();
}

template<typename algorithmFPType, CpuType cpu>
void getCSRTableData(size_t nRows, ReadWriteMode rwMode, SharedPtr<CSRNumericTableIface> &numericTable,
            CSRBlockDescriptor<algorithmFPType> &bd, algorithmFPType **dataArray,
            size_t **colIndices, size_t **rowOffsets)
{
    numericTable->getSparseBlock(0, nRows, rwMode, bd);
    *dataArray = bd.getBlockValuesPtr();
    *colIndices = bd.getBlockColumnIndicesPtr();
    *rowOffsets = bd.getBlockRowIndicesPtr();
}

template<typename algorithmFPType, CpuType cpu>
void getDenseCrossProductAndSums(ReadWriteMode rwMode,
            SharedPtr<NumericTable> &covTable, BlockDescriptor<algorithmFPType> &crossProductBD,
            algorithmFPType **crossProduct,
            SharedPtr<NumericTable> &meanTable, BlockDescriptor<algorithmFPType> &sumBD,
            algorithmFPType **sums,
            SharedPtr<NumericTable> &nObservationsTable, BlockDescriptor<algorithmFPType> &nObservationsBD,
            algorithmFPType **nObservations)
{
    getTableData<algorithmFPType, cpu>(rwMode, covTable, crossProductBD, crossProduct);
    getTableData<algorithmFPType, cpu>(rwMode, meanTable, sumBD, sums);
    getTableData<algorithmFPType, cpu>(rwMode, nObservationsTable, nObservationsBD, nObservations);
}

template<typename algorithmFPType, Method method, CpuType cpu>
void getDenseCrossProductAndSums(size_t nFeatures, ReadWriteMode rwMode,
            SharedPtr<NumericTable> &covTable, BlockDescriptor<algorithmFPType> &crossProductBD,
            algorithmFPType **crossProduct,
            SharedPtr<NumericTable> &meanTable, BlockDescriptor<algorithmFPType> &sumBD,
            algorithmFPType **sums,
            SharedPtr<NumericTable> &nObservationsTable, BlockDescriptor<algorithmFPType> &nObservationsBD,
            algorithmFPType **nObservations,
            services::SharedPtr<NumericTable> &dataTable,
            services::SharedPtr<services::KernelErrorCollection> &_errors)
{
    getDenseCrossProductAndSums<algorithmFPType, cpu>(rwMode,
        covTable, crossProductBD, crossProduct, meanTable, sumBD, sums,
        nObservationsTable, nObservationsBD, nObservations);

    if (method == sumDense || method == sumCSR)
    {
        services::SharedPtr<NumericTable> userSumsTable = dataTable->basicStatistics.get(NumericTable::sum);
        if (!userSumsTable) // move to interface check
        { _errors->add(services::ErrorPrecomputedSumNotAvailable); return; }

        BlockDescriptor<algorithmFPType> userSumsBD;
        userSumsTable->getBlockOfRows(0, 1, readOnly, userSumsBD);
        algorithmFPType *userSums = userSumsBD.getBlockPtr();

        daal_memcpy_s(*sums, nFeatures * sizeof(algorithmFPType), userSums, nFeatures * sizeof(algorithmFPType));

        userSumsTable->releaseBlockOfRows(userSumsBD);
    }
}

template<typename algorithmFPType, CpuType cpu>
void releaseDenseCrossProductAndSums(
            SharedPtr<NumericTable> &covTable, BlockDescriptor<algorithmFPType> &crossProductBD,
            SharedPtr<NumericTable> &meanTable, BlockDescriptor<algorithmFPType> &sumBD,
            SharedPtr<NumericTable> &nObservationsTable, BlockDescriptor<algorithmFPType> &nObservationsBD)
{
    covTable->releaseBlockOfRows(crossProductBD);
    meanTable->releaseBlockOfRows(sumBD);
    nObservationsTable->releaseBlockOfRows(nObservationsBD);
}

template<typename algorithmFPType, Method method, CpuType cpu>
void updateDenseCrossProductAndSums(bool isNormalized, size_t nFeatures, size_t nVectors,
            algorithmFPType *dataBlock, algorithmFPType *crossProduct, algorithmFPType *sums,
            algorithmFPType *nObservations, services::SharedPtr<services::KernelErrorCollection> &_errors)
{
    if (isNormalized)
    {
        char uplo  = 'U';
        char trans = 'N';
        algorithmFPType alpha = 1.0;
        algorithmFPType beta  = 1.0;

        Blas<algorithmFPType, cpu>::xsyrk(&uplo, &trans, (MKL_INT *) &nFeatures, (MKL_INT *) &nVectors,
                           &alpha, dataBlock, (MKL_INT *) &nFeatures, &beta,
                           crossProduct, (MKL_INT *) &nFeatures);
    }
    else
    {
        __int64 mklMethod = __DAAL_VSL_SS_METHOD_FAST;
        switch (method)
        {
        case defaultDense:
            mklMethod = __DAAL_VSL_SS_METHOD_FAST;
            break;
        case singlePassDense:
            mklMethod = __DAAL_VSL_SS_METHOD_1PASS;
            break;
        case sumDense:
            mklMethod = __DAAL_VSL_SS_METHOD_FAST_USER_MEAN;
            break;
        default:
            break;
        }

        int errcode = Statistics<algorithmFPType, cpu>::xcp(dataBlock, (__int64)nFeatures, (__int64)nVectors,
                                       nObservations, sums, crossProduct, mklMethod);
        if (errcode != 0) { _errors->add(services::ErrorCovarianceInternal); return; }
    }
    *nObservations += (algorithmFPType)nVectors;
}

template<typename algorithmFPType, Method method, CpuType cpu>
void updateDensePartialResults(SharedPtr<NumericTable> &dataTable,
        SharedPtr<NumericTable> &crossProductTable, SharedPtr<NumericTable> &sumTable,
        SharedPtr<NumericTable> &nObservationsTable, bool isOnline,
        services::SharedPtr<services::KernelErrorCollection> &_errors)
{
    size_t nFeatures = dataTable->getNumberOfColumns();
    size_t nVectors  = dataTable->getNumberOfRows();
    bool isNormalized = dataTable->isNormalized(NumericTableIface::standardScoreNormalized);

    BlockDescriptor<algorithmFPType> crossProductBD, sumBD, nObservationsBD;
    algorithmFPType *crossProduct, *sums, *nObservations;
    ReadWriteMode rwMode = (isOnline ? readWrite : writeOnly);

    getDenseCrossProductAndSums<algorithmFPType, method, cpu>(nFeatures, rwMode,
        crossProductTable, crossProductBD, &crossProduct, sumTable, sumBD, &sums,
        nObservationsTable, nObservationsBD, &nObservations, dataTable, _errors);

    if (!isOnline)
    {
        algorithmFPType zero = 0.0;
        daal::services::internal::service_memset<algorithmFPType, cpu>(crossProduct, zero, nFeatures * nFeatures);
        if (method != sumDense && method != sumCSR)
        {
            daal::services::internal::service_memset<algorithmFPType, cpu>(sums, zero, nFeatures);
        }
    }

    /* Retrieve data associated with input table */
    BlockDescriptor<algorithmFPType> dataBD;
    dataTable->getBlockOfRows(0, nVectors, readOnly, dataBD);
    algorithmFPType *dataBlock = dataBD.getBlockPtr();

    updateDenseCrossProductAndSums<algorithmFPType, method, cpu>(isNormalized, nFeatures, nVectors,
        dataBlock, crossProduct, sums, nObservations, _errors);

    dataTable->releaseBlockOfRows(dataBD);
    releaseDenseCrossProductAndSums<algorithmFPType, cpu>(crossProductTable, crossProductBD, sumTable, sumBD,
        nObservationsTable, nObservationsBD);
}

template<typename algorithmFPType, Method method, CpuType cpu>
void updateCSRCrossProductAndSums(size_t nFeatures, size_t nVectors,
            algorithmFPType *dataBlock, size_t *colIndices, size_t *rowOffsets,
            algorithmFPType *crossProduct, algorithmFPType *sums, algorithmFPType *nObservations,
            services::SharedPtr<services::KernelErrorCollection> &_errors)
{
    char transa = 'T';
    SpBlas<algorithmFPType, cpu>::xcsrmultd(&transa, (MKL_INT *)&nVectors, (MKL_INT *)&nFeatures, (MKL_INT *)&nFeatures,
            dataBlock, (MKL_INT *)colIndices, (MKL_INT *)rowOffsets,
            dataBlock, (MKL_INT *)colIndices, (MKL_INT *)rowOffsets, crossProduct, (MKL_INT *)&nFeatures);

    if (method != sumCSR)
    {
        algorithmFPType one = 1.0;
        algorithmFPType *ones = (algorithmFPType *)daal_malloc(nVectors * sizeof(algorithmFPType));
        if (!ones) { _errors->add(services::ErrorMemoryAllocationFailed); return; }
        daal::services::internal::service_memset<algorithmFPType, cpu>(ones, one, nVectors);

        char matdescra[6];
        matdescra[0] = 'G';        // general matrix
        matdescra[3] = 'F';        // 1-based indexing

        matdescra[1] = (char) 0;
        matdescra[2] = (char) 0;
        matdescra[4] = (char) 0;
        matdescra[5] = (char) 0;
        SpBlas<algorithmFPType, cpu>::xcsrmv(&transa, (MKL_INT *)&nVectors, (MKL_INT *)&nFeatures, &one,
                matdescra, dataBlock, (MKL_INT *)colIndices, (MKL_INT *)rowOffsets, (MKL_INT *)rowOffsets + 1,
                ones, &one, sums);
        daal_free(ones);
    }

    nObservations[0] += (algorithmFPType)nVectors;
}

template<typename algorithmFPType, CpuType cpu>
void mergeCrossProductAndSums(size_t nFeatures,
    const algorithmFPType *partialCrossProduct, const algorithmFPType *partialSums,
    const algorithmFPType *partialNObservations,
    algorithmFPType *crossProduct, algorithmFPType *sums, algorithmFPType *nObservations)
{
    /* Merge cross-products */
    algorithmFPType partialNObsValue = partialNObservations[0];

    if (partialNObsValue != 0)
    {
        algorithmFPType nObsValue = nObservations[0];

        if (nObsValue == 0)
        {
            daal::threader_for( nFeatures, nFeatures, [ = ](size_t i)
            {
              PRAGMA_IVDEP
              PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j <= i; j++)
                {
                    crossProduct[i * nFeatures + j] += partialCrossProduct[i * nFeatures + j];
                    crossProduct[j * nFeatures + i]  = crossProduct[i * nFeatures + j];
                }
            } );
        }
        else
        {
            algorithmFPType invPartialNObs = 1.0 / partialNObsValue;
            algorithmFPType invNObs = 1.0 / nObsValue;
            algorithmFPType invNewNObs = 1.0 / (nObsValue + partialNObsValue);

            daal::threader_for( nFeatures, nFeatures, [ = ](size_t i)
            {
              PRAGMA_IVDEP
              PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j <= i; j++)
                {
                    crossProduct[i * nFeatures + j] += partialCrossProduct[i * nFeatures + j];
                    crossProduct[i * nFeatures + j] += partialSums[i] * partialSums[j] * invPartialNObs;
                    crossProduct[i * nFeatures + j] += sums[i] * sums[j] * invNObs;
                    crossProduct[i * nFeatures + j] -= (partialSums[i] + sums[i]) * (partialSums[j] + sums[j]) * invNewNObs;
                    crossProduct[j * nFeatures + i]  = crossProduct[i * nFeatures + j];
                }
            } );
        }

        /* Merge number of observations */
        nObservations[0] += partialNObservations[0];

        /* Merge sums */
        for (size_t i = 0; i < nFeatures; i++)
        {
            sums[i] += partialSums[i];
        }
    }
}

template<typename algorithmFPType, CpuType cpu>
void finalizeCovariance(size_t nFeatures, algorithmFPType nObservations,
            algorithmFPType *crossProduct, algorithmFPType *sums,
            algorithmFPType *cov, algorithmFPType *mean, const Parameter *parameter,
            services::SharedPtr<services::KernelErrorCollection> &_errors)
{
    algorithmFPType invNObservations = 1.0 / nObservations;
    algorithmFPType invNObservationsM1 = 1.0;
    if (nObservations > 1.0)
    {
        invNObservationsM1 = 1.0 / (nObservations - 1.0);
    }

    /* Calculate resulting mean vector */
    for (size_t i = 0; i < nFeatures; i++)
    {
        mean[i] = sums[i] * invNObservations;
    }

    if (parameter->outputMatrixType == correlationMatrix)
    {
        /* Calculate resulting correlation matrix */
        algorithmFPType *diagInvSqrts = (algorithmFPType *)daal::services::daal_malloc(nFeatures * sizeof(algorithmFPType));
        if (!diagInvSqrts)
        { _errors->add(services::ErrorMemoryAllocationFailed); return; }

        for (size_t i = 0; i < nFeatures; i++)
        {
            diagInvSqrts[i] = 1.0 / sSqrt<cpu>(crossProduct[i * nFeatures + i]); //TODO: VML->invsqrt
        }

        for (size_t i = 0; i < nFeatures; i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                cov[i * nFeatures + j] = crossProduct[i * nFeatures + j] * diagInvSqrts[i] * diagInvSqrts[j];
            }
            cov[i * nFeatures + i] = 1.0; //diagonal element
        }

        daal::services::daal_free(diagInvSqrts);
    }
    else
    {
        /* Calculate resulting covariance matrix */
        for (size_t i = 0; i < nFeatures; i++)
        {
            for (size_t j = 0; j <= i; j++)
            {
                cov[i * nFeatures + j] = crossProduct[i * nFeatures + j] * invNObservationsM1;
            }
        }
    }

    /* Copy results into symmetric upper triangle */
    for (size_t i = 0; i < nFeatures; i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            cov[j * nFeatures + i] = cov[i * nFeatures + j];
        }
    }
}

template<typename algorithmFPType, CpuType cpu>
void finalizeCovariance(SharedPtr<NumericTable> &covTable, SharedPtr<NumericTable> &meanTable,
        SharedPtr<NumericTable> &nObservationsTable, const Parameter *parameter,
        services::SharedPtr<services::KernelErrorCollection> &_errors)
{
    BlockDescriptor<algorithmFPType> covBD, meanBD, nObservationsBD;
    algorithmFPType *cov, *mean, *nObservations;
    getTableData<algorithmFPType, cpu>(readWrite, covTable, covBD, &cov);
    getTableData<algorithmFPType, cpu>(readWrite, meanTable, meanBD, &mean);
    getTableData<algorithmFPType, cpu>(readOnly, nObservationsTable, nObservationsBD, &nObservations);

    size_t nFeatures = covTable->getNumberOfColumns();

    finalizeCovariance<algorithmFPType, cpu>(nFeatures, *nObservations, cov, mean, cov, mean, parameter, _errors);

    covTable->releaseBlockOfRows(covBD);
    meanTable->releaseBlockOfRows(meanBD);
    nObservationsTable->releaseBlockOfRows(nObservationsBD);
}

template<typename algorithmFPType, CpuType cpu>
void finalizeCovariance(SharedPtr<NumericTable> &crossProductTable,
        SharedPtr<NumericTable> &sumTable, SharedPtr<NumericTable> &nObservationsTable,
        SharedPtr<NumericTable> &covTable, SharedPtr<NumericTable> &meanTable,
        const Parameter *parameter, services::SharedPtr<services::KernelErrorCollection> &_errors)
{
    size_t nFeatures = covTable->getNumberOfColumns();

    BlockDescriptor<algorithmFPType> crossProductBD, sumBD, nObservationsBD;
    algorithmFPType *crossProduct, *sums, *nObservations;
    getDenseCrossProductAndSums<algorithmFPType, cpu>(readOnly,
        crossProductTable, crossProductBD, &crossProduct, sumTable, sumBD, &sums,
        nObservationsTable, nObservationsBD, &nObservations);

    BlockDescriptor<algorithmFPType> covBD, meanBD;
    algorithmFPType *cov, *mean;
    getTableData<algorithmFPType, cpu>(writeOnly, covTable, covBD, &cov);
    getTableData<algorithmFPType, cpu>(writeOnly, meanTable, meanBD, &mean);

    finalizeCovariance<algorithmFPType, cpu>(nFeatures, *nObservations, crossProduct, sums, cov, mean, parameter, _errors);

    releaseDenseCrossProductAndSums<algorithmFPType, cpu>(crossProductTable, crossProductBD, sumTable, sumBD,
        nObservationsTable, nObservationsBD);

    covTable->releaseBlockOfRows(covBD);
    meanTable->releaseBlockOfRows(meanBD);
}

} // namespace internal
} // namespace covariance
} // namespace algorithms
} // namespace daal

#endif
