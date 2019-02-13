/* file: pca_transform_dense_default_batch_impl.i */
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
//  Common functions for pca transformation
//--
*/

#ifndef __PCA_TRANSFORM_DENSE_DEFAULT_BATCH_IMPL_I__
#define __PCA_TRANSFORM_DENSE_DEFAULT_BATCH_IMPL_I__

#include "daal_defines.h"
#include "algorithm.h"
#include "threading.h"
#include "numeric_table.h"

#include "service_blas.h"
#include "service_math.h"
#include "service_unique_ptr.h"
#include "service_numeric_table.h"
#include "service_error_handling.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace transform
{
namespace internal
{
using namespace daal::internal;
using namespace daal::services;

template<typename algorithmFPType, transform::Method method, CpuType cpu>
void TransformKernel<algorithmFPType, method, cpu>::computeTransformedBlock
        (DAAL_INT *numRows, DAAL_INT *numFeatures, DAAL_INT *numComponents,
         const algorithmFPType *dataBlock,
         const algorithmFPType *eigenvectors,
         algorithmFPType *resultBlock)
{
    /* GEMM parameters */
    char trans   = 'T';
    char notrans = 'N';
    algorithmFPType one  = 1.0;
    algorithmFPType zero = 0.0;

    Blas<algorithmFPType, cpu>::xxgemm(&trans, &notrans, numComponents, numRows, numFeatures,
        &one, eigenvectors, numFeatures, dataBlock, numFeatures, &zero, resultBlock, numComponents);

} /* void TransformKernel<algorithmFPType, defaultDense, cpu>::computeTransformedBlock */


template<typename algorithmFPType, CpuType cpu>
services::Status ComputeInvSigmas(NumericTable *pVariances,
                                  TArray<algorithmFPType, cpu>& invSigmas,
                                  size_t numFeatures)
{
    services::Status status;
    if (pVariances != nullptr)
    {
        algorithmFPType* pInvSigmas = invSigmas.reset(numFeatures);
        pInvSigmas = invSigmas.get();
        DAAL_CHECK_MALLOC(pInvSigmas);

        ReadRows<algorithmFPType, cpu> dataRows(*pVariances, 0, numFeatures);
        DAAL_CHECK_BLOCK_STATUS(dataRows);
        const algorithmFPType *pRawVariances = dataRows.get();

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t varianceId = 0; varianceId < numFeatures; ++varianceId)
        {
              pInvSigmas[varianceId] = pRawVariances[varianceId] ?
                algorithmFPType(1.0) / daal::internal::Math<algorithmFPType, cpu>::sSqrt(pRawVariances[varianceId]) :
                algorithmFPType(0.0);
        }
    }
    return status;
}

template<typename algorithmFPType, transform::Method method, CpuType cpu>
services::Status TransformKernel<algorithmFPType, method, cpu>::compute
        (NumericTable& data, NumericTable& eigenvectors,
         NumericTable *pMeans, NumericTable *pVariances, NumericTable *pEigenvalues,
         NumericTable &transformedData)
{
    DAAL_INT numVectors  = data.getNumberOfRows();
    DAAL_INT numFeatures = data.getNumberOfColumns();
    DAAL_INT numComponents = transformedData.getNumberOfColumns();


    /* Retrieve data associated with coefficients */
    ReadRows<algorithmFPType, cpu> basis(eigenvectors, 0, numComponents);
    DAAL_CHECK_BLOCK_STATUS(basis)
    const algorithmFPType *pBasis = basis.get();

    size_t numRowsInBlock = _numRowsInBlock;
    if (numRowsInBlock < 1) { numRowsInBlock = 1; }

    /* Calculate number of blocks of rows including tail block */
    size_t numBlocks = numVectors / numRowsInBlock;
    if (numBlocks * numRowsInBlock < numVectors) { numBlocks++; }

    Status status;

    TArray<algorithmFPType, cpu> invSigmas(0);
    DAAL_CHECK_STATUS(status, ComputeInvSigmas(pVariances, invSigmas, numFeatures));

    TArray<algorithmFPType, cpu> invEigenvalues(0);
    DAAL_CHECK_STATUS(status, ComputeInvSigmas(pEigenvalues, invEigenvalues, numComponents));

    const algorithmFPType *pInvSigmas = invSigmas.get();
    const algorithmFPType *pInvEigenvalues = invEigenvalues.get();

    size_t numInvSigmas = invSigmas.size();
    size_t numInvEigenvalues = invEigenvalues.size();
    size_t numMeans = 0;

    const algorithmFPType *pRawMeans = nullptr;
    ReadRows<algorithmFPType, cpu> meansRows;
    if (pMeans != nullptr)
    {
        numMeans = numFeatures;
        meansRows.set(*pMeans, 0, numFeatures);
        DAAL_CHECK_BLOCK_STATUS(meansRows);
        pRawMeans = meansRows.get();
    }

    bool isWhitening = pEigenvalues != nullptr;
    bool isNormalize = pMeans != nullptr || pVariances != nullptr;

    UniquePtr<daal::tls<algorithmFPType *>, cpu> tls;
    if (isNormalize)
    {
        tls.reset(new daal::tls<algorithmFPType *>([=]()
        {
            return (algorithmFPType *)daal_malloc(numRowsInBlock * numFeatures * sizeof(algorithmFPType));
        }));
        DAAL_CHECK_MALLOC(tls.get());
    }

    SafeStatus safeStat;

    /* Loop over input data blocks */
    daal::threader_for( numBlocks, numBlocks, [ =, &tls, &transformedData, &data, &safeStat ](int iBlock)
    {
        size_t startRow = iBlock * numRowsInBlock;
        size_t endRow = startRow + numRowsInBlock;
        if (endRow > numVectors) { endRow = numVectors; }

        DAAL_INT numRows = endRow - startRow;
        DAAL_INT numFeatures = data.getNumberOfColumns();

        WriteRows<algorithmFPType, cpu> blockRows(transformedData, startRow, endRow);
        DAAL_CHECK_BLOCK_STATUS_THR(blockRows);
        algorithmFPType *pTransformedBlock = blockRows.get();

        ReadRows<algorithmFPType, cpu> dataRows(data, startRow, endRow);
        DAAL_CHECK_BLOCK_STATUS_THR(dataRows);
        const algorithmFPType *pDataBlock = dataRows.get();

        if (isNormalize)
        {
            algorithmFPType* pCopyBlock = tls->local();
            const algorithmFPType* pNormBlock = numMeans ? pCopyBlock : pDataBlock;
            DAAL_CHECK_MALLOC_THR(pCopyBlock);
            for (size_t rowId = 0; rowId < numRows; ++rowId)
            {
                /* compute centering if numMeans != 0 */
                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t colId = 0; colId < numMeans; ++colId)
                {
                    pCopyBlock[rowId * numMeans + colId] = pDataBlock[rowId * numMeans + colId] - pRawMeans[colId];
                }
                /* compute normalization to unit variance if numInvSigmas!= 0 */
                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t colId = 0; colId < numInvSigmas; ++colId)
                {
                    pCopyBlock[rowId * numInvSigmas + colId] = pNormBlock[rowId * numInvSigmas + colId] * pInvSigmas[colId];
                }
            }
            pDataBlock = pCopyBlock;
        }
        computeTransformedBlock(&numRows, &numFeatures, (DAAL_INT*)&numComponents, pDataBlock,
            pBasis, pTransformedBlock);
        /* compute whitening to unit variance of transformed data if required */
        if(isWhitening)
        {
            for (size_t rowId = 0; rowId < numRows; ++rowId)
            {
                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t colId = 0; colId < numComponents; ++colId)
                {
                    pTransformedBlock[rowId * numComponents + colId] = pTransformedBlock[rowId * numComponents + colId] * pInvEigenvalues[colId];
                }
            }
        }
    } ); /* daal::threader_for */

    if (isNormalize)
    {
        tls->reduce([&](algorithmFPType* pCopyBlock)
        {
            daal_free(pCopyBlock);
        });
    }

    return safeStat.detach();
} /* void TransformKernel<algorithmFPType, defaultDense, cpu>::compute */

} /* namespace internal */
} /* namespace transform */
} /* namespace pca */
} /* namespace algorithms */
} /* namespace daal */

#endif
