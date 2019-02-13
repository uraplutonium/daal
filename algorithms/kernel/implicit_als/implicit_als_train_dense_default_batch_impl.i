/* file: implicit_als_train_dense_default_batch_impl.i */
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
//  Implementation of impicit ALS training algorithm for batch processing mode
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_DENSE_DEFAULT_BATCH_IMPL_I__
#define __IMPLICIT_ALS_TRAIN_DENSE_DEFAULT_BATCH_IMPL_I__

#include "threading.h"
#include "service_blas.h"
#include "service_lapack.h"
#include "service_error_handling.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace internal
{

using namespace daal::internal;
using namespace daal::services;

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainKernelCommon<algorithmFPType, cpu>::computeXtX(
    size_t *nRows, size_t *nCols, algorithmFPType *beta, algorithmFPType *x, size_t *ldx,
    algorithmFPType *xtx, size_t *ldxtx)
{
    /* SYRK parameters */
    char uplo = 'U';
    char trans = 'N';
    algorithmFPType alpha = 1.0;

    Blas<algorithmFPType, cpu>::xsyrk(&uplo, &trans, (DAAL_INT *)nCols, (DAAL_INT *)nRows, &alpha, x, (DAAL_INT *)ldx, beta,
                       xtx, (DAAL_INT *)ldxtx);
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainKernelBase<algorithmFPType, cpu>::updateSystem(
    size_t nCols, const algorithmFPType *x, const algorithmFPType *coeff, const algorithmFPType *c,
    algorithmFPType *a, algorithmFPType *b)
{
    /* SYR parameters */
    const char uplo = 'U';
    const DAAL_INT iOne = 1;
    Blas<algorithmFPType, cpu>::xxsyr(&uplo, (DAAL_INT *)&nCols, coeff, x, &iOne, a, (DAAL_INT *)&nCols);

    if (*coeff > 0.0)
    {
        Blas<algorithmFPType, cpu>::xxaxpy((DAAL_INT *)&nCols, c, x, &iOne, b, &iOne);
    }
}

template <typename algorithmFPType, CpuType cpu>
bool ImplicitALSTrainKernelBase<algorithmFPType, cpu>::solve(size_t nCols, algorithmFPType *a, algorithmFPType *b)
{
    /* POTRF parameters */
    char uplo = 'U';
    DAAL_INT iOne = 1;
    DAAL_INT info = 0;

    /* Perform L*L' decomposition of A */
    Lapack<algorithmFPType, cpu>::xxpotrf(&uplo, (DAAL_INT *)&nCols, a, (DAAL_INT *)&nCols, &info);
    if(info != 0)
        return false;

    /* Solve L*L' * x = b */
    Lapack<algorithmFPType, cpu>::xxpotrs(&uplo, (DAAL_INT *)&nCols, &iOne, a, (DAAL_INT *)&nCols, b, (DAAL_INT *)&nCols, &info);
    return (info == 0);
}

static inline void getSizes( size_t  nRows,
                             size_t  nCols,
                             size_t& nBlocks,
                             size_t& blockSize,
                             size_t& tailSize)
{
    const size_t nThreads     = threader_get_threads_number();
    const size_t maxBlockSize  = 100000;
    const size_t maxRowsOnBlock = maxBlockSize / nCols ? maxBlockSize / nCols : 1;

    const size_t minNumOfBlocks = (nRows + maxRowsOnBlock - 1) / maxRowsOnBlock;
    const size_t k = (minNumOfBlocks + nThreads - 1) / nThreads;

    nBlocks = k*nThreads;
    blockSize = nRows / nBlocks;
    if (!blockSize)                // Case where nRows < nThreads
    {
        nBlocks = nRows;
        blockSize = 1;
    }
    tailSize = nRows - nBlocks*blockSize; // Number of blocks with blockSize+1 rows
    return;
}

template <typename algorithmFPType, CpuType cpu>
Status ImplicitALSTrainKernelBase<algorithmFPType, cpu>::computeFactors(
   size_t nRows, size_t nCols, const algorithmFPType *data, const size_t *colIndices, const size_t *rowOffsets,
    size_t nFactors, algorithmFPType *colFactors, algorithmFPType *rowFactors,
    algorithmFPType alpha, algorithmFPType lambda, algorithmFPType *xtx, daal::tls<algorithmFPType *>& lhs)
{
    SafeStatus safeStat;
    size_t nBlocks, blockSize, tailSize;

    getSizes( nRows, nCols, nBlocks, blockSize, tailSize );

    daal::threader_for(nBlocks, nBlocks, [ & ]( size_t i )
    {
        const size_t curBlockSize = ( i < tailSize ) ? blockSize + 1 : blockSize;
        const size_t offset = ( i < tailSize ) ? i * blockSize + i : i * blockSize + tailSize;

        for( size_t j = 0; j < curBlockSize; j++ )
        {
            algorithmFPType *lhs_local = lhs.local();
            algorithmFPType *rhs = rowFactors + (offset + j) * nFactors;

            for(int f = 0; f < nFactors; f++){ rhs[f] = 0.0; }
            daal::services::daal_memcpy_s(lhs_local, nFactors * nFactors * sizeof(algorithmFPType),
                                      xtx, nFactors * nFactors * sizeof(algorithmFPType));

            formSystem( offset + j, nCols, data, colIndices, rowOffsets, nFactors, colFactors, alpha, lhs_local, rhs, lambda);

            /* Solve system of normal equations */
            if(!solve(nFactors, lhs_local, rhs))
                safeStat.add(ErrorALSInternal);
        } /* for(size_t j = 0; j < curBlockSize; j++) */
    }); /* daal::threader_for(nBlocks, nBlocks, [ & ](size_t i) */

    return safeStat.detach();
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainKernel<algorithmFPType, fastCSR, cpu>::computeCostFunction(
    size_t nUsers, size_t nItems, size_t nFactors, algorithmFPType *data, size_t *colIndices, size_t *rowOffsets,
    algorithmFPType *itemsFactors, algorithmFPType *usersFactors, algorithmFPType alpha, algorithmFPType lambda,
    algorithmFPType *costFunctionPtr)
{
    algorithmFPType zero = 0.0;
    algorithmFPType one = 1.0;
    algorithmFPType costFunction = zero;
    algorithmFPType sumUsers2 = zero;
    algorithmFPType sumItems2 = zero;

    for (size_t i = 0; i < nUsers; i++)
    {
        size_t startIdx = rowOffsets[i]   - 1;
        size_t endIdx   = rowOffsets[i + 1] - 1;
        algorithmFPType *usersI = usersFactors + i * nFactors;
        for (size_t j = startIdx; j < endIdx; j++)
        {
            algorithmFPType c = one + alpha * data[j];
            algorithmFPType *itemsJ = itemsFactors + (colIndices[j] - 1) * nFactors;

            algorithmFPType dotProduct = 0.0;
            for (size_t k = 0; k < nFactors; k++)
            {
                dotProduct += usersI[k] * itemsJ[k];
            }
            algorithmFPType sqrError = 1.0 - dotProduct;
            sqrError *= sqrError;
            costFunction += c * sqrError;
        }
    }
    for (size_t i = 0; i < nItems * nFactors; i++)
    {
        sumItems2 += itemsFactors[i] * itemsFactors[i];
    }
    for (size_t i = 0; i < nUsers * nFactors; i++)
    {
        sumUsers2 += usersFactors[i] * usersFactors[i];
    }
    costFunction += lambda * (sumItems2 + sumUsers2);
    *costFunctionPtr = costFunction;
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainKernel<algorithmFPType, defaultDense, cpu>::computeCostFunction(
    size_t nUsers, size_t nItems, size_t nFactors, algorithmFPType *data, size_t *colIndices, size_t *rowOffsets,
    algorithmFPType *itemsFactors, algorithmFPType *usersFactors, algorithmFPType alpha, algorithmFPType lambda,
    algorithmFPType *costFunctionPtr)
{
    algorithmFPType zero = 0.0;
    algorithmFPType one = 1.0;
    algorithmFPType costFunction = zero;
    algorithmFPType sumUsers2 = zero;
    algorithmFPType sumItems2 = zero;

    for (size_t i = 0; i < nUsers; i++)
    {
        algorithmFPType *usersI = usersFactors + i * nFactors;
        for (size_t j = 0; j < nItems; j++)
        {
            if (data[i * nItems + j] > 0.0)
            {
                algorithmFPType *itemsJ = itemsFactors + j * nFactors;
                algorithmFPType c = one + alpha * data[i * nItems + j];

                algorithmFPType dotProduct = 0.0;
                for (size_t k = 0; k < nFactors; k++)
                {
                    dotProduct += usersI[k] * itemsJ[k];
                }
                algorithmFPType sqrError = one - dotProduct;
                sqrError *= sqrError;
                costFunction += c * sqrError;
            }
        }
    }
    for (size_t i = 0; i < nItems * nFactors; i++)
    {
        sumItems2 += itemsFactors[i] * itemsFactors[i];
    }
    for (size_t i = 0; i < nUsers * nFactors; i++)
    {
        sumUsers2 += usersFactors[i] * usersFactors[i];
    }
    costFunction += lambda * (sumItems2 + sumUsers2);
    *costFunctionPtr = costFunction;
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainKernel<algorithmFPType, fastCSR, cpu>::formSystem(
    size_t i, size_t nCols, const algorithmFPType *data, const size_t *colIndices, const size_t *rowOffsets,
    size_t nFactors, algorithmFPType *colFactors,
    algorithmFPType alpha, algorithmFPType *lhs, algorithmFPType *rhs, algorithmFPType lambda)
{
    size_t startIdx = rowOffsets[i]   - 1;
    size_t endIdx   = rowOffsets[i + 1] - 1;
    /* Update the linear system of normal equations */
    for (size_t j = startIdx; j < endIdx; j++)
    {
        algorithmFPType c1 = alpha * data[j];
        algorithmFPType c = c1 + 1.0;
        algorithmFPType *colFactorsRow = colFactors + (colIndices[j] - 1) * nFactors;

        this->updateSystem(nFactors, colFactorsRow, &c1, &c, lhs, rhs);
    }

    /* Add regularization term */
    algorithmFPType gamma = lambda * (endIdx - startIdx);
    for (size_t k = 0; k < nFactors; k++)
    {
        lhs[k * nFactors + k] += gamma;
    }
}


template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainKernel<algorithmFPType, defaultDense, cpu>::formSystem(
    size_t i, size_t nCols, const algorithmFPType *data, const size_t *colIndices, const size_t *rowOffsets,
    size_t nFactors, algorithmFPType *colFactors,
    algorithmFPType alpha, algorithmFPType *lhs, algorithmFPType *rhs, algorithmFPType lambda)
{
    algorithmFPType one = 1.0;
    algorithmFPType gammaMultiplier = 1.0;
    /* Update the linear system of normal equations */
    for (size_t j = 0; j < nCols; j++)
    {
        algorithmFPType rating = data[i * nCols + j];
        if (rating > 0.0)
        {
            algorithmFPType c1 = alpha * rating;
            algorithmFPType c = c1 + 1.0;
            algorithmFPType *colFactorsRow = colFactors + j * nFactors;

            this->updateSystem(nFactors, colFactorsRow, &c1, &c, lhs, rhs);
            gammaMultiplier += one;
        }
    }

    /* Add regularization term */
    const algorithmFPType gamma = lambda * gammaMultiplier;
    for (size_t k = 0; k < nFactors; k++)
    {
        lhs[k * nFactors + k] += gamma;
    }
}

template <typename algorithmFPType, CpuType cpu>
services::Status ImplicitALSTrainBatchKernel<algorithmFPType, fastCSR, cpu>::compute(const NumericTable *dataTable,
                                                                                     implicit_als::Model *initModel,
                                                                                     implicit_als::Model *model,
                                                                                     const Parameter *parameter)
{
    Status s;
    ImplicitALSTrainTask<algorithmFPType, fastCSR, cpu> task(dataTable, model, parameter);
    DAAL_CHECK_STATUS(s, task.init(dataTable, initModel, parameter));

    const algorithmFPType alpha(parameter->alpha);
    const algorithmFPType lambda(parameter->lambda);

    size_t nItems = task.nItems;
    size_t nUsers = task.nUsers;
    size_t nFactors = task.nFactors;
    algorithmFPType *itemsFactors = task.mtItemsFactors.get();
    algorithmFPType *usersFactors = task.mtUsersFactors.get();
    algorithmFPType *xtx = task.xtx.get();

    const algorithmFPType *data = task.mtData.values();
    algorithmFPType *tdata = task.tdata.get();
    const size_t *colIndices = task.mtData.cols();
    const size_t *rowOffsets = task.mtData.rows();
    size_t *rowIndices = task.rowIndices.get();
    size_t *colOffsets = task.colOffsets.get();

#if 0
    algorithmFPType costFunction;
    computeCostFunction(nUsers, nItems, nFactors, data, colIndices, rowOffsets, itemsFactors, usersFactors,
                        alpha, lambda, &costFunction);
#endif
    daal::tls<algorithmFPType *> lhs([=]() -> algorithmFPType*
    {
        return (algorithmFPType *)daal::services::daal_malloc(parameter->nFactors * parameter->nFactors * sizeof(algorithmFPType));
    });

    algorithmFPType beta = 0.0;
    for(size_t i = 0; i < parameter->maxIterations; i++)
    {
        this->computeXtX(&nItems, &nFactors, &beta, itemsFactors, &nFactors, xtx, &nFactors);

        s = this->computeFactors(nUsers, nItems, data, colIndices, rowOffsets, nFactors, itemsFactors, usersFactors,
                             alpha, lambda, xtx, lhs);
        if(!s)
            break;

        this->computeXtX(&nUsers, &nFactors, &beta, usersFactors, &nFactors, xtx, &nFactors);

        s = this->computeFactors(nItems, nUsers, tdata, rowIndices, colOffsets, nFactors, usersFactors, itemsFactors,
                             alpha, lambda, xtx, lhs);
        if(!s)
            break;

#if 0
        computeCostFunction(nUsers, nItems, nFactors, data, colIndices, rowOffsets, itemsFactors, usersFactors,
                            alpha, lambda, &costFunction);
#endif
    }
    lhs.reduce([](algorithmFPType* lhsData)
    {
        if(lhsData) { daal::services::daal_free(lhsData); }
    });
    return s;
}

template <typename algorithmFPType, CpuType cpu>
services::Status ImplicitALSTrainBatchKernel<algorithmFPType, defaultDense, cpu>::compute(const NumericTable *dataTable,
                                                                                          implicit_als::Model *initModel,
                                                                                          implicit_als::Model *model,
                                                                                          const Parameter *parameter)
{
    ImplicitALSTrainTask<algorithmFPType, defaultDense, cpu> task(dataTable, model, parameter);
    Status s = task.init(dataTable, initModel, parameter);
    if(!s)
        return s;

    const algorithmFPType alpha(parameter->alpha);
    const algorithmFPType lambda(parameter->lambda);

    size_t nItems = task.nItems;
    size_t nUsers = task.nUsers;
    size_t nFactors = task.nFactors;
    algorithmFPType *itemsFactors = task.mtItemsFactors.get();
    algorithmFPType *usersFactors = task.mtUsersFactors.get();
    algorithmFPType *xtx = task.xtx.get();

    const algorithmFPType *data = task.mtData.get();
    algorithmFPType *tdata = task.tdata.get();

#if 0
    algorithmFPType costFunction;
    computeCostFunction(nUsers, nItems, nFactors, data, NULL, NULL, itemsFactors, usersFactors,
                        alpha, lambda, &costFunction);
#endif

    daal::tls<algorithmFPType *> lhs([=]() -> algorithmFPType*
    {
        return (algorithmFPType *)daal::services::daal_malloc(parameter->nFactors * parameter->nFactors * sizeof(algorithmFPType));
    });
    algorithmFPType beta = 0.0;
    for(size_t i = 0; i < parameter->maxIterations; i++)
    {
        this->computeXtX(&nItems, &nFactors, &beta, itemsFactors, &nFactors, xtx, &nFactors);

        s = this->computeFactors(nUsers, nItems, data, NULL, NULL, nFactors, itemsFactors, usersFactors,
                             alpha, lambda, xtx, lhs);
        if(!s)
            break;

        this->computeXtX(&nUsers, &nFactors, &beta, usersFactors, &nFactors, xtx, &nFactors);

        s = this->computeFactors(nItems, nUsers, tdata, NULL, NULL, nFactors, usersFactors, itemsFactors,
                             alpha, lambda, xtx, lhs);
        if(!s)
            break;

#if 0
        computeCostFunction(nUsers, nItems, nFactors, data, NULL, NULL, itemsFactors, usersFactors,
                            alpha, lambda, &costFunction);
#endif
    }
    lhs.reduce([](algorithmFPType* lhsData)
    {
        if(lhsData) { daal::services::daal_free(lhsData); }
    });
    return s;
}

}
}
}
}
}

#endif
