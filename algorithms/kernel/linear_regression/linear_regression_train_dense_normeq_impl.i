/* file: linear_regression_train_dense_normeq_impl.i */
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
//  Implementation of auxiliary functions for linear regression
//  Normal Equations (normEqDense) method.
//--
*/

#ifndef __LINEAR_REGRESSION_TRAIN_DENSE_NORMEQ_IMPL_I__
#define __LINEAR_REGRESSION_TRAIN_DENSE_NORMEQ_IMPL_I__

#include "service_memory.h"
#include "service_blas.h"
#include "service_lapack.h"
#include "linear_regression_ne_model.h"
#include "linear_regression_train_kernel.h"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace training
{
namespace internal
{

/**
 *  \brief Get arrays holding partial sums from Linear Regression daal::algorithms::Model
 *
 *  \param  daal::algorithms::Model[in]     Linear regression daal::algorithms::Model
 *  \param  dim[in]       Task dimension
 *  \param  ny[in]        Number of responses
 *  \param  rwmode[in]    Flag specifying read/write access to the daal::algorithms::Model's partial results
 *  \param  xtxTable[out] Numeric table containing matrix X'*X
 *  \param  xtxBD[out]    Buffer manager corresponding to xtxTable
 *  \param  xtx[out]      Array containing matrix X'*X
 *  \param  xtyTable[out] Numeric table containing matrix X'*Y
 *  \param  xtyBD[out]    Buffer manager corresponding to xtyTable
 *  \param  xty[out]      Array containing matrix X'*Y
 */
template<typename interm, CpuType cpu>
static void getModelPartialSums(ModelNormEq *model,
                         MKL_INT dim, MKL_INT ny, ReadWriteMode rwmode,
                         NumericTable **xtxTable, BlockDescriptor<interm> &xtxBD, interm **xtx,
                         NumericTable **xtyTable, BlockDescriptor<interm> &xtyBD, interm **xty)
{
    *xtxTable = model->getXTXTable().get();
    *xtyTable = model->getXTYTable().get();

    (*xtxTable)->getBlockOfRows(0, dim, rwmode, xtxBD);
    *xtx = xtxBD.getBlockPtr();
    (*xtyTable)->getBlockOfRows(0, ny,  rwmode, xtyBD);
    *xty = xtyBD.getBlockPtr();
}

/**
 *  \brief Release arrays holding partial sums in Linear Regression daal::algorithms::Model
 *
 *  \param  xtxTable[in]  Numeric table containing matrix X'*X
 *  \param  xtxBD[in]     Buffer manager corresponding to xtxTable
 *  \param  xtyTable[in]  Numeric table containing matrix X'*Y
 *  \param  xtyBD[in]     Buffer manager corresponding to xtyTable
 */
template<typename interm, CpuType cpu>
static void releaseModelNormEqPartialSums(NumericTable *xtxTable, BlockDescriptor<interm> &xtxBD,
                                    NumericTable *xtyTable, BlockDescriptor<interm> &xtyBD)
{
    xtxTable->releaseBlockOfRows(xtxBD);
    xtyTable->releaseBlockOfRows(xtyBD);
}

/**
 *  \brief Function that calculates X'*X from input matrix
 *         of independent variables X.
 *
 *  \param p[in]        Number of columns in input matrix X
 *  \param n[in]        Number of rows in input matrix X
 *  \param x[in]        Input matrix X
 *  \param ldx[in]      Leading dimension of input matrix X (ldx >= p)
 *  \param xtx[out]     Resulting matrix X'*X of size p x ldxtx
 *  \param ldxtx[in]    Leading dimension of matrix X'*X (ldxtx >= p)
 */
template<typename mklFpType, CpuType cpu>
void linreg_xtx_comp(MKL_INT *p, MKL_INT *n, mklFpType *x, MKL_INT *ldx,
                     mklFpType *xtx, MKL_INT *ldxtx, mklFpType *mklBetaCoeff)
{
    /* SYRK parameters */
    char uplo, trans;
    mklFpType alpha;

    uplo = 'U';
    trans = 'N';
    alpha = 1.0;

    Blas<mklFpType, cpu>::xsyrk(&uplo, &trans, p, n, &alpha, x, ldx, mklBetaCoeff, xtx, ldxtx);
}

/**
 *  \brief Function that calculates (X|e)'*(X|e) from input matrix
 *         of independent variables X.
 *
 *  \param p[in]        Number of columns in input matrix X
 *  \param n[in]        Number of rows in input matrix X
 *  \param x[in]        Input matrix X
 *  \param ldx[in]      Leading dimension of input matrix X (ldx >= p)
 *  \param xtx[out]     Resulting matrix (X|e)'*(X|e) of size p x ldxtx
 *  \param ldxtx[in]    Leading dimension of matrix (X|e)'*(X|e) (ldxtx >= p+1)
 */
template<typename mklFpType, CpuType cpu>
void linreg_xtx_comp_b0(MKL_INT *p, MKL_INT *n, mklFpType *x, MKL_INT *ldx,
                        mklFpType *xtx, MKL_INT *ldxtx, mklFpType *mklBetaCoeff)
{
    MKL_INT i, j;
    MKL_INT p_val, n_val, lda_val, ldxtx_val;
    mklFpType *xtx_ptr;
    mklFpType *x_ptr;
    p_val = *p;
    n_val = *n;
    lda_val = *ldx;
    ldxtx_val = *ldxtx;

    linreg_xtx_comp<mklFpType, cpu>(p, n, x, ldx, xtx, ldxtx, mklBetaCoeff);

    xtx_ptr = xtx + p_val * (*ldxtx);

    if (*mklBetaCoeff == (mklFpType)0.0)
    {
        daal::services::internal::service_memset<mklFpType, cpu>(xtx_ptr, *mklBetaCoeff, ldxtx_val);
    }

    for (i = 0, x_ptr = x; i < n_val; i++, x_ptr += lda_val)
    {
        for (j = 0; j < p_val; j++)
        {
            xtx_ptr[j] += x_ptr[j];
        }
    }

    xtx_ptr[p_val] += (mklFpType)n_val;
}

/**
 *  \brief Function that calculates X'*Y from input matrix
 *         of independent variables X and matrix of responses Y.
 *
 *  \param p[in]        Number of columns in input matrix X
 *  \param n[in]        Number of rows in input matrix X
 *  \param x[in]        Input matrix X
 *  \param ldx[in]      Leading dimension of input matrix X (ldx >= p)
 *  \param ny[in]       Number of columns in input matrix Y
 *  \param y[in]        Input matrix Y
 *  \param ldy[in]      Leading dimension of input matrix Y (ldy >= ny)
 *  \param xty[out]     Resulting matrix X'*Y of size ny x ldxty
 *  \param ldxty[in]    Leading dimension of matrix X'*Y (ldxty >= p)
 */
template<typename mklFpType, CpuType cpu>
void linreg_xty_comp(MKL_INT *p, MKL_INT *n, mklFpType *x, MKL_INT *ldx,
                     MKL_INT *ny, mklFpType *y, MKL_INT *ldy,
                     mklFpType *xty, MKL_INT *ldxty, mklFpType *mklBetaCoeff)
{
    /* GEMM parameters */
    char transa, transb;
    mklFpType alpha;

    transa = 'N';
    transb = 'T';
    alpha = 1.0;

    Blas<mklFpType, cpu>::xgemm(&transa, &transb, p, ny, n, &alpha, x, ldx, y, ldy,
                   mklBetaCoeff, xty, ldxty);
}

/**
 *  \brief Function that calculates (X|e)'*Y from input matrix
 *         of independent variables X and matrix of responses Y.
 *
 *  \param p[in]        Number of columns in input matrix X
 *  \param n[in]        Number of rows in input matrix X
 *  \param x[in]        Input matrix X
 *  \param ldx[in]      Leading dimension of input matrix X (ldx >= p)
 *  \param ny[in]       Number of columns in input matrix Y
 *  \param y[in]        Input matrix Y
 *  \param ldy[in]      Leading dimension of input matrix Y (ldy >= ny)
 *  \param xty[out]     Resulting matrix (X|e)'*Y of size ny x ldxty
 *  \param ldxty[in]    Leading dimension of matrix (X|e)'*Y (ldxty >= p+1)
 */
template<typename mklFpType, CpuType cpu>
void linreg_xty_comp_b0(MKL_INT *p, MKL_INT *n, mklFpType *x, MKL_INT *ldx,
                        MKL_INT *ny, mklFpType *y, MKL_INT *ldy,
                        mklFpType *xty, MKL_INT *ldxty, mklFpType *mklBetaCoeff)
{
    MKL_INT i, j;
    MKL_INT p_val, n_val, ny_val, ldy_val, ldxty_val;
    mklFpType *y_ptr;

    linreg_xty_comp<mklFpType, cpu>(p, n, x, ldx, ny, y, ldy, xty, ldxty, mklBetaCoeff);

    p_val = *p;
    n_val = *n;
    ny_val = *ny;
    ldy_val = *ldy;
    ldxty_val = *ldxty;

    if (*mklBetaCoeff == (mklFpType)0.0)
    {
        for (j = 0; j < ny_val; j++)
        {
            xty[j * ldxty_val + p_val] = 0.0;
        }
    }

    for (i = 0, y_ptr = y; i < n_val; i++, y_ptr += ldy_val)
    {
        for (j = 0; j < ny_val; j++)
        {
            xty[j * ldxty_val + p_val] += y_ptr[j];
        }
    }
}

template <typename interm, CpuType cpu>
static void updatePartialSums(MKL_INT *dim, MKL_INT *n,
                       MKL_INT *betadim,
                       interm *x, interm *xtx,
                       MKL_INT *ny, interm *y, interm *xty, interm *mklBetaCoeff)
{
    if (*dim < *betadim)
    {
        /* Here if intercept term will be calculated */
        linreg_xtx_comp_b0<interm, cpu>(dim, n, x, dim, xtx, betadim, mklBetaCoeff);
        linreg_xty_comp_b0<interm, cpu>(dim, n, x, dim, ny, y, ny, xty, betadim, mklBetaCoeff);
    }
    else
    {
        /* Here if intercept term won't be calculated */
        linreg_xtx_comp<interm, cpu>(dim, n, x, dim, xtx, betadim, mklBetaCoeff);
        linreg_xty_comp<interm, cpu>(dim, n, x, dim, ny, y, ny, xty, betadim, mklBetaCoeff);
    }
}

/**
 *  \brief Function that calculates linear regression coefficients
 *         from matrices X'*X and X'*Y.
 *
 *  \param p[in]        Number of rows in input matrix X'*X
 *  \param xtx[in]      Input matrix X'*X
 *  \param ldxtx[in]    Leading dimension of matrix X'*X (ldxtx >= p)
 *  \param ny[in]       Number of rows in input matrix X'*Y
 *  \param xty[in]      Input matrix X'*Y
 *  \param ldxty[in]    Leading dimension of matrix X'*Y (ldxty >= p)
 *  \param beta[out]    Resulting matrix of coefficients of size ny x ldxty
 */
template <typename interm, CpuType cpu>
static void linreg_finalize(MKL_INT *p,  interm *xtx, MKL_INT *ldxtx,
                     MKL_INT *ny, interm *xty, MKL_INT *ldxty, interm *beta,
                     services::SharedPtr<services::KernelErrorCollection> &_errors)
{
    MKL_INT n;
    MKL_INT i_one = 1;
    char uplo = 'U';
    MKL_INT info;

    n = (*ny) * (*ldxty);
    daal::services::daal_memcpy_s(beta,n*sizeof(interm),xty,n*sizeof(interm));

    /* Perform L*L' decomposition of X'*X */
    Lapack<interm, cpu>::xpotrf( &uplo, p, xtx, ldxtx, &info );
    if ( info < 0 ) { _errors->add(services::ErrorLinearRegressionInternal); return; }
    if ( info > 0 ) { _errors->add(services::ErrorNormEqSystemSolutionFailed); return; }

    /* Solve L*L'*b=Y */
    Lapack<interm, cpu>::xpotrs( &uplo, p, ny, xtx, ldxtx, beta, ldxty, &info );
    if ( info != 0 ) { _errors->add(services::ErrorLinearRegressionInternal); return; }
}

template <typename interm, CpuType cpu>
void updatePartialModelNormEq(SharedPtr<NumericTable> &x, SharedPtr<NumericTable> &y,
            SharedPtr<linear_regression::Model> &r,
            const daal::algorithms::Parameter *par, bool isOnline)
{
    const linear_regression::Parameter *parameter = static_cast<const linear_regression::Parameter *>(par);
    ModelNormEq *rr = static_cast<ModelNormEq *>(r.get());
    MKL_INT nFeatures = (MKL_INT)x->getNumberOfColumns();
    MKL_INT nResponses = (MKL_INT)y->getNumberOfColumns();
    MKL_INT nRows = (MKL_INT)x->getNumberOfRows();
    MKL_INT nBetas = (MKL_INT)rr->getNumberOfBetas();
    MKL_INT nBetasIntercept = nBetas;
    if (parameter && !parameter->interceptFlag) { nBetasIntercept--; }

    /* Retrieve matrices X'*X and X'*Y from daal::algorithms::Model */
    NumericTable *xtxTable, *xtyTable;
    BlockDescriptor<interm> xtxBD, xtyBD;
    interm *xtx, *xty;
    getModelPartialSums<interm, cpu>(rr, nBetasIntercept, nResponses, readWrite, &xtxTable, xtxBD, &xtx, &xtyTable, xtyBD, &xty);

    /* Retrieve data associated with input tables */
    BlockDescriptor<interm> xBD;
    BlockDescriptor<interm> yBD;
    x->getBlockOfRows(0, nRows, readOnly, xBD);
    y->getBlockOfRows(0, nRows, readOnly, yBD);
    interm *dx = xBD.getBlockPtr();
    interm *dy = yBD.getBlockPtr();

    /* Here if daal::algorithms::Model is updated in course of online calculations, mklBetaCoeff = 1.0 */
    interm mklBetaCoeff = (isOnline ? 1.0 : 0.0);

    /* Calculate partial sums */
    updatePartialSums<interm, cpu>(&nFeatures, &nRows, &nBetasIntercept, dx, xtx, &nResponses, dy, xty, &mklBetaCoeff);

    x->releaseBlockOfRows(xBD);
    y->releaseBlockOfRows(yBD);
    releaseModelNormEqPartialSums<interm, cpu>(xtxTable, xtxBD, xtyTable, xtyBD);
}

template <typename interm, CpuType cpu>
void finalizeModelNormEq(SharedPtr<linear_regression::Model> &a, SharedPtr<linear_regression::Model> &r,
                   services::SharedPtr<services::KernelErrorCollection> &_errors)
{
    ModelNormEq *aa = static_cast<ModelNormEq *>(a.get());
    ModelNormEq *rr = static_cast<ModelNormEq *>(r.get());

    MKL_INT nBetas = (MKL_INT)rr->getNumberOfBetas();
    MKL_INT nResponses = (MKL_INT)rr->getNumberOfResponses();
    MKL_INT nBetasIntercept = nBetas;
    if (!rr->getInterceptFlag()) { nBetasIntercept--; }

    interm *betaBuffer = (interm *)daal::services::daal_malloc(nResponses * nBetas * sizeof(interm));
    if (!betaBuffer) { _errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* Retrieve matrices X'*X and X'*Y from daal::algorithms::Model */
    NumericTable *xtxTable, *xtyTable;
    BlockDescriptor<interm> xtxBD, xtyBD;
    interm *xtx, *xty;
    getModelPartialSums<interm, cpu>(aa, nBetas, nResponses, readOnly, &xtxTable, xtxBD, &xtx, &xtyTable, xtyBD, &xty);

    linreg_finalize<interm, cpu>(&nBetasIntercept, xtx, &nBetas, &nResponses, xty, &nBetas, betaBuffer, _errors);

    releaseModelNormEqPartialSums<interm, cpu>(xtxTable, xtxBD, xtyTable, xtyBD);

    services::SharedPtr<NumericTable> betaTable = rr->getBeta();
    BlockDescriptor<interm> betaBD;
    betaTable->getBlockOfRows(0, nResponses, writeOnly, betaBD);
    interm *beta = betaBD.getBlockPtr();

    if (nBetasIntercept == nBetas)
    {
        for(size_t i = 0; i < nResponses; i++)
        {
            for(size_t j = 1; j < nBetas; j++)
            {
                beta[i * nBetas + j] = betaBuffer[i * nBetas + j - 1];
            }
            beta[i * nBetas] = betaBuffer[i * nBetas + nBetas - 1];
        }
    }
    else
    {
        for(size_t i = 0; i < nResponses; i++)
        {
            for(size_t j = 0; j < nBetas - 1; j++)
            {
                beta[i * nBetas + j + 1] = betaBuffer[i * nBetasIntercept + j];
            }
            beta[i * nBetas] = 0.0;
        }
    }

    betaTable->releaseBlockOfRows(betaBD);
    daal::services::daal_free(betaBuffer);
}


template <typename interm, CpuType cpu>
void LinearRegressionTrainBatchKernel<interm, training::normEqDense, cpu>::compute(
            SharedPtr<NumericTable> &x, SharedPtr<NumericTable> &y, SharedPtr<linear_regression::Model> &r,
             const daal::algorithms::Parameter *par)
{
    bool isOnline = false;
    updatePartialModelNormEq<interm, cpu>(x, y, r, par, isOnline);
    finalizeModelNormEq<interm, cpu>(r, r, this->_errors);
}

template <typename interm, CpuType cpu>
void LinearRegressionTrainOnlineKernel<interm, training::normEqDense, cpu>::compute(
            SharedPtr<NumericTable> &x, SharedPtr<NumericTable> &y, SharedPtr<linear_regression::Model> &r,
             const daal::algorithms::Parameter *par)
{
    bool isOnline = true;
    updatePartialModelNormEq<interm, cpu>(x, y, r, par, isOnline);
}

template <typename interm, CpuType cpu>
void LinearRegressionTrainOnlineKernel<interm, training::normEqDense, cpu>::finalizeCompute(
            SharedPtr<linear_regression::Model> &a, SharedPtr<linear_regression::Model> &r,
            const daal::algorithms::Parameter *par)
{
    finalizeModelNormEq<interm, cpu>(a, r, this->_errors);
}

}
}
}
}
}

#endif
