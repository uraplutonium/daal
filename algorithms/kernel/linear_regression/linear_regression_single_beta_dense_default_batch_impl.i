/* file: linear_regression_single_beta_dense_default_batch_impl.i */
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

#ifndef __LINEAR_REGRESSION_SINGLE_BETA_DEFAULT_IMPL_I__
#define __LINEAR_REGRESSION_SINGLE_BETA_DEFAULT_IMPL_I__

#include "service_memory.h"
#include "service_math.h"
#include "linear_regression_train_dense_normeq_impl.i"
#include "service_lapack.h"
#include "service_numeric_table.h"
#include "svd/svd_dense_default_kernel.h"

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
namespace single_beta
{
namespace internal
{

template <typename algorithmFPType, CpuType cpu>
void getXtX(MKL_INT p, MKL_INT n, const algorithmFPType *x, MKL_INT nBeta,
    algorithmFPType *xtx, algorithmFPType mklBetaCoeff)
{
    if (p < nBeta)
    {
        /* Here if intercept term will be calculated */
        linear_regression::training::internal::linreg_xtx_comp_b0<algorithmFPType, cpu>(&p, &n,
            const_cast<algorithmFPType*>(x), &p, xtx, &nBeta, &mklBetaCoeff);
    }
    else
    {
        /* Here if intercept term won't be calculated */
        linear_regression::training::internal::linreg_xtx_comp<algorithmFPType, cpu>(&p, &n,
            const_cast<algorithmFPType*>(x), &p, xtx, &nBeta, &mklBetaCoeff);
    }
}

template <typename algorithmFPType, CpuType cpu>
bool getInverseXtX(size_t nBetas, algorithmFPType *xtxInv)
{
    char uplo = 'U';
    MKL_INT info = 0;
    MKL_INT nBeta(nBetas);
    Lapack<algorithmFPType, cpu>::xpotrf(&uplo, &nBeta, xtxInv, &nBeta, &info);
    if (info != 0)
        return false;

    Lapack<algorithmFPType, cpu>::xpotri(&uplo, &nBeta, xtxInv, &nBeta, &info);
    return (info == 0);
}

template<Method method, typename algorithmFPType, CpuType cpu>
bool SingleBetaKernel<method, algorithmFPType, cpu>::getPseudoInverseXtX(NumericTable& xtx)
{
    //SVD decomposition: xtx = U*S*Vt
    const size_t n = xtx.getNumberOfColumns();
    const NumericTable *tbl = &xtx;
    const NumericTable *const *svdInputs = &tbl;

    SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > u(new HomogenNumericTableCPU<algorithmFPType, cpu>(n, n));
    SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > s(new HomogenNumericTableCPU<algorithmFPType, cpu>(n, 1));
    SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > v(new HomogenNumericTableCPU<algorithmFPType, cpu>(n, n));

    NumericTable *svdResults[3] = { s.get(), u.get(), v.get() };
    svd::Parameter params;
    daal::algorithms::svd::internal::SVDBatchKernel<algorithmFPType, svd::defaultDense, cpu> svdKernel;
    svdKernel.compute(1, svdInputs, 3, svdResults, &params);

    if (svdKernel.getErrorCollection()->size() > 0)
    {
        this->_errors->add(svdKernel.getErrorCollection());
        return false;
    }

    //calculate pinv(xtx) = V*inv(S)*Ut

    //calculate inv(S)
    WriteRows<algorithmFPType, cpu, NumericTable> sBD(*s.get(), 1);
    algorithmFPType* ps = sBD.get();
    for(size_t i = 0; i < n; ++i)
        ps[i] = 1/ps[i];

    //calculate V = V*inv(S)
    WriteRows<algorithmFPType, cpu, NumericTable> vBD(*v.get(), n);
    algorithmFPType* pv = vBD.get();
    for(size_t i = 0; i < n; ++i)
    {
        algorithmFPType* pRow = pv + n*i;
        for(size_t j = 0; j < n; ++j)
            pRow[j] *= ps[j];
    }

    //calculate xtx = V*Ut
    ReadRows<algorithmFPType, cpu, NumericTable> uBD(*u.get(), n);
    const algorithmFPType* pu = uBD.get();

    WriteRows<algorithmFPType, cpu, NumericTable> xtxBD(xtx, n);
    algorithmFPType* pXtX = xtxBD.get();

    char transa = 'T';
    char transb = 'N';
    algorithmFPType alpha = 1.0;
    algorithmFPType beta = 0;
    MKL_INT* sz = (MKL_INT*)&n;
    Blas<algorithmFPType, cpu>::xgemm(&transa, &transb, sz, sz, sz, &alpha, pv, sz, const_cast<algorithmFPType*>(pu), sz, &beta, pXtX, sz);
    return true;
}

template<Method method, typename algorithmFPType, CpuType cpu>
void SingleBetaKernel<method, algorithmFPType, cpu>::computeTestStatistics(
    const NumericTable* betas, const algorithmFPType* v,
    algorithmFPType alpha, algorithmFPType accuracyThreshold, SingleBetaOutput& out)
{
    const size_t nBeta = betas->getNumberOfColumns();
    const size_t nResponse = betas->getNumberOfRows();

    ReadRows<algorithmFPType, cpu, NumericTable> betasBD(*betas, nResponse);
    const algorithmFPType* beta = betasBD.get();

    WriteRows<algorithmFPType, cpu, NumericTable> zScoreBD(*out.zScore, nResponse);
    algorithmFPType* zScore = zScoreBD.get();

    WriteRows<algorithmFPType, cpu, NumericTable> confidenceIntervalsBD(*out.confidenceIntervals, nResponse);
    algorithmFPType* confInt = confidenceIntervalsBD.get();

    ReadRows<algorithmFPType, cpu, NumericTable> varianceBD(*out.variance, 1);
    const algorithmFPType* variance = varianceBD.get();

    const algorithmFPType z_1_alpha = sCdfNormInv<cpu>(1 - alpha);
    for(size_t i = 0; i < nResponse; ++i)
    {
        const algorithmFPType sigma = z_1_alpha*sSqrt<cpu>(variance[i]);
        for(size_t j = 0; j < nBeta; ++j)
        {
            algorithmFPType vsigma = sigma*v[j];
            if(vsigma < accuracyThreshold)
                vsigma = accuracyThreshold;
            const algorithmFPType betaVal = beta[i*nBeta + j];
            zScore[i*nBeta + j] = betaVal/vsigma;
            confInt[i*2*nBeta + 2*j] = betaVal - vsigma;
            confInt[i*2*nBeta + 2*j + 1] = betaVal + vsigma;
        }
    }
}

template<Method method, typename algorithmFPType, CpuType cpu>
bool SingleBetaKernel<method, algorithmFPType, cpu>::computeInverseXtX(const NumericTable* xtx, bool bModelNe, NumericTable* xtxInv)
{
    const auto nBetas = xtx->getNumberOfColumns();

    ReadRows<algorithmFPType, cpu, NumericTable> xtxBD(*xtx, nBetas);
    const algorithmFPType *pXtX = xtxBD.get();

    {
        WriteRows<algorithmFPType, cpu, NumericTable> xtxInvBD(*xtxInv, nBetas);
        algorithmFPType *pXtXInv = xtxInvBD.get();

        if(bModelNe)
        {
            //xtx contains Cholessky decompositon matrix L
            //calculate its inverse

            //Copy xtx to xtxInv
            const auto dataSize = nBetas * nBetas * sizeof(algorithmFPType);
            services::daal_memcpy_s(pXtXInv, dataSize, pXtX, dataSize);

            char uplo = 'U';
            MKL_INT nBeta(nBetas);
            MKL_INT info = 0;
            Lapack<algorithmFPType, cpu>::xpotri(&uplo, &nBeta, pXtXInv, &nBeta, &info);
            if(info == 0)
                return true;

            //Calculate Xt*X to find pseudo inverse
            // SYRK parameters
            char trans = 'T';
            algorithmFPType mklAlphaCoeff = 1.0;
            algorithmFPType mklBetaCoeff = 0.0;
            Blas<algorithmFPType, cpu>::xsyrk(&uplo, &trans, &nBeta, &nBeta, &mklAlphaCoeff,
                const_cast<algorithmFPType* >(pXtX), &nBeta, &mklBetaCoeff, pXtXInv, &nBeta);
        }
        else
        {
            //xtx contains R matrix from QR
            //restore Xt*X = Rt*R
            char uplo = 'U';
            char trans = 'N';
            MKL_INT nBeta(nBetas);
            algorithmFPType mklAlphaCoeff = 1.0;
            algorithmFPType mklBetaCoeff = 0.0;
            Blas<algorithmFPType, cpu>::xsyrk(&uplo, &trans, &nBeta, &nBeta, &mklAlphaCoeff,
                const_cast<algorithmFPType* >(pXtX), &nBeta, &mklBetaCoeff, pXtXInv, &nBeta);
            if(getInverseXtX<algorithmFPType, cpu>(nBetas, pXtXInv))
                return true;
            //restore Xt*X = Rt*R again
            Blas<algorithmFPType, cpu>::xsyrk(&uplo, &trans, &nBeta, &nBeta, &mklAlphaCoeff,
                const_cast<algorithmFPType* >(pXtX), &nBeta, &mklBetaCoeff, pXtXInv, &nBeta);
        }
    }
    return getPseudoInverseXtX(*xtxInv);
}

template<Method method, typename algorithmFPType, CpuType cpu>
void SingleBetaKernel<method, algorithmFPType, cpu>::compute(
    const NumericTable* y, const NumericTable* z, size_t p,
    const NumericTable* betas, const NumericTable* xtx, bool bModelNe,
    algorithmFPType accuracyThreshold, algorithmFPType alpha, SingleBetaOutput& out)
{
    computeRmsVariance(y, z, p, out.rms, out.variance);

    // Calculate inverse(Xt*X)
    if(!computeInverseXtX(xtx, bModelNe, out.inverseOfXtX))
    {
        this->_errors->add(services::ErrorLinearRegressionInternal);
        return;
    }

    const auto nBetas = xtx->getNumberOfColumns();
    //Compute vector V (sqrt of inverse (Xt*X) diagonal elements)
    SmartPtr<cpu> aDiagElem(nBetas * sizeof(algorithmFPType));
    algorithmFPType* v = (algorithmFPType *)aDiagElem.get();
    if(!v)
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed);
        return;
    }

    {
        ReadRows<algorithmFPType, cpu, NumericTable> xtxInvBD(*out.inverseOfXtX, nBetas);
        const algorithmFPType *xtxInv = xtxInvBD.get();

        const algorithmFPType* pXtxInv = xtxInv;
        algorithmFPType* pV = v;
        for(auto i = 0; i < nBetas; ++i, pXtxInv += nBetas + 1, ++pV)
            *pV = (*pXtxInv < 0 ? sSqrt<cpu>(-*pXtxInv) : sSqrt<cpu>(*pXtxInv));

        //Compute beta variance-covariance matrices
        ReadRows<algorithmFPType, cpu, NumericTable> varianceBD(*out.variance, 1);
        const algorithmFPType* variance = varianceBD.get();

        const auto k = y->getNumberOfColumns();
        for(auto i = 0; i < k; ++i)
        {
            WriteRows<algorithmFPType, cpu, NumericTable> betaCovBD(*out.betaCovariances[i], nBetas);
            algorithmFPType *betaCov = betaCovBD.get();
            const algorithmFPType sigmaSq = variance[i];
            for(auto j = 0; j < nBetas*nBetas; ++j)
                betaCov[j] = xtxInv[j]*sigmaSq;
        }
    }
    computeTestStatistics(betas, v, alpha, accuracyThreshold, out);
}

template<Method method, typename algorithmFPType, CpuType cpu>
void SingleBetaKernel<method, algorithmFPType, cpu>::computeRmsVariance(const NumericTable* y,
    const NumericTable* z, size_t p, NumericTable* rms, NumericTable* variance)
{
    const auto n = y->getNumberOfRows();
    const auto k = y->getNumberOfColumns();

    ReadRows<algorithmFPType, cpu, NumericTable> yBD(*y, n);
    const algorithmFPType* py = yBD.get();

    ReadRows<algorithmFPType, cpu, NumericTable> zBD(*z, n);
    const algorithmFPType* pz = zBD.get();

    WriteRows<algorithmFPType, cpu, NumericTable> rmsBD(*rms, 1);
    algorithmFPType *pRms = rmsBD.get();

    WriteRows<algorithmFPType, cpu, NumericTable> varBD(*variance, 1);
    algorithmFPType *pVar = varBD.get();

    for(size_t j = 0; j < k; pRms[j] = 0, pVar[j] = 0, ++j);

    for(size_t i = 0; i < n; ++i)
    {
        for(size_t j = 0; j < k; ++j)
            pRms[j] += (py[i*k + j] - pz[i*k + j])*(py[i*k + j] - pz[i*k + j]);
    }

    const algorithmFPType div1 = 1. / n;
    const algorithmFPType div2 = 1. / (n - p - 1);
    for(size_t j = 0; j < k; ++j)
    {
        pVar[j] = div2*pRms[j];
        pRms[j] = div1*sSqrt<cpu>(pRms[j]);
    }
}

}
}
}
}
}
}

#endif
