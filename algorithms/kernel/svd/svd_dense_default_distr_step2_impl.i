/* file: svd_dense_default_distr_step2_impl.i */
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
//  Implementation of svds
//--
*/

#ifndef __SVD_KERNEL_DISTR_IMPL_I__
#define __SVD_KERNEL_DISTR_IMPL_I__

#include "service_memory.h"
#include "service_math.h"
#include "service_defines.h"
#include "service_numeric_table.h"
#include "service_error_handling.h"

#include "svd_dense_default_impl.i"

#include "threading.h"

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace internal
{

template <typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
Status SVDDistributedStep2Kernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable *const *a,
                                                                      const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    size_t i, j;
    svd::Parameter defaultParams;
    const svd::Parameter *svdPar = &defaultParams;

    if ( par != 0 )
    {
        svdPar = static_cast<const svd::Parameter *>(par);
    }

    const NumericTable *ntAux2_0 = a[0];
    NumericTable *ntSigma = const_cast<NumericTable *>(r[0]);

    size_t nBlocks = na;

    size_t n   = ntAux2_0->getNumberOfColumns();
    size_t nxb = n * nBlocks;

    WriteOnlyRows<algorithmFPType, cpu, NumericTable> sigmaBlock(ntSigma, 0, 1); /* Sigma [1][n]   */
    DAAL_CHECK_BLOCK_STATUS(sigmaBlock);
    algorithmFPType *Sigma = sigmaBlock.get();

    TArray<algorithmFPType, cpu> Aux2TPtr(n * nxb);
    TArray<algorithmFPType, cpu> VTPtr   (n * n  );
    TArray<algorithmFPType, cpu> Aux3TPtr(n * nxb);
    algorithmFPType *Aux2T = Aux2TPtr.get();
    algorithmFPType *VT = VTPtr.get();
    algorithmFPType *Aux3T = Aux3TPtr.get();

    DAAL_CHECK(Aux2T && VT && Aux3T, ErrorMemoryAllocationFailed);

    SafeStatus safeStat;

    daal::threader_for( nBlocks, nBlocks, [=,&safeStat](int k)
    {
        ReadRows<algorithmFPType, cpu, NumericTable> aux2Block(const_cast<NumericTable *>(a[k]), 0, n); /* Aux2  [nxb][n] */
        DAAL_CHECK_BLOCK_STATUS_THR(aux2Block);
        const algorithmFPType *Aux2 = aux2Block.get();

        for ( size_t i = 0 ; i < n ; i++ )
        {
            for ( size_t j = 0 ; j < n; j++ )
            {
                Aux2T[j * nxb + k * n + i] = Aux2[i * n + j];
            }
        }
    } );

    if(!safeStat)
        return safeStat.detach();

    {
        DAAL_INT ldAux2 = nxb;
        DAAL_INT ldAux3 = nxb;
        DAAL_INT ldV    = n;

        DAAL_INT ldR = n;
        DAAL_INT ldU = n;

        TArray<algorithmFPType, cpu> RPtr(n * ldR);
        TArray<algorithmFPType, cpu> UPtr(n * ldU);
        algorithmFPType *R = RPtr.get();
        algorithmFPType *U = UPtr.get();

        DAAL_CHECK(U && R, ErrorMemoryAllocationFailed);

        /* By some reason, there was this part in Sample */
        for (size_t i = 0; i < n * ldR; i++) { R[i] = 0.0; }

        // Rc = P*R
        const auto ecQr = compute_QR_on_one_node<algorithmFPType, cpu>( nxb, n, Aux2T, ldAux2, R, ldR );
        if(!ecQr)
            return ecQr;

        // Qn*R -> Qn*(U*Sigma*V) -> (Qn*U)*Sigma*V
        const auto ecSvd = compute_svd_on_one_node<algorithmFPType, cpu>( n, n, R, ldR, Sigma, U, ldU, VT, ldV );
        if(!ecSvd)
            return ecSvd;

        if (svdPar->leftSingularMatrix == requiredInPackedForm)
        {
            const auto ecGemm = compute_gemm_on_one_node<algorithmFPType, cpu>( nxb, n, Aux2T, ldAux2, U, ldU, Aux3T, ldAux3 );
            if(!ecGemm)
                return ecGemm;
        }
    }

    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        daal::threader_for( nBlocks, nBlocks, [=,&safeStat](int k)
        {
            WriteOnlyRows<algorithmFPType, cpu, NumericTable> aux3Block(r[2 + k], 0, n); /* Aux3  [nxb][n] */
            DAAL_CHECK_BLOCK_STATUS_THR(aux3Block);
            algorithmFPType *Aux3 = aux3Block.get();

            for ( size_t i = 0 ; i < n ; i++ )
            {
                for ( size_t j = 0 ; j < n; j++ )
                {
                    Aux3[i * n + j] = Aux3T[j * nxb + k * n + i];
                }
            }
        } );
        if(!safeStat)
            return safeStat.detach();
    }

    if (svdPar->rightSingularMatrix == requiredInPackedForm)
    {
        WriteOnlyRows<algorithmFPType, cpu, NumericTable> VBlock(r[1], 0, n); /* V[n][n] */
        DAAL_CHECK_BLOCK_STATUS(VBlock);
        algorithmFPType *V = VBlock.get();

        for ( i = 0 ; i < n ; i++ )
        {
            for ( j = 0 ; j < n; j++ )
            {
                V[i + j * n] = VT[i * n + j];
            }
        }
    }

    return Status();
}

} // namespace daal::internal
}
}
} // namespace daal

#endif
