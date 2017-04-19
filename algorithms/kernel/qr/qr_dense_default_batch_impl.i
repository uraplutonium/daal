/* file: qr_dense_default_batch_impl.i */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of qrs
//--
*/

#ifndef __QR_KERNEL_BATCH_IMPL_I__
#define __QR_KERNEL_BATCH_IMPL_I__

#include "service_lapack.h"
#include "service_memory.h"
#include "service_math.h"
#include "service_defines.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

#include "qr_dense_default_impl.i"

#include "threading.h"

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace qr
{
namespace internal
{

#include "qr_dense_default_pcl_impl.i"

/**
 *  \brief Kernel for QR QR calculation
 */
template <typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
Status QRBatchKernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable *const *a,
                                                          const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    NumericTable *ntAi = const_cast<NumericTable *>(a[0]);

    size_t  n = ntAi->getNumberOfColumns();
    size_t  m = ntAi->getNumberOfRows();
    size_t  t = threader_get_threads_number();

    if(m >= 2*n)
    {
        if((m > n*t) && (n>10) && (!(n>=200 && m<=100000)))
        {
            return QRBatchKernel<algorithmFPType, method, cpu>::compute_pcl( na, a, nr, r, par);
        }
        else
        {
            return QRBatchKernel<algorithmFPType, method, cpu>::compute_thr( na, a, nr, r, par);
        }
    }
    else
    {
        return QRBatchKernel<algorithmFPType, method, cpu>::compute_seq( na, a, nr, r, par);
    }
    return Status();
}


template <typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
Status QRBatchKernel<algorithmFPType, method, cpu>::compute_seq(const size_t na, const NumericTable *const *a,
                                                                             const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    size_t i, j;
    qr::Parameter defaultParams;

    NumericTable *ntAi = const_cast<NumericTable *>(a[0]);
    NumericTable *ntRi = const_cast<NumericTable *>(r[1]);

    size_t  n   = ntAi->getNumberOfColumns();
    size_t  m   = ntAi->getNumberOfRows();

    DAAL_INT ldAi = m;
    DAAL_INT ldRi = n;

    TArray<algorithmFPType, cpu> QiTPtr(n * m);
    algorithmFPType *QiT = QiTPtr.get();
    DAAL_CHECK(QiT, ErrorMemoryAllocationFailed);

    TArray<algorithmFPType, cpu> RiTPtr(n * n);
    algorithmFPType *RiT = RiTPtr.get();
    DAAL_CHECK(RiT, ErrorMemoryAllocationFailed);

    WriteRows<algorithmFPType, cpu, NumericTable> aiBlock(ntAi, 0, m);
    algorithmFPType *Ai = aiBlock.get();
    DAAL_CHECK(Ai, ErrorMemoryAllocationFailed);

    WriteOnlyRows<algorithmFPType, cpu, NumericTable> riBlock(ntRi, 0, n);  /* Ri = Ri [n][n] */
    algorithmFPType *Ri = riBlock.get();
    DAAL_CHECK(Ri, ErrorMemoryAllocationFailed);

    for ( i = 0 ; i < n ; i++ )
    {
        for ( j = 0 ; j < m; j++ )
        {
            QiT[i * m + j] = Ai[i + j * n];
        }
    }

    compute_QR_on_one_node<algorithmFPType, cpu>( m, n, QiT, ldAi, RiT, ldRi );

    NumericTable *ntQi = const_cast<NumericTable *>(r[0]);
    WriteOnlyRows<algorithmFPType, cpu, NumericTable> qiBlock(ntQi, 0, m); /* Qi = Qin[m][n] */
    algorithmFPType *Qi = qiBlock.get();
    DAAL_CHECK(Qi, ErrorMemoryAllocationFailed);

    for ( i = 0 ; i < n ; i++ )
    {
        for ( j = 0 ; j < m; j++ )
        {
            Qi[i + j * n] = QiT[i * m + j];
        }
    }

    for ( i = 0 ; i < n ; i++ )
    {
        for ( j = 0 ; j <= i; j++ )
        {
            Ri[i + j * n] = RiT[i * n + j];
        }
        for (     ; j < n; j++ )
        {
            Ri[i + j * n] = 0.0;
        }
    }

    return Status();
}


/* Max number of blocks depending on arch */
#if( __CPUID__(DAAL_CPU) >= __avx512_mic__ )
    #define DEF_MAX_BLOCKS 256
#else
    #define DEF_MAX_BLOCKS 128
#endif


/*
    Algorithm for parallel QR computation:
    -------------------------------------
    A[m,n] input matrix to be factorized by output Q[m,n] and R[n,n]

    1st step:
    Split A[m,n] matrix to 'b' blocks -> a1[m1,n],a2[m2,]...ab[mb,n]
    Compute QR decomposition for each block in threads:
                               a1[m1,n] -> q1[m1,n] , r1[n,n] ... ab[mb,n] -> qb[mb,n] and rb[n,n]

    2nd step:
    Concatenate r1[n,n] , r2[n,n] ... rb[n,n] into one matrix B[n*b,n]
    Compute QR decomposition for B[n*b,n] -> P[n*b,n] , R[n,n]. R - resulted matrix

    3rd step: Split P[n*b,n] matrix to 'b' blocks -> p1[n,n],p2[n,n]...pb[n,n]
    Multiply by q1..qb matrices from 1st step using GEMM for each block in threads:
                               q1[m1,n] * p1[n,n] -> q'1[m1,n] ... qb[mb,n] * pb[n,n] -> q'b[mb,n]
    Concatenate q'1[m1,n]...q'b[mb,n] into one resulted Q[m,n]  matrix.

    Notice: before and after QR and GEMM computations matrices need to be transposed
*/
template <typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
Status QRBatchKernel<algorithmFPType, method, cpu>::compute_thr(const size_t na, const NumericTable *const *a,
                                                              const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    qr::Parameter defaultParams;

    NumericTable *ntA_input  = const_cast<NumericTable *>(a[0]);
    NumericTable *ntQ_output = const_cast<NumericTable *>(r[0]);
    NumericTable *ntR_output = const_cast<NumericTable *>(r[1]);

    size_t  n = ntA_input->getNumberOfColumns();
    size_t  m = ntA_input->getNumberOfRows();

    DAAL_INT ldA_input = m;
    DAAL_INT ldR_output = n;

    size_t rows = m;
    size_t cols = n;

    /* Getting real pointers to input and output arrays */
    WriteRows<algorithmFPType, cpu, NumericTable> bkA_input(ntA_input, 0, m);
    algorithmFPType *A_input = bkA_input.get();
    DAAL_CHECK(A_input, ErrorMemoryAllocationFailed);

    WriteOnlyRows<algorithmFPType, cpu, NumericTable> bkQ_output(ntQ_output, 0, m);
    algorithmFPType *Q_output = bkQ_output.get();
    DAAL_CHECK(Q_output, ErrorMemoryAllocationFailed);

    WriteOnlyRows<algorithmFPType, cpu, NumericTable> bkR_output(ntR_output, 0, n);
    algorithmFPType *R_output = bkR_output.get();
    DAAL_CHECK(R_output, ErrorMemoryAllocationFailed);

    /* Block size calculation (empirical) */
    int bshift = (rows <= 10000 )?11:12;
    int bsize  = ((rows*cols)>>bshift) & (~0xf);
    bsize = (bsize < 200)?200:bsize;

    /*
    Calculate sizes:
    blocks     = number of blocks,
    brows      = number of rows in blocks,
    brows_last = number of rows in last block,
    */
    size_t def_min_brows = rows/DEF_MAX_BLOCKS; // min block size
    size_t brows      = ( rows > bsize )? bsize:rows; /* brows cannot be less than rows */
    brows             = ( brows < cols )? cols:brows; /* brows cannot be less than cols */
    brows             = (brows < def_min_brows)?def_min_brows:brows; /* brows cannot be less than n/DEF_MAX_BLOCKS */
    size_t blocks     = rows / brows;

    size_t brows_last = brows + (rows - blocks * brows); /* last block is generally biggest */

    size_t len = blocks * n * n;
    TArray<algorithmFPType, cpu> R_buffPtr(len);
    algorithmFPType *R_buff = R_buffPtr.get();
    DAAL_CHECK(R_buff, ErrorMemoryAllocationFailed);

    TArray<algorithmFPType, cpu> RT_buffPtr(len);
    algorithmFPType *RT_buff = R_buffPtr.get();
    DAAL_CHECK(RT_buff, ErrorMemoryAllocationFailed);

    bool isMemoryAllocationFailed = false;

    /* Step1: calculate QR on local nodes */
    /* ================================== */
    daal::threader_for( blocks, blocks, [=, &A_input, &RT_buff, &isMemoryAllocationFailed](int k)
    {
        algorithmFPType *A_block = A_input  + k * brows * cols;
        algorithmFPType *Q_block = Q_output + k * brows * cols;

        /* Last block size brows_last (generally larger than other blocks) */
        size_t brows_local = (k==(blocks-1))?brows_last:brows;
        size_t cols_local  = cols;

        algorithmFPType *QT_local = service_scalable_malloc<algorithmFPType, cpu>(cols_local * brows_local);
        algorithmFPType *RT_local = service_scalable_malloc<algorithmFPType, cpu>(cols_local * cols_local);

        if( (QT_local) && (RT_local) )
        {
            /* Get transposed Q from A */
            for ( int i = 0 ; i < cols_local ; i++ ) {
            PRAGMA_IVDEP
                for ( int j = 0 ; j < brows_local; j++ ) {
                    QT_local[i*brows_local+j] = A_block[i+j*cols_local]; } }

            /* Call QR on local nodes */
            compute_QR_on_one_node_seq<algorithmFPType, cpu>( brows_local, cols_local, QT_local, brows_local, RT_local, cols_local ); // DK: add tls with errorCodes

            /* Transpose Q */
            for ( int i = 0 ; i < cols_local ; i++ ) {
            PRAGMA_IVDEP
                for ( int j = 0 ; j < brows_local; j++ ) {
                    Q_block[i+j*cols_local] = QT_local[i*brows_local+j]; } }

            /* Transpose R and zero lower values */
            for ( int i = 0 ; i < cols_local ; i++ ) {
                int j;
                PRAGMA_IVDEP
                       for ( j = 0 ; j <= i; j++ ) {
                           RT_buff[k*cols_local + i*cols_local*blocks +  j] = RT_local[i*cols_local+j];}
                PRAGMA_IVDEP
                       for (     ; j < cols_local; j++ ) {
                           RT_buff[k*cols_local + i*cols_local*blocks +  j] = 0.0; } }
        }
        else /* if( (QT_local) && (RT_local) ) */
        {
            isMemoryAllocationFailed = true;
        }

        if(QT_local) { service_scalable_free<algorithmFPType, cpu>(QT_local); }
        if(RT_local) { service_scalable_free<algorithmFPType, cpu>(RT_local); }

    } );

    /* Step2: calculate QR on master node for resulted RB[blocks*n*n] */
    /* ============================================================== */

    /* Call QR on master node for RB */
    Status s = compute_QR_on_one_node_seq<algorithmFPType, cpu>( cols * blocks, cols, RT_buff, cols * blocks, R_buff, cols );

    /* Transpose R */
    for ( int i = 0 ; i < cols ; i++ ) {
    PRAGMA_IVDEP
        for ( int j = 0 ; j < cols; j++ ) {
            R_output[i + j * cols] = R_buff[i * cols + j]; } }


    /* Step3: calculate Q by merging Q*RB */
    /* ================================== */

    daal::threader_for( blocks, blocks, [=, &R_buff, &Q_output, &isMemoryAllocationFailed](int k)
    {
        algorithmFPType *Q_block         = Q_output  + k * brows * cols;
        algorithmFPType *R_block         = R_buff    + k * cols  * cols;

        /* Last block size brows_last (generally larger than other blocks) */
        size_t brows_local = (k==(blocks-1))?brows_last:brows;
        size_t cols_local  = cols;

        algorithmFPType *QT_local        = service_scalable_malloc<algorithmFPType, cpu>(cols_local * brows_local);
        algorithmFPType *QT_result_local = service_scalable_malloc<algorithmFPType, cpu>(cols_local * brows_local);
        algorithmFPType *RT_local        = service_scalable_malloc<algorithmFPType, cpu>(cols_local * cols_local);

        if( (QT_local) && (QT_result_local) && (RT_local))
        {
            /* Transpose RB */
            for ( int i = 0 ; i < cols_local ; i++ ) {
            PRAGMA_IVDEP
                for ( int j = 0 ; j < cols_local; j++ ) {
                    RT_local[j*cols_local + i] = RT_buff[j*cols_local*blocks + k*cols_local + i]; } }

            /* Transpose Q to QT */
            for ( int i = 0 ; i < cols_local ; i++ ) {
            PRAGMA_IVDEP
                for ( int j = 0 ; j < brows_local; j++ ) {
                    QT_local[i*brows_local + j] = Q_block[i + j*cols_local]; } }

            /* Call GEMMs to multiply Q*R */
            compute_gemm_on_one_node_seq<algorithmFPType, cpu>( brows_local, cols_local, QT_local, brows_local, RT_local, cols_local, QT_result_local,
                                                                brows_local ); // DK: add tls with errorCodes

            /* Transpose result Q */
            for ( int i = 0 ; i < cols_local ; i++ ) {
            PRAGMA_IVDEP
                for ( int j = 0 ; j < brows_local; j++ ) {
                    Q_block[i + j*cols_local] = QT_result_local[i*brows_local + j]; } }
        }
        else /* if( (QT_local) && (QT_result_local) && (RT_local)) */
        {
            isMemoryAllocationFailed = true;
        }

        if(QT_local) {        service_scalable_free<algorithmFPType, cpu>(QT_local);        }
        if(RT_local) {        service_scalable_free<algorithmFPType, cpu>(RT_local);        }
        if(QT_result_local) { service_scalable_free<algorithmFPType, cpu>(QT_result_local); }

    } );

    if(isMemoryAllocationFailed) { s.add(ErrorMemoryAllocationFailed); }

    return s;
}

template <typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
Status QRBatchKernel<algorithmFPType, method, cpu>::compute_pcl(const size_t na, const NumericTable *const *a,
                                                                   const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    size_t i, j;
    qr::Parameter defaultParams;

    NumericTable *ntAi = const_cast<NumericTable *>(a[0]);
    NumericTable *ntQi = const_cast<NumericTable *>(r[0]);
    NumericTable *ntRi = const_cast<NumericTable *>(r[1]);

    size_t  n   = ntAi->getNumberOfColumns();
    size_t  m   = ntAi->getNumberOfRows();

    DAAL_INT ldAi = m;
    DAAL_INT ldRi = n;

    BlockDescriptor<algorithmFPType> aiBlock;
    BlockDescriptor<algorithmFPType> qiBlock;
    BlockDescriptor<algorithmFPType> riBlock;

    ntAi->getBlockOfRows( 0, m, readOnly,  aiBlock);
    ntQi->getBlockOfRows( 0, m, writeOnly, qiBlock);
    ntRi->getBlockOfRows( 0, n, writeOnly, riBlock);

    algorithmFPType *Ai = aiBlock.getBlockPtr();
    algorithmFPType *Qi = qiBlock.getBlockPtr();
    algorithmFPType *Ri = riBlock.getBlockPtr();

    Status s;
    services::ErrorID st = (services::ErrorID)qr_pcl<algorithmFPType,cpu>(Ai, m, n, Qi, Ri );
    if(st) s.add(st);

    ntAi->releaseBlockOfRows( aiBlock );
    ntQi->releaseBlockOfRows( qiBlock );
    ntRi->releaseBlockOfRows( riBlock );

    return s;
}


} // namespace daal::internal
}
}
} // namespace daal

#endif
