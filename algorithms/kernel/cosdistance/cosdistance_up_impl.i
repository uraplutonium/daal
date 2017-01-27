/* file: cosdistance_up_impl.i */
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
//  Implementation of cosine distance for result in upper triangular layout.
//--
*/
#include "service_defines.h"
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace cosine_distance
{
namespace internal
{

template <typename algorithmFPType, CpuType cpu>
int cosDistanceUpperPacked(const NumericTable* xTable, NumericTable* rTable)
{
    size_t p = xTable->getNumberOfColumns();      /* Dimension of input feature vector */
    size_t n = xTable->getNumberOfRows();         /* Number of input feature vectors   */

    algorithmFPType* r;
    PackedArrayMicroTable<algorithmFPType, readWrite, cpu> rPackedMicroTable( rTable );
    rPackedMicroTable.getPackedArray(&r);

    size_t nBlocks = n / blockSizeDefault;
    nBlocks += (nBlocks*blockSizeDefault != n);

    /* compute major diagonal blocks of the distance matrix */
    daal::threader_for(nBlocks, nBlocks, [ = ](size_t k1)
    {
        DAAL_INT blockSize1 = blockSizeDefault;
        if (k1 == nBlocks - 1)
        {
            blockSize1 = n - k1*blockSizeDefault;
        }

        /* read access to blockSize1 rows in input dataset at k1*blockSizeDefault*p row */
        algorithmFPType* x;
        BlockMicroTable<algorithmFPType, readOnly, cpu> xBlock( xTable );
        xBlock.getBlockOfRows(k1*blockSizeDefault, blockSize1, &x);

        algorithmFPType buf[blockSizeDefault * blockSizeDefault];
        algorithmFPType alpha = 1.0, beta = 0.0;
        char transa = 'T', transb = 'N';
        DAAL_INT m = blockSize1, k = p, nn = blockSize1;
        DAAL_INT lda = k, ldb = p, ldc = m;

        Blas<algorithmFPType, cpu>::xxgemm(&transa, &transb, &m, &nn, &k, &alpha, x, &lda, x, &ldb, &beta, buf, &ldc);

        /* compute inverse of sqrt of gemm result and save for use in computation off-diagonal blocks */
        PRAGMA_SIMD_ASSERT
        for (int i = 0; i < blockSize1; i++)
        {
            if (buf[i * blockSize1 + i] > (algorithmFPType)0.0)
            {
                buf[i * blockSize1 + i] = (algorithmFPType)1.0 / daal::internal::Math<algorithmFPType,cpu>::sSqrt(buf[i*blockSize1 + i]);
            }
        }

        /* compute cosine distance for k1 block of rows in the input dataset */
        for (int i = 0; i < blockSize1; i++)
        {
            PRAGMA_SIMD_ASSERT
            for (int j = i + 1; j < blockSize1; j++)
            {
                buf[i * blockSize1 + j] = 1.0 - buf[i * blockSize1 + j] * buf[i * blockSize1 + i] * buf[j * blockSize1 + j];
            }
        }

        /* unpack the results into user's memory */
        size_t shift1 = k1*blockSizeDefault;
        /* beginning of the memory to copy in starts at n * shift1 - shift1 * ( shift1 - 1 ) / 2 */
        algorithmFPType* rr = r + ( n * shift1 - shift1 * ( shift1 - 1 ) / 2 );
        for (size_t idx = shift1, i = 0; i < blockSize1; idx++, i++)
        {
            PRAGMA_SIMD_ASSERT
            for (size_t j = i; j < blockSize1; j++)
            {
                rr[j-i] = buf[i*blockSize1 + j];
            }

            /* the next "row" in the user memory is shifted by n-idx positions vs the previous one; idx is incremented for each row */
            rr += (n-idx);
        }

        xBlock.release();
    } );

    /* compute off-diagonal blocks of the distance matrix */
    daal::threader_for(nBlocks, nBlocks, [ = ](size_t k1)
    {
        DAAL_INT blockSize1 = blockSizeDefault;
        if (k1 == nBlocks - 1)
        {
            blockSize1 = n - k1*blockSizeDefault;
        }

        size_t shift1 = k1 * blockSizeDefault;
        /* read access to blockSize1 rows in input dataset at k1*blockSizeDefault row */
        algorithmFPType* x1;
        BlockMicroTable<algorithmFPType, readOnly, cpu> xBlock1( xTable );
        xBlock1.getBlockOfRows(shift1, blockSize1, &x1);

        /* compute upper triangular of the distance matrix */
        daal::threader_for(nBlocks-k1-1, nBlocks-k1-1, [ = ](size_t k3)
        {
            DAAL_INT blockSize2 = blockSizeDefault;
            size_t k2 = k3+k1+1;
            size_t nl = n;

            if (k2 == nBlocks - 1)
            {
                blockSize2 = nl - k2*blockSizeDefault;
            }

            /* extract diagonal elements of k1-th diagonal block of the matrix */
            size_t shift1l = shift1, ns = nl-shift1l;
            algorithmFPType *rr,  diag1[blockSizeDefault];

            /* shift to the first diagonal element of the k1 block */
            rr = r + ( nl * shift1l - shift1l * ( shift1l - 1 ) / 2 );
            for (size_t idx = 0, i = 0; i < blockSize1; idx += (ns-i), i++)
            {
                /* the next diagonal element is n-k1 * blockSize - i elements from the current in the packed array, i is row index */
                diag1[i] = rr[idx];
            }

            /* extract diagonal elements of k2-th diagonal block of the matrix */
            size_t shift2 = k2 * blockSizeDefault;
            ns = nl - shift2;
            algorithmFPType diag2[blockSizeDefault], buf[blockSizeDefault * blockSizeDefault];
            /* shift to the first diagonal element of the k2 block */
            rr = r + ( nl * shift2 - shift2 * ( shift2 - 1 ) / 2 );
            for (size_t idx = 0, i = 0; i < blockSize2; idx += (ns-i), i++)
            {
                /* the next diagonal element is n-k2 * blockSize - i elements from the current in the packed array, i is row index */
                diag2[i] = rr[idx];
            }

            /* read access to blockSize1 rows in input dataset at k1*blockSizeDefault row */
            algorithmFPType* x2;
            BlockMicroTable<algorithmFPType, readOnly, cpu> xBlock2( xTable );
            xBlock2.getBlockOfRows(shift2, blockSize2, &x2);

            algorithmFPType alpha = 1.0, beta = 0.0;
            char transa = 'T', transb = 'N';
            DAAL_INT m = blockSize2, k = p, nn = blockSize1;
            DAAL_INT lda = k, ldb = p, ldc = m;

            /* compute the distance between k1 and k2 blocks of rows in the input dataset */
            Blas<algorithmFPType, cpu>::xxgemm(&transa, &transb, &m, &nn, &k, &alpha, x2, &lda, x1, &ldb, &beta, buf, &ldc);

            for (size_t i = 0; i < blockSize1; i++)
            {
                PRAGMA_SIMD_ASSERT
                for (size_t j = 0; j < blockSize2; j++)
                {
                    buf[i * blockSize2 + j] = 1.0 - buf[i * blockSize2 + j] * diag1[i] * diag2[j];
                }
            }

            /* copy the results into user memory */
            ns = nl - shift1l;
            /* The memory to copy in starts at the position n * shiftl - shiftl * ( shiftl - 1 ) / 2, shift1 = k1 * blockSize */
            rr  = r + ( nl * shift1l - shift1l * ( shift1l - 1 ) / 2 );
            for (size_t idx1 = 0, idx2 = (k2-k1) * blockSizeDefault, i = 0; i < blockSize1; idx1+=(ns-i), idx2--, i++)
            {
                /* the next row is shifted by a number defined by complex expression */
                size_t idx = idx1+idx2;
                PRAGMA_SIMD_ASSERT
                for (size_t j = 0; j < blockSize2; j++)
                {
                    rr[idx+j] = buf[i*blockSize2 + j];
                }
            }

            xBlock2.release();
        } );

        xBlock1.release();
    } );


    /* set the diagonal of the distance matrix to zeros */
    algorithmFPType zero = (algorithmFPType)0.0;
    daal::threader_for(n, n, [ = ](size_t i)
    {
        r[n*i-i*(i-1)/2 ] = zero;
    } );

    rPackedMicroTable.release();
    return 0;
}

} // namespace internal

} // namespace cosine_distance

} // namespace algorithms

} // namespace daal