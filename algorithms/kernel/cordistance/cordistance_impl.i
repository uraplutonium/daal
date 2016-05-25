/* file: cordistance_impl.i */
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
//  Implementation of distances
//--
*/

#include "service_micro_table.h"
#include "service_math.h"
#include "service_memory.h"
#include "service_blas.h"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace correlation_distance
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu> bool isFull(NumericTableIface::StorageLayout layout);
template <typename algorithmFPType, CpuType cpu> bool isUpper(NumericTableIface::StorageLayout layout);
template <typename algorithmFPType, CpuType cpu> bool isLower(NumericTableIface::StorageLayout layout);
/**
 *  \brief Kernel for Correlation distances calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
void DistanceKernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable *const *a,
                                                           const size_t nr, NumericTable *r[],
                                                           const daal::algorithms::Parameter *par)
{
    NumericTable *xTable = const_cast<NumericTable *>( a[0] );  /* Input data */
    MKL_INT n   = (MKL_INT)(xTable->getNumberOfRows());         /* Number of input feature vectors */
    MKL_INT dim = (MKL_INT)(xTable->getNumberOfColumns());      /* Dimension of input feature vectors */
    MKL_INT ione = 1;
    algorithmFPType one = (algorithmFPType)1.0;
    algorithmFPType *x;      /* Input data           */
    algorithmFPType *d;      /* Resulting distances  */
    algorithmFPType *xxt;    /* Buffer for algorithmFPTypeediate results */
    algorithmFPType *vone;   /* Vector of ones */
    algorithmFPType *xsum;   /* Vector of sums of rows of matrix X */
    NumericTableIface::StorageLayout rLayout = r[0]->getDataLayout();

    xxt  = (algorithmFPType *)daal::services::daal_malloc(n * n * sizeof(algorithmFPType));
    vone = (algorithmFPType *)daal::services::daal_malloc(dim * sizeof(algorithmFPType));
    xsum = (algorithmFPType *)daal::services::daal_malloc(n * sizeof(algorithmFPType));
    if (!xxt || !vone || !xsum) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    for (size_t i = 0; i < dim; i++)
    {
        vone[i] = one;
    }

    BlockMicroTable<algorithmFPType, readOnly, cpu> *aMicroTable =
        new BlockMicroTable<algorithmFPType, readOnly, cpu>(xTable);
    aMicroTable->getBlockOfRows(0, n, &x);

    char uplo, trans;
    algorithmFPType alpha, beta;
    /* Calculate X*X' */
    uplo  = 'U';
    trans = 'T';
    alpha = 1.0;
    beta  = 0.0;
    Blas<algorithmFPType, cpu>::xsyrk(&uplo, &trans, &n, &dim, &alpha, x, &dim, &beta, xxt, &n);
    /* Calculate X*vone */
    trans = 'T';
    alpha = 1.0;
    beta  = 0.0;
    Blas<algorithmFPType, cpu>::xgemv(&trans, &dim, &n, &alpha, x, &dim, vone, &ione, &beta, xsum, &ione);

    aMicroTable->release();
    delete aMicroTable;

    algorithmFPType invDim = one / (algorithmFPType)dim;
    for (size_t i = 0; i < n; i++)
    {
        if (xxt[i * n + i] != 0.0)
        {
            xxt[i * n + i] = one / sSqrt<cpu>(xxt[i * n + i] - xsum[i] * xsum[i] * invDim);
        }
    }

    if(isFull<algorithmFPType, cpu>(rLayout))
    {
        BlockMicroTable<algorithmFPType, writeOnly, cpu> *rMicroTable = new BlockMicroTable<algorithmFPType, writeOnly, cpu>
        (r[0]);
        rMicroTable->getBlockOfRows(0, n, &d);

        /* Pack the results into output array */
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                d[i * n + j] = one - (xxt[i * n + j] - xsum[i] * xsum[j] * invDim) *
                               xxt[i * n + i] * xxt[j * n + j];
                d[j * n + i] = d[i * n + j];
            }
            d[i * n + i] = one;
        }

        rMicroTable->release();
        delete rMicroTable;
    }
    else
    {
        PackedArrayMicroTable<algorithmFPType, writeOnly, cpu> *rPackedMicroTable
            = new PackedArrayMicroTable<algorithmFPType, writeOnly, cpu>(r[0]);
        rPackedMicroTable->getPackedArray(&d);

        /* Pack the results into output array */
        size_t dIndex = 0;
        if(isLower<algorithmFPType, cpu>(rLayout))
        {
            for (size_t i = 0; i < n; i++)
            {
                for (size_t j = 0; j < i; j++)
                {
                    d[dIndex++] = one - (xxt[i * n + j] - xsum[i] * xsum[j] * invDim) *
                                  xxt[i * n + i] * xxt[j * n + j];
                }
                d[dIndex++] = one;
            }
        }
        else if(isUpper<algorithmFPType, cpu>(rLayout))
        {
            for (size_t j = 0; j < n; j++)
            {
                d[dIndex++] = one;
                for (size_t i = j + 1; i < n; i++)
                {
                    d[dIndex++] = one - (xxt[i * n + j] - xsum[i] * xsum[j] * invDim) *
                                  xxt[i * n + i] * xxt[j * n + j];
                }
            }
        }
        else
        {
            this->_errors->add(services::ErrorIncorrectTypeOfOutputNumericTable); return;
        }

        rPackedMicroTable->release();
        delete rPackedMicroTable;
    }

    daal::services::daal_free(xxt);
    daal::services::daal_free(vone);
    daal::services::daal_free(xsum);

}

template <typename algorithmFPType, CpuType cpu>
bool isFull(NumericTableIface::StorageLayout layout)
{
    int layoutInt = (int) layout;
    if (packed_mask & layoutInt)
    {
        return false;
    }
    return true;
}

template <typename algorithmFPType, CpuType cpu>
bool isUpper(NumericTableIface::StorageLayout layout)
{
    if (layout == NumericTableIface::upperPackedSymmetricMatrix  ||
        layout == NumericTableIface::upperPackedTriangularMatrix)
    {
        return true;
    }
    return false;
}

template <typename algorithmFPType, CpuType cpu>
bool isLower(NumericTableIface::StorageLayout layout)
{
    if (layout == NumericTableIface::lowerPackedSymmetricMatrix  ||
        layout == NumericTableIface::lowerPackedTriangularMatrix)
    {
        return true;
    }
    return false;
}

} // namespace internal

} // namespace correlation_distance

} // namespace algorithms

} // namespace daal
