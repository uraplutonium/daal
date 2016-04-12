/* file: cosdistance_impl.i */
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
#include "daal_defines.h"
#include "service_blas.h"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace cosine_distance
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu> bool isFull(NumericTableIface::StorageLayout layout);
template <typename algorithmFPType, CpuType cpu> bool isUpper(NumericTableIface::StorageLayout layout);
template <typename algorithmFPType, CpuType cpu> bool isLower(NumericTableIface::StorageLayout layout);
/**
 *  \brief Kernel for Cosine distances calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
void DistanceKernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable *const *a,
                                                           const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    NumericTable *xTable = const_cast<NumericTable *>( a[0] );  /* Input data */
    MKL_INT n   = (MKL_INT)(xTable->getNumberOfRows());         /* Number of input feature vectors */
    MKL_INT p   = (MKL_INT)(xTable->getNumberOfColumns());      /* Number of input vector dimension */
    NumericTableIface::StorageLayout rLayout = r[0]->getDataLayout();

    algorithmFPType *x;      /* Input data           */
    algorithmFPType *d;      /* Resulting distances  */
    algorithmFPType *xxt;    /* Buffer for algorithmFPTypeediate results */

    xxt = (algorithmFPType *)daal::services::daal_malloc(n * n * sizeof(algorithmFPType));
    if (!xxt) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    BlockMicroTable<algorithmFPType, readOnly, cpu> *aMicroTable  =
        new BlockMicroTable<algorithmFPType, readOnly, cpu>(xTable);
    aMicroTable->getBlockOfRows(0, n, &x);

    char uplo, trans;
    algorithmFPType alpha, beta;
    /* Calculate X*X' */
    uplo  = 'U';
    trans = 'T';
    alpha = 1.0;
    beta  = 0.0;
    Blas<algorithmFPType, cpu>::xsyrk(&uplo, &trans, &n, &p, &alpha, x, &p, &beta, xxt, &n);

    aMicroTable->release();
    delete aMicroTable;

    algorithmFPType one = (algorithmFPType)1.0;
    for (size_t i = 0; i < n; i++)
    {
        if (xxt[i * n + i] != 0.0)
        {
            xxt[i * n + i] = one / sSqrt<cpu>(xxt[i * n + i]);
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
                d[i * n + j] = one - xxt[i * n + j] * xxt[i * n + i] * xxt[j * n + j];
                d[j * n + i] = d[i * n + j];
            }
            d[i * n + i] = one;
        }

        rMicroTable->release();
        delete rMicroTable;
    }
    else
    {
        PackedArrayMicroTable<algorithmFPType, writeOnly, cpu> *rPackedMicroTable =
            new PackedArrayMicroTable<algorithmFPType, writeOnly, cpu>
        (r[0]);
        rPackedMicroTable->getPackedArray(&d);

        /* Pack the results into output array */
        size_t dIndex = 0;
        if(isLower<algorithmFPType, cpu>(rLayout))
        {
            for (size_t i = 0; i < n; i++)
            {
                for (size_t j = 0; j < i; j++)
                {
                    d[dIndex++] = one - xxt[i * n + j] * xxt[i * n + i] * xxt[j * n + j];
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
                    d[dIndex++] = one - xxt[i * n + j] * xxt[i * n + i] * xxt[j * n + j];
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

} // namespace cosine_distance

} // namespace algorithms

} // namespace daal
