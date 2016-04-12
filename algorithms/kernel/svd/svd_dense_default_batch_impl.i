/* file: svd_dense_default_batch_impl.i */
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
//  Implementation of svds
//--
*/

#ifndef __SVD_KERNEL_BATCH_IMPL_I__
#define __SVD_KERNEL_BATCH_IMPL_I__

#include "service_memory.h"
#include "service_math.h"
#include "service_defines.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

#include "svd_dense_default_impl.i"

#include "threading.h"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace internal
{

/**
 *  \brief Kernel for SVD calculation
 */
template <typename interm, daal::algorithms::svd::Method method, CpuType cpu>
void SVDBatchKernel<interm, method, cpu>::compute(const size_t na, const NumericTable *const *a,
                                                                const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    size_t i, j;
    svd::Parameter defaultParams;
    const svd::Parameter *svdPar = &defaultParams;

    if ( par != 0 )
    {
        svdPar = static_cast<const svd::Parameter *>(par);
    }

    BlockMicroTable<interm, readOnly , cpu> mtA    (a[0]);
    BlockMicroTable<interm, writeOnly, cpu> mtSigma(r[0]);

    size_t n = mtA.getFullNumberOfColumns();
    size_t m = mtA.getFullNumberOfRows();

    interm *A;
    interm *Sigma;

    mtA    .getBlockOfRows( 0, m, &A     );
    mtSigma.getBlockOfRows( 0, 1, &Sigma );

    interm *AT = (interm *)daal::services::daal_malloc(m * n * sizeof(interm));
    interm *QT = (interm *)daal::services::daal_malloc(m * n * sizeof(interm));
    interm *VT = (interm *)daal::services::daal_malloc(n * n * sizeof(interm));

    for ( i = 0 ; i < n ; i++ )
    {
        for ( j = 0 ; j < m; j++ )
        {
            AT[i * m + j] = A[i + j * n];
        }
    }

    compute_svd_on_one_node<interm, cpu>( m, n, AT, m, Sigma, QT, m, VT, n );

    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        BlockMicroTable<interm, writeOnly, cpu> mtQ(r[1]);
        interm *Q;
        mtQ.getBlockOfRows( 0, m, &Q );
        for ( i = 0 ; i < n ; i++ )
        {
            for ( j = 0 ; j < m; j++ )
            {
                Q[i + j * n] = QT[i * m + j];
            }
        }
        mtQ.release();
    }

    if (svdPar->rightSingularMatrix == requiredInPackedForm)
    {
        BlockMicroTable<interm, writeOnly, cpu> mtV(r[2]);
        interm *V;
        mtV.getBlockOfRows( 0, n, &V );
        for ( i = 0 ; i < n ; i++ )
        {
            for ( j = 0 ; j < n; j++ )
            {
                V[i + j * n] = VT[i * n + j];
            }
        }
        mtV.release();
    }

    mtA    .release();
    mtSigma.release();

    daal::services::daal_free(AT);
    daal::services::daal_free(QT);
    daal::services::daal_free(VT);
}

} // namespace daal::internal
}
}
} // namespace daal

#endif
