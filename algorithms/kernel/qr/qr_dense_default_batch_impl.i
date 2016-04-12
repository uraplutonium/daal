/* file: qr_dense_default_batch_impl.i */
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

namespace daal
{
namespace algorithms
{
namespace qr
{
namespace internal
{
/**
 *  \brief Kernel for QR QR calculation
 */
template <typename interm, daal::algorithms::qr::Method method, CpuType cpu>
void QRBatchKernel<interm, method, cpu>::compute(const size_t na, const NumericTable *const *a,
                                                 const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    size_t i, j;
    qr::Parameter defaultParams;

    NumericTable *ntAi = const_cast<NumericTable *>(a[0]);
    NumericTable *ntRi = const_cast<NumericTable *>(r[1]);

    size_t  n   = ntAi->getNumberOfColumns();
    size_t  m   = ntAi->getNumberOfRows();

    MKL_INT ldAi = m;
    MKL_INT ldRi = n;

    interm *QiT = (interm *)daal::services::daal_malloc(n * m * sizeof(interm));
    interm *RiT = (interm *)daal::services::daal_malloc(n * n * sizeof(interm));

    BlockDescriptor<interm> aiBlock;
    BlockDescriptor<interm> riBlock;

    ntAi->getBlockOfRows( 0, m, readOnly,  aiBlock); /*      Ai [m][n] */
    ntRi->getBlockOfRows( 0, n, writeOnly, riBlock); /* Ri = Ri [n][n] */

    interm *Ai = aiBlock.getBlockPtr();
    interm *Ri = riBlock.getBlockPtr();

    for ( i = 0 ; i < n ; i++ )
    {
        for ( j = 0 ; j < m; j++ )
        {
            QiT[i * m + j] = Ai[i + j * n];
        }
    }

    compute_QR_on_one_node<interm, cpu>( m, n, QiT, ldAi, RiT, ldRi );

    NumericTable *ntQi = const_cast<NumericTable *>(r[0]);
    BlockDescriptor<interm> qiBlock;
    ntQi->getBlockOfRows( 0, m, writeOnly, qiBlock ); /* Qi = Qin[m][n] */
    interm *Qi = qiBlock.getBlockPtr();
    for ( i = 0 ; i < n ; i++ )
    {
        for ( j = 0 ; j < m; j++ )
        {
            Qi[i + j * n] = QiT[i * m + j];
        }
    }
    ntQi->releaseBlockOfRows( qiBlock );

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

    ntAi->releaseBlockOfRows( aiBlock );
    ntRi->releaseBlockOfRows( riBlock );

    daal::services::daal_free(QiT);
    daal::services::daal_free(RiT);
}

} // namespace daal::internal
}
}
} // namespace daal

#endif
