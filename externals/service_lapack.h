/* file: service_lapack.h */
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
//  Template wrappers for LAPACK functions.
//--
*/


#ifndef __SERVICE_LAPACK_H__
#define __SERVICE_LAPACK_H__

#include "daal_defines.h"
#include "service_memory.h"

#include "service_lapack_mkl.h"

namespace daal
{
namespace internal
{

/*
// Template functions definition
*/
template<typename fpType, CpuType cpu, template<typename, CpuType> class _impl=mkl::MklLapack>
struct Lapack
{
    typedef typename _impl<fpType,cpu>::SizeType SizeType;

    static void xpotrf(char *uplo, SizeType *p, fpType *ata, SizeType *ldata, SizeType *info)
    {
        _impl<fpType,cpu>::xpotrf(uplo, p, ata, ldata, info);
    }

    static void xpotrs(char *uplo, SizeType *p, SizeType *ny, fpType *ata, SizeType *ldata, fpType *beta, SizeType *ldaty,
                SizeType *info)
    {
        _impl<fpType,cpu>::xpotrs(uplo, p, ny, ata, ldata, beta, ldaty, info);
    }

    static void xpotri(char *uplo, SizeType *p, fpType *ata, SizeType *ldata, SizeType *info)
    {
        _impl<fpType,cpu>::xpotri(uplo, p, ata, ldata, info);
    }

    static void xgerqf(SizeType *m, SizeType *n, fpType *a, SizeType *lda, fpType *tau, fpType *work, SizeType *lwork,
                SizeType *info)
    {
        _impl<fpType,cpu>::xgerqf(m, n, a, lda, tau, work, lwork, info);
    }

    static void xormrq(char *side, char *trans, SizeType *m, SizeType *n, SizeType *k, fpType *a, SizeType *lda,
                fpType *tau, fpType *c, SizeType *ldc, fpType *work, SizeType *lwork, SizeType *info)
    {
        _impl<fpType,cpu>::xormrq(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
    }

    static void xtrtrs(char *uplo, char *trans, char *diag, SizeType *n, SizeType *nrhs,
                fpType *a, SizeType *lda, fpType *b, SizeType *ldb, SizeType *info)
    {
        _impl<fpType,cpu>::xtrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
    }

    static void xpptrf(char *uplo, SizeType *n, fpType *ap, SizeType *info)
    {
        _impl<fpType,cpu>::xpptrf(uplo, n, ap, info);
    }

    static void xgeqrf(SizeType m, SizeType n, fpType *a,
                SizeType lda, fpType *tau, fpType *work, SizeType lwork, SizeType *info)
    {
        _impl<fpType,cpu>::xgeqrf(m, n, a, lda, tau, work, lwork, info);
    }

    static void xgeqp3(SizeType m, SizeType n, fpType *a,
                SizeType lda, SizeType *jpvt, fpType *tau, fpType *work, SizeType lwork, SizeType *info)
    {
        _impl<fpType,cpu>::xgeqp3(m, n, a, lda, jpvt, tau, work, lwork, info);
    }

    static void xorgqr(SizeType m, SizeType n, SizeType k,
                fpType *a, SizeType lda, fpType *tau, fpType *work, SizeType lwork, SizeType *info)
    {
        _impl<fpType,cpu>::xorgqr(m, n, k, a, lda, tau, work, lwork, info);
    }

    static void xgesvd(char jobu, char jobvt, SizeType m, SizeType n,
                fpType *a, SizeType lda, fpType *s, fpType *u, SizeType ldu, fpType *vt, SizeType ldvt,
                fpType *work, SizeType lwork, SizeType *info)
    {
        _impl<fpType,cpu>::xgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
    }

    static void xsyevd(char *jobz, char *uplo, SizeType *n, fpType *a, SizeType *lda, fpType *w, fpType *work,
                SizeType *lwork, SizeType *iwork, SizeType *liwork, SizeType *info)
    {
        _impl<fpType,cpu>::xsyevd(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
    }
};

} // namespace internal
} // namespace daal

#endif
