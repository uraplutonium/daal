/* file: qr_dense_default_batch_fpt_cpu.cpp */
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
//  Instantiation of QR algorithm classes.
//--
*/

#include "qr_dense_default_kernel.h"
#include "qr_dense_default_batch_impl.i"
#include "qr_dense_default_container.h"

namespace daal
{
namespace algorithms
{
namespace qr
{
namespace interface1
{
template class BatchContainer<DAAL_FPTYPE, daal::algorithms::qr::defaultDense, DAAL_CPU>;
}
namespace internal
{
template class QRBatchKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
}
}
}
}
