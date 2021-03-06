/* file: implicit_als_predict_ratings_csr_default_distr_step1_fpt_cpu.cpp */
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
//  Implementation of implicit ALS prediction functions.
//--
*/

#include "implicit_als_predict_ratings_dense_default_kernel.h"
#include "implicit_als_predict_ratings_dense_default_container.h"
#include "implicit_als_predict_ratings_dense_default_impl.i"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace prediction
{
namespace ratings
{
namespace interface1
{
template class DistributedContainer<step1Local, DAAL_FPTYPE, defaultDense, DAAL_CPU>;
}
}
}
}
}
}
