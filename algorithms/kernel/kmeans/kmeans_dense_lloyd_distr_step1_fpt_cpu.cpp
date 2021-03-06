/* file: kmeans_dense_lloyd_distr_step1_fpt_cpu.cpp */
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
//  Implementation of Lloyd method for K-means algorithm.
//--
*/

#include "kmeans_lloyd_kernel.h"
#include "kmeans_lloyd_distr_step1_impl.i"
#include "kmeans_container.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace interface1
{
template class DistributedContainer<step1Local, DAAL_FPTYPE, lloydDense, DAAL_CPU>;
}
namespace internal
{
template class KMeansDistributedStep1Kernel<lloydDense, DAAL_FPTYPE, DAAL_CPU>;
} // namespace daal::algorithms::kmeans::internal
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
