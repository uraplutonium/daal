/* file: mcg59_impl.i */
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
//  Implementation of mcg59 algorithm
//--
*/

#ifndef __MCG59_IMPL_I__
#define __MCG59_IMPL_I__

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace mcg59
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
Status Mcg59Kernel<algorithmFPType, method, cpu>::compute(NumericTable *resultTensor)
{
    return Status();
}

} // namespace internal
} // namespace mcg59
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif
