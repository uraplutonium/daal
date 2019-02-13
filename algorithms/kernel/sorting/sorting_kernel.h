/* file: sorting_kernel.h */
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
//  Declaration of template structs that sorts data.
//--
*/

#ifndef __SORTING_KERNEL_H__
#define __SORTING_KERNEL_H__

#include "numeric_table.h"
#include "sorting_batch.h"
#include "service_numeric_table.h"
#include "service_stat.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace sorting
{
namespace internal
{

template<Method method, typename algorithmFPType, CpuType cpu>
struct SortingKernel : public Kernel
{
    virtual ~SortingKernel() {}
    Status compute(const NumericTable &inputTable, NumericTable &outputTable);
};

} // namespace daal::algorithms::sorting::internal
} // namespace daal::algorithms::sorting
} // namespace daal::algorithms
} // namespace daal

#endif
