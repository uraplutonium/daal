/* file: logistic_regression_train_kernel.h */
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
//  Declaration of structure containing kernels for logistic regression
//  training.
//--
*/

#ifndef __LOGISTIC_REGRESSION_TRAIN_KERNEL_H__
#define __LOGISTIC_REGRESSION_TRAIN_KERNEL_H__

#include "numeric_table.h"
#include "algorithm_base_common.h"
#include "algorithms/logistic_regression/logistic_regression_training_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace training
{
namespace internal
{

template <typename algorithmFPType, Method method, CpuType cpu>
class TrainBatchKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(const HostAppIfacePtr& pHost, const NumericTablePtr& x, const NumericTablePtr& y,
        logistic_regression::Model& m, Result& res, const Parameter& par);
};

} // namespace internal
}
}
}
} // namespace daal


#endif
