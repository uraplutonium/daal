/* file: linear_regression_single_beta_dense_default_batch_kernel.h */
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
//  Declaration of template class that computes linear regression quality metrics.
//--
*/

#ifndef __LINEAR_REGRESSION_SINGLE_BETA_DENSE_DEFAULT_BATCH_KERNEL_H__
#define __LINEAR_REGRESSION_SINGLE_BETA_DENSE_DEFAULT_BATCH_KERNEL_H__

#include "linear_regression_single_beta_types.h"
#include "kernel.h"
#include "numeric_table.h"
#include "algorithm_base_common.h"


namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace quality_metric
{
namespace single_beta
{
namespace internal
{

using namespace daal::data_management;
struct SingleBetaOutput
{
    NumericTable* rms;
    NumericTable* variance;
    NumericTable** betaCovariances;
    NumericTable* zScore;
    NumericTable* confidenceIntervals;
    NumericTable* inverseOfXtX;

    SingleBetaOutput(size_t nResponses);
    ~SingleBetaOutput();
};


template<Method method, typename algorithmFPType, CpuType cpu>
class SingleBetaKernel : public daal::algorithms::Kernel
{
public:
    virtual ~SingleBetaKernel() {}
    services::Status compute(const NumericTable* y, const NumericTable* z, size_t p,
        const NumericTable* betas, const NumericTable* xtx, bool bModelNe,
        algorithmFPType accuracyThreshold, algorithmFPType alpha, SingleBetaOutput& out);

protected:
    static const size_t _nRowsInBlock = 1024;
    services::Status computeTestStatistics(const NumericTable* betas, const algorithmFPType* v,
        algorithmFPType alpha, algorithmFPType accuracyThreshold, SingleBetaOutput& out);
    services::Status computeRmsVariance(const NumericTable* y, const NumericTable* z, size_t p, NumericTable* rms, NumericTable* variance);
    services::Status computeInverseXtX(const NumericTable* xtx, bool bModelNe, NumericTable* xtxInv);
};

}
}
}
}
}
}

#endif
