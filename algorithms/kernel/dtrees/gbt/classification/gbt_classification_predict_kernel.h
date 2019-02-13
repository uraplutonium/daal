/* file: gbt_classification_predict_kernel.h */
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
//  Declaration of template function that computes gradient boosted trees
//  prediction results.
//--
*/

#ifndef __GBT_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_H__
#define __GBT_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_H__

#include "gbt_classification_predict.h"
#include "service_memory.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace classification
{
namespace prediction
{
namespace internal
{

template <typename algorithmFpType, gbt::classification::prediction::Method method, CpuType cpu>
class PredictKernel : public daal::algorithms::Kernel
{
public:
    /**
     *  \brief Compute gradient boosted trees prediction results.
     *
     *  \param a[in]    Matrix of input variables X
     *  \param m[in]   Gradient boosted trees model obtained on training stage
     *  \param r[out]   Prediction results
     *  \param nClasses[in]     Number of classes in gradient boosted trees algorithm parameter
     *  \param nIterations[in]  Number of iterations to predict in gradient boosted trees algorithm parameter
     */
    services::Status compute(services::HostAppIface* pHostApp, const NumericTable *a, const classification::Model *m, NumericTable *r,
        size_t nClasses, size_t nIterations);
};

} // namespace internal
}
}
}
}
} // namespace daal

#endif
