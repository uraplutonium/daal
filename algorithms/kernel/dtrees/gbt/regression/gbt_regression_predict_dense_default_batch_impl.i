/* file: gbt_regression_predict_dense_default_batch_impl.i */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
* All Rights Reserved.
*
* If this  software was obtained  under the  Intel Simplified  Software License,
* the following terms apply:
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*
*
* If this  software  was obtained  under the  Apache License,  Version  2.0 (the
* "License"), the following terms apply:
*
* You may  not use this  file except  in compliance  with  the License.  You may
* obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
*
*
* Unless  required  by   applicable  law  or  agreed  to  in  writing,  software
* distributed under the License  is distributed  on an  "AS IS"  BASIS,  WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*
* See the   License  for the   specific  language   governing   permissions  and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Common functions for gradient boosted trees regression predictions calculation
//--
*/

#ifndef __GBT_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__
#define __GBT_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__

#include "algorithm.h"
#include "numeric_table.h"
#include "gbt_regression_predict_kernel.h"
#include "threading.h"
#include "daal_defines.h"
#include "gbt_regression_model_impl.h"
#include "service_numeric_table.h"
#include "service_error_handling.h"
#include "service_memory.h"
#include "dtrees_predict_dense_default_impl.i"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace prediction
{
namespace internal
{

static const size_t nRowsInBlock = 500;

//////////////////////////////////////////////////////////////////////////////////////////
// PredictRegressionTask
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class PredictRegressionTask
{
public:
    typedef dtrees::internal::TreeImpRegression<> TreeType;
    PredictRegressionTask(const NumericTable *x, NumericTable *y, const gbt::regression::internal::ModelImpl* m) :
        _data(x), _res(y), _model(m){}

    services::Status run(size_t nIterations);

protected:
    static algorithmFPType predict(const dtrees::internal::DecisionTreeTable& t,
        const dtrees::internal::FeatureTypeHelper<cpu>& featHelper, const algorithmFPType* x)
    {
        const typename dtrees::internal::DecisionTreeNode* pNode =
            dtrees::prediction::internal::findNode<algorithmFPType, TreeType, cpu>(t, featHelper, x);
        DAAL_ASSERT(pNode);
        return pNode ? pNode->featureValueOrResponse : 0.;
    }

protected:
    const NumericTable* _data;
    NumericTable* _res;
    const gbt::regression::internal::ModelImpl* _model;
};

//////////////////////////////////////////////////////////////////////////////////////////
// RandomForestPredictKernel
//////////////////////////////////////////////////////////////////////////////////////////
template<typename algorithmFPType, prediction::Method method, CpuType cpu>
services::Status PredictKernel<algorithmFPType, method, cpu>::compute(const NumericTable *x,
    const regression::Model *m, NumericTable *r, size_t nIterations)
{
    const daal::algorithms::gbt::regression::internal::ModelImpl* pModel =
        static_cast<const daal::algorithms::gbt::regression::internal::ModelImpl*>(m);
    PredictRegressionTask<algorithmFPType, cpu> task(x, r, pModel);
    return task.run(nIterations);
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictRegressionTask<algorithmFPType, cpu>::run(size_t nIterations)
{
    dtrees::internal::FeatureTypeHelper<cpu> featHelper;
    DAAL_CHECK(featHelper.init(_data), services::ErrorMemoryAllocationFailed);

    const auto nRows = _data->getNumberOfRows();
    const auto nCols = _data->getNumberOfColumns();
    size_t nBlocks = nRows / nRowsInBlock;
    nBlocks += (nBlocks * nRowsInBlock != nRows);

    WriteOnlyRows<algorithmFPType, cpu> resBD(_res, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(resBD);
    daal::services::internal::service_memset<algorithmFPType, cpu>(resBD.get(), 0, nRows);

    DAAL_ASSERT(!_nIterations || _nIterations <= _model->size());
    const auto size = (nIterations ? nIterations : _model->size());
    SafeStatus safeStat;
    daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock)
    {
        const size_t iStartRow = iBlock*nRowsInBlock;
        const size_t nRowsToProcess = (iBlock == nBlocks - 1) ? nRows - iBlock * nRowsInBlock : nRowsInBlock;
        ReadRows<algorithmFPType, cpu> xBD(const_cast<NumericTable*>(_data), iStartRow, nRowsToProcess);
        DAAL_CHECK_BLOCK_STATUS_THR(xBD);
        algorithmFPType* res = resBD.get() + iStartRow;
        daal::threader_for(nRowsToProcess, nRowsToProcess, [&](size_t iRow)
        {
            res[iRow] = 0;
            const auto ptr = xBD.get() + iRow*nCols;
            for(size_t iTree = 0; iTree < size; ++iTree)
            {
                //recalculate response incrementally, as a sum of all trees responses
                res[iRow] += predict(*_model->at(iTree), featHelper, ptr);
            }
        });
    });
    return safeStat.detach();
}

} /* namespace internal */
} /* namespace prediction */
} /* namespace regression */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif
