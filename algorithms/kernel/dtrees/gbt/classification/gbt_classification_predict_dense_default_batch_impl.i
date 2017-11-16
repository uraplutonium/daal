/* file: gbt_classification_predict_dense_default_batch_impl.i */
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
//  Common functions for gradient boosted trees classification predictions calculation
//--
*/

#ifndef __GBT_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__
#define __GBT_CLASSIFICATION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__

#include "algorithm.h"
#include "numeric_table.h"
#include "gbt_classification_predict_kernel.h"
#include "threading.h"
#include "daal_defines.h"
#include "gbt_classification_model_impl.h"
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
namespace classification
{
namespace prediction
{
namespace internal
{

static const size_t nRowsInBlock = 500;

//////////////////////////////////////////////////////////////////////////////////////////
// PredictClassificationTask
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class PredictClassificationTask
{
public:
    typedef dtrees::internal::TreeImpRegression<> TreeType;
    PredictClassificationTask(const NumericTable *x, NumericTable *y,
        const gbt::classification::internal::ModelImpl* m, size_t nIterations) :
        _data(x), _res(y), _model(m), _nIterations(nIterations){}

    services::Status run(size_t nClasses);

protected:
    services::Status runBinary(size_t nRows, size_t nCols, size_t nBlocks,
        dtrees::internal::FeatureTypeHelper<cpu>& featHelper, algorithmFPType* aRes);
    services::Status runMulticlass(size_t nClasses, size_t nRows, size_t nCols, size_t nBlocks,
        dtrees::internal::FeatureTypeHelper<cpu>& featHelper, algorithmFPType* aRes);
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
    const gbt::classification::internal::ModelImpl* _model;
    size_t _nIterations;
};

//////////////////////////////////////////////////////////////////////////////////////////
// PredictKernel
//////////////////////////////////////////////////////////////////////////////////////////
template<typename algorithmFPType, prediction::Method method, CpuType cpu>
services::Status PredictKernel<algorithmFPType, method, cpu>::compute(const NumericTable *x,
    const classification::Model *m, NumericTable *r, size_t nClasses, size_t nIterations)
{
    const daal::algorithms::gbt::classification::internal::ModelImpl* pModel =
        static_cast<const daal::algorithms::gbt::classification::internal::ModelImpl*>(m);
    PredictClassificationTask<algorithmFPType, cpu> task(x, r, pModel, nIterations);
    return task.run(nClasses);
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictClassificationTask<algorithmFPType, cpu>::run(size_t nClasses)
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
    return nClasses == 2 ? runBinary(nRows, nCols, nBlocks, featHelper, resBD.get()) :
        runMulticlass(nClasses, nRows, nCols, nBlocks, featHelper, resBD.get());
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictClassificationTask<algorithmFPType, cpu>::runBinary(size_t nRows, size_t nCols, size_t nBlocks,
    dtrees::internal::FeatureTypeHelper<cpu>& featHelper, algorithmFPType* aRes)
{
    const algorithmFPType label[2] = { algorithmFPType(1.), algorithmFPType(0.) };
    DAAL_ASSERT(!_nIterations || _nIterations <= _model->size());
    const auto size = (_nIterations ? _nIterations : _model->size());
    SafeStatus safeStat;
    daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock)
    {
        const size_t iStartRow = iBlock*nRowsInBlock;
        const size_t nRowsToProcess = (iBlock == nBlocks - 1) ? nRows - iBlock * nRowsInBlock : nRowsInBlock;
        ReadRows<algorithmFPType, cpu> xBD(const_cast<NumericTable*>(_data), iStartRow, nRowsToProcess);
        DAAL_CHECK_BLOCK_STATUS_THR(xBD);
        algorithmFPType* res = aRes + iStartRow;
        daal::threader_for(nRowsToProcess, nRowsToProcess, [&](size_t iRow)
        {
            algorithmFPType f = 0; //raw boosted value
            const auto ptr = xBD.get() + iRow*nCols;
            for(size_t iTree = 0; iTree < size; ++iTree)
            {
                //recalculate response incrementally, as a sum of all trees responses
                f += predict(*_model->at(iTree), featHelper, ptr);
            }
            //probablity is a sigmoid(f) hence sign(f) can be checked
            res[iRow] = label[daal::data_feature_utils::internal::SignBit<algorithmFPType, cpu>::get(f)];
        });
    });
    return safeStat.detach();
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictClassificationTask<algorithmFPType, cpu>::runMulticlass(size_t nClasses, size_t nRows, size_t nCols, size_t nBlocks,
    dtrees::internal::FeatureTypeHelper<cpu>& featHelper, algorithmFPType* aRes)
{
    static const size_t s_cMaxClassesBufSize = 12;
    const bool bUseTLS(nClasses > s_cMaxClassesBufSize);
    DAAL_ASSERT(!_nIterations || nClasses*_nIterations <= _model->size());
    const auto size = (_nIterations ? _nIterations*nClasses : _model->size());

    daal::tls<algorithmFPType *> lsData([=]()-> algorithmFPType*
    {
        return service_scalable_malloc<algorithmFPType, cpu>(nClasses);
    });

    daal::SafeStatus safeStat;
    daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock)
    {
        const size_t iStartRow = iBlock*nRowsInBlock;
        const size_t nRowsToProcess = (iBlock == nBlocks - 1) ? nRows - iBlock * nRowsInBlock : nRowsInBlock;
        ReadRows<algorithmFPType, cpu> xBD(const_cast<NumericTable*>(_data), iStartRow, nRowsToProcess);
        DAAL_CHECK_BLOCK_STATUS_THR(xBD);
        algorithmFPType* res = aRes + iStartRow;
        daal::threader_for(nRowsToProcess, nRowsToProcess, [&](size_t iRow)
        {
            const auto ptr = xBD.get() + iRow*nCols;
            algorithmFPType buf[s_cMaxClassesBufSize];
            algorithmFPType* val = bUseTLS ? lsData.local() : buf;
            PRAGMA_VECTOR_ALWAYS
            for(size_t i = 0; i < nClasses; ++i)
                val[i] = 0;

            for(size_t iTree = 0; iTree < size; ++iTree)
                val[iTree%nClasses] += predict(*_model->at(iTree), featHelper, ptr);

            algorithmFPType maxVal = val[0];
            size_t maxIdx = 0;
            PRAGMA_VECTOR_ALWAYS
            for(size_t i = 1; i < nClasses; ++i)
            {
                if(maxVal < val[i])
                {
                    maxVal = val[i];
                    maxIdx = i;
                }
            }
            res[iRow] = maxIdx;
        });
    });
    if(bUseTLS)
    {
        lsData.reduce([](algorithmFPType* ptr)-> void
        {
            if(ptr)
                service_scalable_free<algorithmFPType, cpu>(ptr);
        });
    }
    return safeStat.detach();
}

} /* namespace internal */
} /* namespace prediction */
} /* namespace classification */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif
