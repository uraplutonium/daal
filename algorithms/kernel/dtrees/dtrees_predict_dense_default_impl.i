/* file: dtrees_predict_dense_default_impl.i */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation
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
//  Implementation of auxiliary functions for decision forest predict algorithms
//  (defaultDense) method.
//--
*/

#ifndef __DTREES_PREDICT_DENSE_DEFAULT_IMPL_I__
#define __DTREES_PREDICT_DENSE_DEFAULT_IMPL_I__

#include "dtrees_model_impl.h"
#include "service_data_utils.h"
#include "dtrees_feature_type_helper.h"
#include "service_environment.h"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace dtrees
{
namespace prediction
{
namespace internal
{

using namespace dtrees::internal;
//////////////////////////////////////////////////////////////////////////////////////////
// Common service function. Finds node corresponding to the given observation
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, typename TreeType, CpuType cpu>
const typename TreeType::NodeType::Base* findNode(const dtrees::internal::Tree& t, const algorithmFPType* x)
{
    const TreeType& tree = static_cast<const TreeType&>(t);
    const typename TreeType::NodeType::Base* pNode = tree.top();
    if(tree.hasUnorderedFeatureSplits())
    {
        for(; pNode && pNode->isSplit();)
        {
            auto pSplit = TreeType::NodeType::castSplit(pNode);
            const int sn = (pSplit->featureUnordered ? (int(x[pSplit->featureIdx]) != int(pSplit->featureValue)) :
                daal::services::internal::SignBit<algorithmFPType, cpu>::get(pSplit->featureValue - x[pSplit->featureIdx]));
            pNode = pSplit->kid[sn];
        }
    }
    else
    {
        for(; pNode && pNode->isSplit();)
        {
            auto pSplit = TreeType::NodeType::castSplit(pNode);
            const int sn = daal::services::internal::SignBit<algorithmFPType, cpu>::get(pSplit->featureValue - x[pSplit->featureIdx]);
            pNode = pSplit->kid[sn];
        }
    }
    return pNode;
}

//////////////////////////////////////////////////////////////////////////////////////////
// Common service function. Finds a node corresponding to the given observation
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, typename TreeType, CpuType cpu>
const DecisionTreeNode* findNode(const dtrees::internal::DecisionTreeTable& t,
    const FeatureTypes& featTypes, const algorithmFPType* x)
{
    const DecisionTreeNode* aNode = (const DecisionTreeNode*)t.getArray();
    if(!aNode)
        return nullptr;
    const DecisionTreeNode* pNode = aNode;
    if(featTypes.hasUnorderedFeatures())
    {
        for(; pNode->isSplit();)
        {
            const int sn = (featTypes.isUnordered(pNode->featureIndex) ? (int(x[pNode->featureIndex]) != int(pNode->featureValue())) :
                daal::services::internal::SignBit<algorithmFPType, cpu>::get(algorithmFPType(pNode->featureValue()) - x[pNode->featureIndex]));
            DAAL_ASSERT(pNode->leftIndexOrClass > 0);
            DAAL_ASSERT(sn == 0 || sn == 1);
            DAAL_ASSERT(pNode->leftIndexOrClass + sn > 0 && pNode->leftIndexOrClass + sn < t.getNumberOfRows());
            pNode = aNode + (pNode->leftIndexOrClass + sn);
        }
    }
    else
    {
        for(; pNode->isSplit();)
        {
            const int sn = daal::services::internal::SignBit<algorithmFPType, cpu>::get(algorithmFPType(pNode->featureValue()) - x[pNode->featureIndex]);
            DAAL_ASSERT(pNode->leftIndexOrClass > 0);
            DAAL_ASSERT(sn == 0 || sn == 1);
            DAAL_ASSERT(pNode->leftIndexOrClass + sn > 0 && pNode->leftIndexOrClass + sn < t.getNumberOfRows());
            pNode = aNode + (pNode->leftIndexOrClass + sn);
        }
    }
    return pNode;
}

template <typename algorithmFPType>
struct TileDimensions
{
    size_t nRowsTotal = 0;
    size_t nTreesTotal = 0;
    size_t nCols = 0;
    size_t nRowsInBlock = 0;
    size_t nTreesInBlock = 0;
    size_t nDataBlocks = 0;
    size_t nTreeBlocks = 0;

    TileDimensions(const NumericTable& data, size_t nTrees, size_t treeSize, size_t nYPerRow = 1) :
        nTreesTotal(nTrees), nRowsTotal(data.getNumberOfRows()), nCols(data.getNumberOfColumns())
    {
        nRowsInBlock = services::internal::getNumElementsFitInMemory(services::internal::getL1CacheSize()*0.8,
            (nCols + nYPerRow)*sizeof(algorithmFPType), nRowsInBlockDefault);
        nTreesInBlock = services::internal::getNumElementsFitInMemory(services::internal::getLLCacheSize()*0.8, treeSize, nTrees);
        nDataBlocks = nRowsTotal / nRowsInBlock + !!(nRowsTotal%nRowsInBlock);
        nTreeBlocks = nTreesTotal / nTreesInBlock + !!(nTreesTotal%nTreesInBlock);
    }
    static const size_t nRowsInBlockDefault = 500;
};

} /* namespace internal */
} /* namespace prediction */
} /* namespace dtrees */
} /* namespace algorithms */
} /* namespace daal */

#endif
