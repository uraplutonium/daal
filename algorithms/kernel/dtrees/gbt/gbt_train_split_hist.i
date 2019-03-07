/* file: gbt_train_split_hist.i */
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
//  Implementation of histogram method for gradient boosted trees training
//  (defaultDense) method.
//--
*/

#ifndef __GBT_TRAIN_SPLIT_HIST_I__
#define __GBT_TRAIN_SPLIT_HIST_I__

#include "dtrees_model_impl.h"
#include "dtrees_train_data_helper.i"
#include "dtrees_predict_dense_default_impl.i"
#include "gbt_train_aux.i"
#include "service_defines.h"
#include "gbt_train_hist_kernel.i"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace training
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu> class TreeBuilder;

namespace hist
{

template<typename algorithmFPType, typename IndexType, typename GHSumType, CpuType cpu> class GHSumsHelper;
template<typename algorithmFPType, typename IndexType, typename ImpurityType, typename GHSumType, typename SplitType, typename ResultType, CpuType cpu> class MaxImpurityDecreaseHelper;
template<typename IndexType, typename algorithmFPType, CpuType cpu> struct ComputeGHSumByRows;
template<typename algorithmFPType, typename IndexType, CpuType cpu> struct MergeGHSums;

template<typename algorithmFPType, typename IndexType, CpuType cpu>
class SplitTaskByColumns: public GbtTask
{
public:
    using SharedDataType      = SharedDataForTree<algorithmFPType, IndexType, cpu>;
    using NodeInfoType        = SplitJob<algorithmFPType,cpu>;
    using ResultType          = Result<algorithmFPType, cpu>;
    using ImpurityType        = ImpurityData<algorithmFPType, cpu>;
    using SplitDataType       = SplitData<algorithmFPType, ImpurityType>;
    using BestSplitType       = typename TreeBuilder<algorithmFPType, cpu>::BestSplit;
    using GHSumType           = ghSum<algorithmFPType, cpu>;
    using MaxImpurityDecrease = MaxImpurityDecreaseHelper<algorithmFPType, IndexType, ImpurityType, GHSumType, SplitDataType, ResultType, cpu>;
    using GHSums              = GHSumsHelper<algorithmFPType, IndexType, GHSumType, cpu>;

    SplitTaskByColumns(size_t iFeature, SharedDataType& data, const NodeInfoType& nodeInfo, BestSplitType& bestSplit, ResultType& res):
        _iFeature(iFeature), _data(data), _node(nodeInfo), _bestSplit(bestSplit), _res(res)
    {
    }

    virtual GbtTask* execute()
    {
        _res.ghSums = nullptr;
        _res.isFailed = true;
        computeGHSums();

        if(!_data.ctx.dataHelper().hasDiffFeatureValues(_iFeature, _data.aIdx + _node.iStart, _node.n))
            return nullptr; //all values of the feature are the same

        const bool featureUnordered = _data.ctx.featTypes().isUnordered(_iFeature);
        algorithmFPType bestImpDec;
        int iBestFeat;
        _bestSplit.safeGetData(bestImpDec, iBestFeat); // TODO: use them
        SplitDataType split(bestImpDec, featureUnordered);

        DAAL_INT idxFeatureBestSplit = -1;
        auto& par = _data.ctx.par();
        MaxImpurityDecrease::find(_node.n, par.minObservationsInLeafNode, par.lambda, split, _res, idxFeatureBestSplit, featureUnordered, _data, _iFeature);

        if(idxFeatureBestSplit >= 0)
        {
            _bestSplit.update(split, idxFeatureBestSplit, _iFeature);
            _res.isFailed = false;
        }

        return nullptr;
    }

    virtual void computeGHSums()
    {
        const size_t nUnique = _data.ctx.dataHelper().indexedFeatures().numIndices(_iFeature);
        const IndexType* indexedFeature = _data.ctx.dataHelper().indexedFeatures().data(_iFeature);

        auto* aGHSum = _data.GH_SUMS_BUF->singleGHSums.get(_iFeature).getBlockFromStorage();
        DAAL_ASSERT(aGHSum); //TODO: return status

        GHSums::fillByZero(nUnique, aGHSum);
        algorithmFPType gTotal = 0, hTotal = 0;

        GHSums::compute(_node.iStart, _node.n, indexedFeature, _data.aIdx, _data.ctx.aSampleToF(),
                (algorithmFPType*)_data.ctx.grad(_data.iTree), aGHSum, gTotal, hTotal, _node.level);

        _res.ghSums   = aGHSum;
        _res.iFeature = _iFeature;
        _res.nUnique  = nUnique;
        _res.gTotal   = gTotal;
        _res.hTotal   = hTotal;
    }

protected:
    const size_t _iFeature;
    SharedDataType& _data;
    const NodeInfoType& _node;
    ResultType& _res;
    BestSplitType& _bestSplit;
};

template<typename algorithmFPType, typename IndexType, typename MergedRType, CpuType cpu>
class FindMaxImpurityDecreaseWithGHSumsReduceTaskMerged: public GbtTask
{
public:
    using SharedDataType = SharedDataForTree<algorithmFPType, IndexType, cpu>;
    using NodeInfoType = SplitJob<algorithmFPType,cpu>;
    using ResultType = Result<algorithmFPType, cpu>;
    using GHSumType = ghSum<algorithmFPType, cpu>;
    using ImpurityType =  ImpurityData<algorithmFPType, cpu>;
    using SplitDataType  = SplitData<algorithmFPType, ImpurityType>;
    using MaxImpurityDecrease = MaxImpurityDecreaseHelper<algorithmFPType, IndexType, ImpurityType, GHSumType, SplitDataType, ResultType, cpu>;
    using GHSums = GHSumsHelper<algorithmFPType, IndexType, GHSumType, cpu>;
    using BestSplitType  = typename TreeBuilder<algorithmFPType, cpu>::BestSplit;
    using TlsType = TlsGHSumMerge<GHSumForTLS<GHSumType, cpu>, algorithmFPType, cpu>;

    FindMaxImpurityDecreaseWithGHSumsReduceTaskMerged(size_t iFeature, size_t nBlocks, SharedDataType& sharedData, const NodeInfoType& node1, const NodeInfoType& node2,
            BestSplitType& bestSplit1, BestSplitType& bestSplit2, const ResultType& prevRes, ResultType& res1, ResultType& res2, algorithmFPType** results, size_t size):
        _iFeature(iFeature), _nBlocks(nBlocks), _data(sharedData), _node1(node1), _node2(node2), _bestSplit1(bestSplit1), _bestSplit2(bestSplit2),
        _res1(res1), _res2(res2), _prevRes(prevRes), _results(results), _size(size)
    {
    }

    virtual GbtTask* execute()
    {
        const size_t nUnique = _data.ctx.dataHelper().indexedFeatures().numIndices(_iFeature);

        _res1.isFailed = true;
        _res1.ghSums = _data.GH_SUMS_BUF->singleGHSums.get(_iFeature).getBlockFromStorage();
        _res1.gTotal = 0;
        _res1.hTotal = 0;
        _res1.iFeature = _iFeature;
        _res1.nUnique = nUnique;

        const size_t iStart = _data.GH_SUMS_BUF->nUniquesArr[_iFeature];
        const size_t iEnd = iStart + nUnique;

        MergeGHSums<algorithmFPType, IndexType, cpu>::run(nUnique, iStart, iEnd, _results, _size, _res1);

        daal::threader_for(2, 2, [&](size_t iBlock)
        {
            if(iBlock == 0)
            {
                // TODO: check for hasDiffFeatureValues()

                const bool featureUnordered = _data.ctx.featTypes().isUnordered(_iFeature);
                algorithmFPType bestImpDec;
                int iBestFeat;
                _bestSplit1.safeGetData(bestImpDec, iBestFeat); // TODO: use them
                SplitDataType split(bestImpDec, featureUnordered);

                DAAL_INT idxFeatureBestSplit = -1;
                auto& par = _data.ctx.par();
                MaxImpurityDecrease::find(_node1.n, par.minObservationsInLeafNode, par.lambda, split, _res1, idxFeatureBestSplit, featureUnordered, _data, _iFeature);

                if(idxFeatureBestSplit >= 0)
                {
                    _bestSplit1.update(split, idxFeatureBestSplit, _iFeature);
                    _res1.isFailed = false;
                }
            }
            else
            {
                auto* aGHSum = _data.GH_SUMS_BUF->singleGHSums.get(_iFeature).getBlockFromStorage();

                DAAL_ASSERT(aGHSum); //TODO: return status
                const algorithmFPType gTotal = _prevRes.gTotal - _res1.gTotal;
                const algorithmFPType hTotal = _prevRes.hTotal - _res1.hTotal;

                GHSums::computeDiff(nUnique, _prevRes.ghSums, _res1.ghSums, aGHSum);

                _res2.ghSums  = aGHSum;
                _res2.iFeature = _iFeature;
                _res2.nUnique = nUnique;
                _res2.gTotal  = gTotal;
                _res2.hTotal  = hTotal;

                // TODO: check for hasDiffFeatureValues()

                const bool featureUnordered = _data.ctx.featTypes().isUnordered(_iFeature);
                algorithmFPType bestImpDec;
                int iBestFeat;
                _bestSplit1.safeGetData(bestImpDec, iBestFeat); // TODO: use them
                SplitDataType split(bestImpDec, featureUnordered);

                DAAL_INT idxFeatureBestSplit = -1;
                auto& par = _data.ctx.par();

                MaxImpurityDecrease::find(_node2.n, par.minObservationsInLeafNode, par.lambda, split, _res2, idxFeatureBestSplit, featureUnordered, _data, _iFeature);

                if(idxFeatureBestSplit >= 0)
                {
                    _bestSplit2.update(split, idxFeatureBestSplit, _iFeature);
                    _res2.isFailed = false;
                }
            }
        });
        return nullptr;
    }


protected:
    const size_t _iFeature;
    const size_t _nBlocks;
    SharedDataType& _data;
    const NodeInfoType& _node1;
    const NodeInfoType& _node2;
    BestSplitType& _bestSplit1;
    BestSplitType& _bestSplit2;
    ResultType& _res1;
    ResultType& _res2;
    const ResultType& _prevRes;
    algorithmFPType** _results;
    const size_t _size;
};

template<typename algorithmFPType, typename IndexType, typename MergedRType, CpuType cpu>
class FindMaxImpurityDecreaseWithGHSumsReduceTask: public GbtTask
{
public:
    using SharedDataType = SharedDataForTree<algorithmFPType, IndexType, cpu>;
    using NodeInfoType = SplitJob<algorithmFPType,cpu>;
    using ResultType = Result<algorithmFPType, cpu>;
    using GHSumType = ghSum<algorithmFPType, cpu>;
    using ImpurityType =  ImpurityData<algorithmFPType, cpu>;
    using SplitDataType  = SplitData<algorithmFPType, ImpurityType>;
    using MaxImpurityDecrease = MaxImpurityDecreaseHelper<algorithmFPType, IndexType, ImpurityType, GHSumType, SplitDataType, ResultType, cpu>;
    using GHSums = GHSumsHelper<algorithmFPType, IndexType, GHSumType, cpu>;
    using BestSplitType  = typename TreeBuilder<algorithmFPType, cpu>::BestSplit;


    FindMaxImpurityDecreaseWithGHSumsReduceTask(size_t iFeature, size_t nBlocks, SharedDataType& sharedData, const NodeInfoType& node1,
            BestSplitType& bestSplit1,   ResultType& res1, algorithmFPType** results, size_t size):
        _iFeature(iFeature), _nBlocks(nBlocks), _data(sharedData), _node1(node1), _bestSplit1(bestSplit1),
        _res1(res1), _results(results), _size(size)
    {
    }

    virtual GbtTask* execute()
    {
        const size_t nUnique = _data.ctx.dataHelper().indexedFeatures().numIndices(_iFeature);

        _res1.isFailed = true;
        _res1.ghSums = _data.GH_SUMS_BUF->singleGHSums.get(_iFeature).getBlockFromStorage();
        _res1.gTotal = 0;
        _res1.hTotal = 0;
        _res1.iFeature = _iFeature;
        _res1.nUnique = nUnique;

        const size_t iStart = _data.GH_SUMS_BUF->nUniquesArr[_iFeature];
        const size_t iEnd = iStart + nUnique;

        MergeGHSums<algorithmFPType, IndexType, cpu>::run(nUnique, iStart, iEnd, _results, _size, _res1);

        // TODO: check for hasDiffFeatureValues()

        const bool featureUnordered = _data.ctx.featTypes().isUnordered(_iFeature);

        algorithmFPType bestImpDec;
        int iBestFeat;
        _bestSplit1.safeGetData(bestImpDec, iBestFeat); // TODO: use them
        SplitDataType split(bestImpDec, featureUnordered);

        DAAL_INT idxFeatureBestSplit = -1;
        auto& par = _data.ctx.par();
        MaxImpurityDecrease::find(_node1.n, par.minObservationsInLeafNode, par.lambda, split, _res1, idxFeatureBestSplit, featureUnordered, _data, _iFeature);

        if(idxFeatureBestSplit >= 0)
        {
            _bestSplit1.update(split, idxFeatureBestSplit, _iFeature);
            _res1.isFailed = false;
        }
        return nullptr;
    }

protected:
    const size_t _iFeature;
    const size_t _nBlocks;
    SharedDataType& _data;
    const NodeInfoType& _node1;
    BestSplitType& _bestSplit1;
    ResultType& _res1;
    algorithmFPType** _results;
    size_t _size;
};

template<typename algorithmFPType, typename IndexType, CpuType cpu>
class ComputeGHSumsByRowsTask: public GbtTask
{
public:
    using GHSumType = ghSum<algorithmFPType, cpu>;
    using SharedDataType = SharedDataForTree<algorithmFPType, IndexType, cpu>;
    using NodeInfoType = SplitJob<algorithmFPType,cpu>;
    using ResultType = TVector<Result<algorithmFPType, cpu>, cpu, ScalableAllocator<cpu>>;
    using GHSums = GHSumsHelper<algorithmFPType, IndexType, GHSumType, cpu>;
    using TlsType = TlsGHSumMerge<GHSumForTLS<GHSumType, cpu>, algorithmFPType, cpu>;

    ComputeGHSumsByRowsTask(size_t iBlock, size_t blockSize, SharedDataType& data, const NodeInfoType& nodeInfo,
            TlsType* res): _iBlock(iBlock), _blockSize(blockSize), _data(data), _node(nodeInfo), _res(res)
    {
    }

    virtual GbtTask* execute()
    {
        const IndexType* indexedFeature = _data.GH_SUMS_BUF->newFI;
        IndexType* aIdx = _data.aIdx;
        const IndexType nFeatures = _data.ctx.nFeatures();

        const size_t iStart = _iBlock*_blockSize + _node.iStart;
        const size_t iEnd = (((_iBlock+1)*_blockSize > _node.n) ? _node.iStart + _node.n : iStart + _blockSize);

        auto* local = _res->local();
        GHSumType* aGHSum = local->ghSum;
        algorithmFPType* aGHSumFP = (algorithmFPType*)local->ghSum;

        if (!local->isInitilized)
        {
            GHSums::fillByZero(_data.GH_SUMS_BUF->nDiffFeatMax, aGHSum);
            local->isInitilized = true;
        }

        algorithmFPType* pgh = (algorithmFPType*)_data.ctx.grad(_data.iTree);
        ComputeGHSumByRows<IndexType, algorithmFPType, cpu>::run(aGHSumFP, indexedFeature, aIdx, pgh, nFeatures, iStart, iEnd, _node.iStart + _node.n);
        return nullptr;
    }

protected:
    const size_t _iBlock;
    const size_t _blockSize;
    SharedDataType& _data;
    const NodeInfoType& _node;
    TlsType* _res;
};

} /* namespace hist */
} /* namespace internal */
} /* namespace training */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif