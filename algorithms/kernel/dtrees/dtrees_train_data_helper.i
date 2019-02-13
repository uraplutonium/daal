/* file: dtrees_train_data_helper.i */
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
//  Implementation of auxiliary functions for decision trees train algorithms
//  (defaultDense) method.
//--
*/

#ifndef __DTREES_TRAIN_DATA_HELPER_I__
#define __DTREES_TRAIN_DATA_HELPER_I__

#include "service_memory.h"
#include "daal_defines.h"
#include "service_memory.h"
#include "service_rng.h"
#include "service_numeric_table.h"
#include "service_data_utils.h"
#include "service_sort.h"
#include "service_math.h"
#include "dtrees_feature_type_helper.i"

typedef int IndexType;

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::algorithms::dtrees::internal;

namespace daal
{
namespace algorithms
{
namespace dtrees
{
namespace training
{
namespace internal
{

//////////////////////////////////////////////////////////////////////////////////////////
// Service functions, compare real values with tolerance
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
bool isPositive(algorithmFPType val, algorithmFPType eps = algorithmFPType(10)*daal::services::internal::EpsilonVal<algorithmFPType>::get())
{
    return (val > eps);
}

template <typename algorithmFPType, CpuType cpu>
bool isZero(algorithmFPType val, algorithmFPType eps = algorithmFPType(10)*daal::services::internal::EpsilonVal<algorithmFPType>::get())
{
    return (val <= eps) && (val >= -eps);
}

template <typename algorithmFPType, CpuType cpu>
bool isGreater(algorithmFPType val1, algorithmFPType val2)
{
    return isPositive<algorithmFPType, cpu>(val1 - val2);
}

//////////////////////////////////////////////////////////////////////////////////////////
// Service function, randomly permutes given array
//////////////////////////////////////////////////////////////////////////////////////////
template <CpuType cpu>
void shuffle(void* state, size_t n, IndexType* dst)
{
    RNGs<IndexType, cpu> rng;
    IndexType idx[2];
    for(size_t i = 0; i < n; ++i)
    {
        rng.uniform(2, idx, state, 0, n);
        daal::services::internal::swap<cpu, IndexType>(dst[idx[0]], dst[idx[1]]);
    }
}

template <CpuType cpu>
void shuffle(void* state, size_t n, IndexType* dst, IndexType* auxBuf)
{
    RNGs<IndexType, cpu> rng;
    rng.uniform(n, auxBuf, state, 0, n);
    for(size_t i = 0; i < n; ++i)
        daal::services::internal::swap<cpu, IndexType>(dst[i], dst[auxBuf[i]]);
}

//////////////////////////////////////////////////////////////////////////////////////////
// Service structure, keeps response-dependent split data
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, typename TImpurityData>
struct SplitData
{
    TImpurityData left;
    algorithmFPType featureValue;
    volatile algorithmFPType impurityDecrease;
    size_t nLeft;
    size_t iStart;
    bool featureUnordered;
    SplitData() : impurityDecrease(-daal::services::internal::MaxVal<algorithmFPType>::get()) {}
    SplitData(algorithmFPType impDecr, bool bFeatureUnordered) : impurityDecrease(impDecr), featureUnordered(bFeatureUnordered){}
    SplitData(const SplitData& o) = delete;
    void copyTo(SplitData& o) const
    {
        o.featureValue = featureValue;
        o.nLeft = nLeft;
        o.iStart = iStart;
        o.left = left;
        o.featureUnordered = featureUnordered;
        o.impurityDecrease = impurityDecrease;
    }
};

template <typename TResponse>
struct SResponse
{
    TResponse val;
    IndexType idx;
};

//////////////////////////////////////////////////////////////////////////////////////////
// DataHelper. Base class for response-specific services classes.
// Keeps indices of the bootstrap samples and provides optimal access to columns in case
// of homogenious numeric table
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class DataHelperBase
{
public:
    DataHelperBase(const dtrees::internal::IndexedFeatures* indexedFeatures) : _indexedFeatures(indexedFeatures){}
    const NumericTable* data() const { return _data; }
    const dtrees::internal::IndexedFeatures& indexedFeatures() const { DAAL_ASSERT(_indexedFeatures); return *_indexedFeatures; }
    void init(const NumericTable* data, const NumericTable* resp)
    {
        _data = const_cast<NumericTable*>(data);
        _nCols = data->getNumberOfColumns();
        const HomogenNumericTable<algorithmFPType>* hmg = dynamic_cast<const HomogenNumericTable<algorithmFPType>*>(data);
        _dataDirect = (hmg ? hmg->getArray() : nullptr);
    }
    algorithmFPType getValue(size_t iCol, size_t iRow) const
    {
        if(_dataDirect)
            return _dataDirect[iRow*_nCols + iCol];

        data_management::BlockDescriptor<algorithmFPType> bd;
        _data->getBlockOfColumnValues(iCol, iRow, 1, readOnly, bd);
        algorithmFPType val = *bd.getBlockPtr();
        _data->releaseBlockOfColumnValues(bd);
        return val;
    }

protected:
    const dtrees::internal::IndexedFeatures* _indexedFeatures;
    const algorithmFPType* _dataDirect = nullptr;
    NumericTable* _data = nullptr;
    size_t _nCols = 0;
};

//////////////////////////////////////////////////////////////////////////////////////////
// DataHelper. Base class for response-specific services classes.
// Keeps indices of the bootstrap samples and provides optimal access to columns in case
// of homogenious numeric table
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, typename TResponse, CpuType cpu>
class DataHelper : public DataHelperBase<algorithmFPType, cpu>
{
public:
    typedef SResponse<TResponse> Response;
    typedef DataHelperBase<algorithmFPType, cpu> super;

public:
    DataHelper(const dtrees::internal::IndexedFeatures* indexedFeatures):
        super(indexedFeatures){}
    size_t size() const { return _aResponse.size(); }
    TResponse response(size_t i) const { return _aResponse[i].val; }
    const Response* responses() const { return _aResponse.get(); }
    bool reset(size_t n)
    {
        _aResponse.reset(n);
        return _aResponse.get() != nullptr;
    }

    virtual bool init(const NumericTable* data, const NumericTable* resp, const IndexType* aSample)
    {
        super::init(data, resp);
        DAAL_ASSERT(_aResponse.size());
        if(aSample)
        {
        const IndexType firstRow = aSample[0];
        const IndexType lastRow = aSample[_aResponse.size() - 1];
        ReadRows<algorithmFPType, cpu> bd(const_cast<NumericTable*>(resp), firstRow, lastRow - firstRow + 1);
            const auto pbd = bd.get();
        for(size_t i = 0; i < _aResponse.size(); ++i)
        {
            _aResponse[i].idx = aSample[i];
                _aResponse[i].val = TResponse(pbd[aSample[i] - firstRow]);
        }
        }
        else
        {
            ReadRows<algorithmFPType, cpu> bd(const_cast<NumericTable*>(resp), 0, _aResponse.size());
            const auto pbd = bd.get();
            for(size_t i = 0; i < _aResponse.size(); ++i)
            {
                _aResponse[i].idx = i;
                _aResponse[i].val = TResponse(pbd[i]);
            }
        }
        return true;
    }

    void getColumnValues(size_t iCol, const IndexType* aIdx, size_t n, algorithmFPType* aVal) const
    {
        if(this->_dataDirect)
        {
            for(size_t i = 0; i < n; ++i)
            {
                const IndexType iRow = getObsIdx(aIdx[i]);
                aVal[i] = this->_dataDirect[iRow*this->_nCols + iCol];
            }
        }
        else
        {
            data_management::BlockDescriptor<algorithmFPType> bd;
            for(size_t i = 0; i < n; ++i)
            {
                this->_data->getBlockOfColumnValues(iCol, getObsIdx(aIdx[i]), 1, readOnly, bd);
                aVal[i] = *bd.getBlockPtr();
                this->_data->releaseBlockOfColumnValues(bd);
            }
        }
    }

    size_t getNumOOBIndices() const
    {
        if(!_aResponse.size())
            return 0;

        size_t count = _aResponse[0].idx;
        size_t prev = count;
        for(size_t i = 1; i < _aResponse.size(); prev = _aResponse[i++].idx)
            count += (_aResponse[i].idx > (prev + 1) ? (_aResponse[i].idx - prev - 1) : 0);
        const size_t nRows = this->_data->getNumberOfRows();
        count += (nRows > (prev + 1) ? (nRows - prev - 1) : 0);
        return count;
    }

    void getOOBIndices(IndexType* dst) const
    {
        if(!_aResponse.size())
            return;

        const IndexType* savedDst = dst;
        size_t idx = _aResponse[0].idx;
        size_t iDst = 0;
        for(; iDst < idx; dst[iDst] = iDst, ++iDst);

        for(size_t iResp = 1; iResp < _aResponse.size(); idx = _aResponse[iResp].idx, ++iResp)
        {
            for(++idx; idx < _aResponse[iResp].idx; ++idx, ++iDst)
                dst[iDst] = idx;
        }

        const size_t nRows = this->_data->getNumberOfRows();
        for(++idx; idx < nRows; ++idx, ++iDst)
            dst[iDst] = idx;
        DAAL_ASSERT(iDst == getNumOOBIndices());
    }

    bool hasDiffFeatureValues(IndexType iFeature, const IndexType* aIdx, size_t n) const
    {
        if(this->indexedFeatures().numIndices(iFeature) == 1)
            return false; //single value only
        const IndexedFeatures::IndexType* indexedFeature = this->indexedFeatures().data(iFeature);
        const auto aResponse = this->_aResponse.get();
        const IndexedFeatures::IndexType idx0 = indexedFeature[aResponse[aIdx[0]].idx];
        size_t i = 1;
        for(; i < n; ++i)
        {
            const Response& r = aResponse[aIdx[i]];
            const IndexedFeatures::IndexType idx = indexedFeature[r.idx];
            if(idx != idx0)
                break;
        }
        return (i != n);
    }

protected:
    IndexType getObsIdx(size_t i) const { DAAL_ASSERT(i < _aResponse.size());  return _aResponse.get()[i].idx; }

protected:
    TArray<Response, cpu> _aResponse;
};

//partition given set of indices into the left and right parts
//corresponding to the split feature value (cut value)
//given as the index in the sorted feature values array
//returns index of the row in the dataset corresponding to the split feature value (cut value)
template <typename ResponseType, typename IndexType, typename FeatureIndexType, typename SizeType, CpuType cpu>
int doPartition(SizeType n, const IndexType* aIdx, const ResponseType* aResponse,
    const FeatureIndexType* indexedFeature, bool featureUnordered,
    IndexType idxFeatureValueBestSplit,
    IndexType* bestSplitIdxRight, IndexType* bestSplitIdx,
    SizeType nLeft) //for DAAL_ASSERT only
{
    SizeType iLeft = 0;
    SizeType iRight = 0;
    int iRowSplitVal = -1;

    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for(SizeType i = 0; i < n; ++i)
    {
        const IndexType iSample = aIdx[i];
        const IndexType iRow = aResponse[iSample].idx;
        const FeatureIndexType idx = indexedFeature[iRow];

        if((featureUnordered && (idx != idxFeatureValueBestSplit)) || ((!featureUnordered) && (idx > idxFeatureValueBestSplit)))
        {
            bestSplitIdxRight[iRight++] = iSample;
        }
        else
        {
            if(idx == idxFeatureValueBestSplit)
                iRowSplitVal = iRow;
            bestSplitIdx[iLeft++] = iSample;
        }
    }
    DAAL_ASSERT(iRight == n - nLeft);
    DAAL_ASSERT(iLeft == nLeft);
    return iRowSplitVal;
}

//partition given set of indices into the left and right parts
//corresponding to the split feature value (cut value)
//given as the index in the sorted feature values array
//returns index of the row in the dataset corresponding to the split feature value (cut value)
template <typename IndexType, typename FeatureIndexType, typename SizeType, CpuType cpu>
int doPartitionIdx(SizeType n, const IndexType* aIdx, const IndexType* aIdx2,
    const FeatureIndexType* indexedFeature, bool featureUnordered,
    IndexType idxFeatureValueBestSplit,
    IndexType* bestSplitIdxRight, IndexType* bestSplitIdx,
    SizeType nLeft) //for DAAL_ASSERT only
{
    SizeType iLeft = 0;
    SizeType iRight = 0;
    int iRowSplitVal = -1;

    if(aIdx2)
    {
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for(SizeType i = 0; i < n; ++i)
        {
            const IndexType iSample = aIdx[i];
            const IndexType iRow = aIdx2[iSample];
            const FeatureIndexType idx = indexedFeature[iRow];

            if((featureUnordered && (idx != idxFeatureValueBestSplit)) || ((!featureUnordered) && (idx > idxFeatureValueBestSplit)))
            {
                bestSplitIdxRight[iRight++] = iSample;
            }
            else
            {
                if(idx == idxFeatureValueBestSplit)
                    iRowSplitVal = iRow;
                bestSplitIdx[iLeft++] = iSample;
            }
        }
    }
    else
    {
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for(SizeType i = 0; i < n; ++i)
        {
            const IndexType iRow = aIdx[i];
            const FeatureIndexType idx = indexedFeature[iRow];

            if((featureUnordered && (idx != idxFeatureValueBestSplit)) || ((!featureUnordered) && (idx > idxFeatureValueBestSplit)))
            {
                bestSplitIdxRight[iRight++] = iRow;
            }
            else
            {
                if(idx == idxFeatureValueBestSplit)
                    iRowSplitVal = iRow;
                bestSplitIdx[iLeft++] = iRow;
            }
        }
    }
    DAAL_ASSERT(iRight == n - nLeft);
    DAAL_ASSERT(iLeft == nLeft);
    return iRowSplitVal;
}

} /* namespace internal */
} /* namespace training */
} /* namespace dtrees */
} /* namespace algorithms */
} /* namespace daal */

#endif
