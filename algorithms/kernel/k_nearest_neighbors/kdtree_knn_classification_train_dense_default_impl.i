/* file: kdtree_knn_classification_train_dense_default_impl.i */
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
//  Implementation of auxiliary functions for K-Nearest Neighbors K-D Tree (kDTreeDense) method.
//--
*/

#ifndef __KDTREE_KNN_CLASSIFICATION_TRAIN_DENSE_DEFAULT_IMPL_I__
#define __KDTREE_KNN_CLASSIFICATION_TRAIN_DENSE_DEFAULT_IMPL_I__

#define KNN_INT_RANDOM_NUMBER_GENERATOR

#include "daal_defines.h"
#include "threading.h"
#include "daal_atomic_int.h"
#include "service_memory.h"
#include "service_data_utils.h"
#include "service_math.h"
#include "service_rng.h"
#include "service_sort.h"
#include "numeric_table.h"
#include "kdtree_knn_classification_model_impl.h"
#include "kdtree_knn_classification_train_kernel.h"
#include "kdtree_knn_impl.i"

#if defined(__INTEL_COMPILER_BUILD_DATE)
    #include <immintrin.h>
#endif

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace training
{
namespace internal
{

using namespace daal::services::internal;
using namespace daal::services;
using namespace daal::internal;
using namespace daal::data_management;
using namespace kdtree_knn_classification::internal;

#define __BBOX_LOWER 0
#define __BBOX_UPPER 1

template <typename T, CpuType cpu>
class Queue
{
public:
    Queue() : _data(nullptr)
    {
    }

    ~Queue()
    {
        services::daal_free(_data);
    }

    bool init(size_t size)
    {
        clear();
        _first = _count = 0;
        _last = _sizeMinus1 = (_size = size) - 1;
        return ((_data = static_cast<T *>(daal_malloc(size * sizeof(T)))) != nullptr);
    }

    void clear()
    {
        daal_free(_data);
        _data = nullptr;
    }

    void reset()
    {
        _first = _count = 0;
        _last = _sizeMinus1;
    }

    DAAL_FORCEINLINE void push(const T & value)
    {
        _data[_last = (_last + 1) & _sizeMinus1] = value;
        ++_count;
    }

    DAAL_FORCEINLINE T pop()
    {
        const T value = _data[_first++];
        _first *= (_first != _size);
        --_count;
        return value;
    }

    bool empty() const { return (_count == 0); }

    size_t size() const { return _count; }

private:
    T * _data;
    size_t _first;
    size_t _last;
    size_t _count;
    size_t _size;
    size_t _sizeMinus1;
};

struct BuildNode
{
    size_t start;
    size_t end;
    size_t nodePos;
    size_t queueOrStackPos;
};

template <typename T>
struct BoundingBox
{
    T lower;
    T upper;
};

template <typename algorithmFpType, CpuType cpu>
struct IndexValuePair
{
    algorithmFpType value;
    size_t idx;

    inline bool operator< (const IndexValuePair & rhs) const { return (value < rhs.value); }
};

template <typename algorithmFpType, CpuType cpu>
int compareIndexValuePair(const void * p1, const void * p2)
{
    typedef IndexValuePair<algorithmFpType, cpu> IVPair;

    const IVPair & v1 = *static_cast<const IVPair *>(p1);
    const IVPair & v2 = *static_cast<const IVPair *>(p2);
    return (v1.value < v2.value) ? -1 : 1;
}

template <typename algorithmFpType, CpuType cpu>
void BaseKernel<algorithmFpType, cpu>::allocateTableData(size_t rowCount, NumericTable & table)
{
    if (table.getDataMemoryStatus() == NumericTableIface::notAllocated)
    {
        table.setNumberOfRows(rowCount);
        table.allocateDataMemory();
    }
    else
    {
        table.setNumberOfRows(rowCount);
    }
}

template <typename algorithmFpType, CpuType cpu>
void BaseKernel<algorithmFpType, cpu>::setNumericTableValue(NumericTable & table, size_t value)
{
    typedef int IntermediateType;

    BlockDescriptor<IntermediateType> tableBD;
    table.getBlockOfRows(0, 1, writeOnly, tableBD);
    IntermediateType * const dtable = tableBD.getBlockPtr();
    *dtable = static_cast<IntermediateType>(value);
    table.releaseBlockOfRows(tableBD);
}

template <typename algorithmFpType, CpuType cpu>
size_t BaseKernel<algorithmFpType, cpu>::calculateColor(size_t loop, size_t loops, size_t nodeCount, size_t nodeIndex)
{
    const size_t p2 = static_cast<size_t>(1) << loop;
    const size_t b = nodeCount - (nodeCount / p2);
    const size_t color = (nodeIndex & b) / (static_cast<size_t>(1) << (loops - loop));
    return color;
}

template <typename algorithmFpType, CpuType cpu>
void BaseKernel<algorithmFpType, cpu>::copyNTRows(size_t firstSrcRow, size_t firstDestRow, size_t rowCount, const NumericTable & src,
                                                  NumericTable & dest)
{
    typedef BlockDescriptor<algorithmFpType> BD;

    BD destBD, srcBD;
    dest.getBlockOfRows(firstDestRow, rowCount, data_management::writeOnly, destBD);
    const_cast<NumericTable &>(src).getBlockOfRows(firstSrcRow, rowCount, data_management::readOnly, srcBD);
    daal_memcpy_s(destBD.getBlockPtr(), destBD.getNumberOfColumns() * destBD.getNumberOfRows() * sizeof(algorithmFpType),
                  srcBD.getBlockPtr(), srcBD.getNumberOfColumns() * srcBD.getNumberOfRows() * sizeof(algorithmFpType));
    const_cast<NumericTable &>(src).releaseBlockOfRows(srcBD);
    dest.releaseBlockOfRows(destBD);
}

template <typename algorithmFpType, CpuType cpu>
Status BaseKernel<algorithmFpType, cpu>::buildKDTree(NumericTable & x, NumericTable & y, int seed,
                                                     SharedPtr<KDTreeTable> & kDTreeTable,
                                                     size_t & rootKDTreeNodeIndex,
                                                     size_t & lastContiguousKDTreeNodeIndex)
{
    Status status;

    typedef daal::internal::Math<algorithmFpType, cpu> Math;
    typedef BoundingBox<algorithmFpType> BBox;

    const size_t xRowCount = x.getNumberOfRows();
    if (xRowCount > 0)
    {
        const algorithmFpType base = 2.0;
        const size_t maxKDTreeNodeCount = ((size_t)Math::sPowx(base, Math::sCeil(Math::sLog(base * xRowCount - 1) / Math::sLog(base)))
            * __KDTREE_MAX_NODE_COUNT_MULTIPLICATION_FACTOR) / __KDTREE_LEAF_BUCKET_SIZE + 1;
        kDTreeTable.reset(new KDTreeTable(maxKDTreeNodeCount));

        size_t * const indexes  = static_cast<size_t *>(daal_malloc(xRowCount * sizeof(size_t)));
        for (size_t i = 0; i < xRowCount; ++i)
        {
            indexes[i] = i;
        }

        Queue<BuildNode, cpu> q;
        BBox * bboxQ = nullptr;

        DAAL_CHECK_STATUS(status, buildFirstPartOfKDTree(q, bboxQ, x, kDTreeTable, rootKDTreeNodeIndex,
                                                         lastContiguousKDTreeNodeIndex, indexes, seed));
        DAAL_CHECK_STATUS(status, buildSecondPartOfKDTree(q, bboxQ, x, kDTreeTable, rootKDTreeNodeIndex,
                                                          lastContiguousKDTreeNodeIndex, indexes, seed));
        DAAL_CHECK_STATUS(status, rearrangePoints(x, indexes));
        DAAL_CHECK_STATUS(status, rearrangePoints(y, indexes));

        daal_free(bboxQ);
        daal_free(indexes);
    }
    else
    {
        kDTreeTable.reset(new KDTreeTable());
        rootKDTreeNodeIndex = 0;
        lastContiguousKDTreeNodeIndex = 0;
    }

    return status;
}

template <typename algorithmFpType, CpuType cpu>
Status BaseKernel<algorithmFpType, cpu>::
    buildFirstPartOfKDTree(Queue<BuildNode, cpu> & q, BoundingBox<algorithmFpType> * & bboxQ,
                           const NumericTable & x, SharedPtr<KDTreeTable> & kDTreeTable,
                           size_t & rootKDTreeNodeIndex, size_t & lastContiguousKDTreeNodeIndex,
                           size_t * indexes, int seed)
{
    Status status;

    typedef daal::internal::Math<algorithmFpType, cpu> Math;
    typedef BoundingBox<algorithmFpType> BBox;

    const auto maxThreads = threader_get_threads_number();
    const algorithmFpType base = 2.0;
    const size_t queueSize = 2 * Math::sPowx(base, Math::sCeil(Math::sLog(__KDTREE_FIRST_PART_LEAF_NODES_PER_THREAD * maxThreads)
                                                               / Math::sLog(base)));
    const size_t firstPartLeafNodeCount = queueSize / 2;
    q.init(queueSize);
    const size_t xColumnCount = x.getNumberOfColumns();
    const size_t xRowCount = x.getNumberOfRows();
    const size_t bboxSize = queueSize * xColumnCount;
    bboxQ = static_cast<BBox *>(daal_malloc(bboxSize * sizeof(BBox), sizeof(BBox)));
    rootKDTreeNodeIndex = 0;
    lastContiguousKDTreeNodeIndex = 0;
    BBox * bboxCur = nullptr;
    BBox * bboxLeft = nullptr;
    BBox * bboxRight = nullptr;
    BuildNode bn, bnLeft, bnRight;
    bn.start = 0;
    bn.end = xRowCount;
    bn.nodePos = lastContiguousKDTreeNodeIndex++;
    bn.queueOrStackPos = bn.nodePos;
    bboxCur = &bboxQ[bn.queueOrStackPos * xColumnCount];
    DAAL_CHECK_STATUS(status, computeLocalBoundingBoxOfKDTree(bboxCur, x, indexes));

    q.push(bn);

    size_t depth = 0;
    size_t maxNodeCountForCurrentDepth = 1;

    size_t sophisticatedSampleIndexes[__KDTREE_DIMENSION_SELECTION_SIZE];
    algorithmFpType sophisticatedSampleValues[__KDTREE_DIMENSION_SELECTION_SIZE];
    const size_t subSampleCount = xRowCount / __KDTREE_SEARCH_SKIP + 1;
    algorithmFpType * subSamples = static_cast<algorithmFpType *>(daal_malloc(subSampleCount * sizeof(algorithmFpType)));

    while (maxNodeCountForCurrentDepth < firstPartLeafNodeCount)
    {
        for (size_t i = 0; i < maxNodeCountForCurrentDepth; ++i)
        {
            bn = q.pop();
            KDTreeNode & curNode = *(static_cast<KDTreeNode *>(kDTreeTable->getArray()) + bn.nodePos);
            bboxCur = &bboxQ[bn.queueOrStackPos * xColumnCount];
            if (bn.end - bn.start > __KDTREE_LEAF_BUCKET_SIZE)
            {
                const size_t d = selectDimensionSophisticated(bn.start, bn.end, sophisticatedSampleIndexes, sophisticatedSampleValues,
                                                              __KDTREE_DIMENSION_SELECTION_SIZE, x, indexes, seed);
                const algorithmFpType approximatedMedian = computeApproximatedMedianInParallel(bn.start, bn.end, d, bboxCur[d].upper, x, indexes,
                                                                                               seed, subSamples, subSampleCount, status);
                const size_t idx = adjustIndexesInParallel(bn.start, bn.end, d, approximatedMedian, x, indexes);
                curNode.cutPoint = approximatedMedian;
                curNode.dimension = d;
                curNode.leftIndex = lastContiguousKDTreeNodeIndex++;
                curNode.rightIndex = lastContiguousKDTreeNodeIndex++;

                bnLeft.start = bn.start;
                bnLeft.end = idx;
                bnLeft.queueOrStackPos = bnLeft.nodePos = curNode.leftIndex;
                bboxLeft = &bboxQ[bnLeft.queueOrStackPos * xColumnCount];
                copyBBox(bboxLeft, bboxCur, xColumnCount);
                bboxLeft[d].upper = approximatedMedian;
                q.push(bnLeft);

                bnRight.start = idx;
                bnRight.end = bn.end;
                bnRight.queueOrStackPos = bnRight.nodePos = curNode.rightIndex;
                bboxRight = &bboxQ[bnRight.queueOrStackPos * xColumnCount];
                copyBBox(bboxRight, bboxCur, xColumnCount);
                bboxRight[d].lower = approximatedMedian;
                q.push(bnRight);
            }
            else
            { // Should be leaf node.
                curNode.cutPoint = 0;
                curNode.dimension = __KDTREE_NULLDIMENSION;
                curNode.leftIndex = bn.start;
                curNode.rightIndex = bn.end;

                if (q.empty())
                {
                    break;
                }
            }
        }

        if (q.empty())
        {
            break;
        }

        ++depth;
        maxNodeCountForCurrentDepth = static_cast<size_t>(1) << depth;
    }

    daal_free(subSamples);

    return status;
}

template <typename algorithmFpType, CpuType cpu>
Status BaseKernel<algorithmFpType, cpu>::
    computeLocalBoundingBoxOfKDTree(BoundingBox<algorithmFpType> * bbox,
                                    const NumericTable & x, const size_t * indexes)
{
    Status status;

    typedef BoundingBox<algorithmFpType> BBox;
    typedef daal::data_feature_utils::internal::MaxVal<algorithmFpType, cpu> MaxVal;

    const size_t xRowCount = x.getNumberOfRows();
    const size_t xColumnCount = x.getNumberOfColumns();

    const size_t rowsPerBlock = 128;
    const size_t blockCount = (xRowCount + rowsPerBlock - 1) / rowsPerBlock;
    data_management::BlockDescriptor<algorithmFpType> columnBD;
    for (size_t j = 0; j < xColumnCount; ++j)
    {
        bbox[j].upper = - MaxVal::get();
        bbox[j].lower = MaxVal::get();

        const_cast<NumericTable &>(x).getBlockOfColumnValues(j, 0, xRowCount, readOnly, columnBD);
        const algorithmFpType * const dx = columnBD.getBlockPtr();

        daal::tls<BBox *> bboxTLS([=, &status]()-> BBox *
        {
            BBox * const ptr = service_scalable_calloc<BBox, cpu>(1);
            if (ptr)
            {
                ptr->lower = MaxVal::get();
                ptr->upper = - MaxVal::get();
            }
            else { status.add(services::ErrorMemoryAllocationFailed); }
            return ptr;
        } );

        if(!status.ok()) return status;

        daal::threader_for(blockCount, blockCount, [=, &bboxTLS](int iBlock)
        {
            BBox * const bboxLocal = bboxTLS.local();
            if (bboxLocal)
            {
                const size_t first = iBlock * rowsPerBlock;
                const size_t last = min<cpu>(static_cast<decltype(xRowCount)>(first + rowsPerBlock), xRowCount);

                if (first < last)
                {
                    BBox b;
                    size_t i = first;
                    b.upper = dx[indexes[i]];
                    b.lower = dx[indexes[i]];
                    PRAGMA_IVDEP
                    for (++i; i < last; ++i)
                    {
                        if (b.lower > dx[indexes[i]]) { b.lower = dx[indexes[i]]; }
                        if (b.upper < dx[indexes[i]]) { b.upper = dx[indexes[i]]; }
                    }

                    if (bboxLocal->upper < b.upper) { bboxLocal->upper = b.upper; }
                    if (bboxLocal->lower > b.lower) { bboxLocal->lower = b.lower; }
                }
            }
        } );

        bboxTLS.reduce([=](BBox * v) -> void
        {
            if (v)
            {
                if (bbox[j].lower > v->lower) { bbox[j].lower = v->lower; }
                if (bbox[j].upper < v->upper) { bbox[j].upper = v->upper; }
                service_scalable_free<BBox, cpu>(v);
            }
        } );

        const_cast<NumericTable &>(x).releaseBlockOfColumnValues(columnBD);
    }

    return status;
}

template <typename algorithmFpType, CpuType cpu>
size_t BaseKernel<algorithmFpType, cpu>::
    selectDimensionSophisticated(size_t start, size_t end, size_t * sampleIndexes, algorithmFpType * sampleValues, size_t sampleCount,
                                 const NumericTable & x, const size_t * indexes, int seed)
{
    const size_t elementCount = min<cpu>(end - start, sampleCount);
    const size_t xColumnCount = x.getNumberOfColumns();
    const size_t xRowCount = x.getNumberOfRows();

    algorithmFpType maxVarianceValue = 0;
    size_t maxVarianceDim = 0;

    if (end - start < sampleCount)
    {
        data_management::BlockDescriptor<algorithmFpType> columnBD;
        for (size_t j = 0; j < xColumnCount; ++j)
        {
            const_cast<NumericTable &>(x).getBlockOfColumnValues(j, 0, xRowCount, readOnly, columnBD);
            const algorithmFpType * const dx = columnBD.getBlockPtr();

            PRAGMA_IVDEP
            for (size_t i = 0; i < elementCount; ++i)
            {
                sampleValues[i] = dx[indexes[start + i]];
            }

            algorithmFpType meanValue = 0;

            for (size_t i = 0; i < elementCount; ++i)
            {
                meanValue += sampleValues[i];
            }

            meanValue /= static_cast<algorithmFpType>(elementCount);

            algorithmFpType varValue = 0;
            for (size_t i = 0; i < elementCount; ++i)
            {
                varValue += (sampleValues[i] - meanValue) * (sampleValues[i] - meanValue);
            }

            if (varValue > maxVarianceValue)
            {
                maxVarianceValue = varValue;
                maxVarianceDim = j;
            }

            const_cast<NumericTable &>(x).releaseBlockOfColumnValues(columnBD);
        }
    }
    else
    {
        daal::internal::BaseRNGs<cpu> brng(seed);
#ifdef KNN_INT_RANDOM_NUMBER_GENERATOR
        daal::internal::RNGs<int, cpu> rng;
        int * const tempSampleIndexes = static_cast<int *>(daal_malloc(elementCount * sizeof(*tempSampleIndexes)));
        rng.uniform(elementCount, tempSampleIndexes, brng, start, end);
        for (size_t i = 0; i < elementCount; ++i) { sampleIndexes[i] = tempSampleIndexes[i]; }
        daal_free(tempSampleIndexes);
#else
        daal::internal::RNGs<size_t, cpu> rng;
        rng.uniform(elementCount, sampleIndexes, brng, start, end);
#endif
        data_management::BlockDescriptor<algorithmFpType> columnBD;
        for (size_t j = 0; j < xColumnCount; ++j)
        {
            const_cast<NumericTable &>(x).getBlockOfColumnValues(j, 0, xRowCount, readOnly, columnBD);
            const algorithmFpType * const dx = columnBD.getBlockPtr();

            PRAGMA_VECTOR_ALWAYS
            for (size_t i = 0; i < elementCount; ++i)
            {
                sampleValues[i] = dx[indexes[sampleIndexes[i]]];
            }

            algorithmFpType meanValue = 0;

            for (size_t i = 0; i < elementCount; ++i)
            {
                meanValue += sampleValues[i];
            }

            meanValue /= static_cast<algorithmFpType>(elementCount);

            algorithmFpType varValue = 0;
            for (size_t i = 0; i < elementCount; ++i)
            {
                varValue += (sampleValues[i] - meanValue) * (sampleValues[i] - meanValue);
            }
            if (varValue > maxVarianceValue)
            {
                maxVarianceValue = varValue;
                maxVarianceDim = j;
            }

            const_cast<NumericTable &>(x).releaseBlockOfColumnValues(columnBD);
        }
    }

    return maxVarianceDim;
}

template <typename algorithmFpType, CpuType cpu>
algorithmFpType BaseKernel<algorithmFpType, cpu>::
    computeApproximatedMedianInParallel(size_t start, size_t end, size_t dimension,
                                        algorithmFpType upper, const NumericTable & x,
                                        const size_t * indexes, int seed, algorithmFpType * subSamples,
                                        size_t subSampleCapacity, Status &status)
{
    algorithmFpType samples[__KDTREE_MEDIAN_RANDOM_SAMPLE_COUNT + 1];
    const size_t sampleCount = sizeof(samples) / sizeof(samples[0]);

    if (end - start <= sampleCount)
    {
        data_management::BlockDescriptor<algorithmFpType> sampleBD;
        for (size_t i = start; i < end; ++i)
        {
            const_cast<NumericTable &>(x).getBlockOfColumnValues(dimension, indexes[i], 1, readOnly, sampleBD);
            const algorithmFpType * const dx = sampleBD.getBlockPtr();
            samples[i - start] = *dx;
            const_cast<NumericTable &>(x).releaseBlockOfColumnValues(sampleBD);
        }
        daal::algorithms::internal::qSort<algorithmFpType, cpu>(end - start, samples);
        const algorithmFpType approximatedMedian = ((end - start) % 2 != 0) ? samples[(end - start) / 2] :
            (samples[(end - start) / 2 - 1] + samples[(end - start) / 2]) / 2.0;
        return approximatedMedian;
    }

    {
        daal::internal::BaseRNGs<cpu> brng(seed);
#ifdef KNN_INT_RANDOM_NUMBER_GENERATOR
        daal::internal::RNGs<int, cpu> rng;
        int pos;
#else
        daal::internal::RNGs<size_t, cpu> rng;
        size_t pos;
#endif
        data_management::BlockDescriptor<algorithmFpType> sampleBD;
        size_t i = 0;
        for (; i < sampleCount - 1; ++i)
        {
            rng.uniform(1, &pos, brng, start, end);
            const_cast<NumericTable &>(x).getBlockOfColumnValues(dimension, indexes[pos], 1, readOnly, sampleBD);
            const algorithmFpType * const dx = sampleBD.getBlockPtr();
            samples[i] = *dx;
            const_cast<NumericTable &>(x).releaseBlockOfColumnValues(sampleBD);
        }
        samples[i] = upper;
    }

    daal::algorithms::internal::qSort<algorithmFpType, cpu>(sampleCount, samples);

    typedef size_t Hist;
    Hist masterHist[__KDTREE_MEDIAN_RANDOM_SAMPLE_COUNT + 1] = {};

    data_management::BlockDescriptor<algorithmFpType> columnBD;
    const size_t xRowCount = x.getNumberOfRows();
    const_cast<NumericTable &>(x).getBlockOfColumnValues(dimension, 0, xRowCount, readOnly, columnBD);
    const algorithmFpType * const dx = columnBD.getBlockPtr();

    const auto rowsPerBlock = 64;
    const auto blockCount = (xRowCount + rowsPerBlock - 1) / rowsPerBlock;

    size_t subSampleCount = 0;
    for (size_t l = 0; l < sampleCount; l += __KDTREE_SEARCH_SKIP)
    {
        subSamples[subSampleCount++] = samples[l];
    }
    const size_t subSampleCount16 = subSampleCount / __SIMDWIDTH * __SIMDWIDTH;

    daal::tls<Hist *> histTLS([=, &status]()-> Hist *
    {
        Hist * const ptr = service_scalable_calloc<Hist, cpu>(sampleCount);
        if (!ptr) { status.add(services::ErrorMemoryAllocationFailed); }
        return ptr;
    } );
    if(!status.ok()) { return (algorithmFpType)0; }

    daal::threader_for(blockCount, blockCount, [=, &histTLS, &samples, &subSamples](int iBlock)
    {
        Hist * const hist = histTLS.local();
        if (hist)
        {
            const size_t first = start + iBlock * rowsPerBlock;
            const size_t last = min<cpu>(first + rowsPerBlock, end);

            for (size_t l = first; l < last; ++l)
            {
                const size_t bucketID = computeBucketID(samples, sampleCount, subSamples, subSampleCount, subSampleCount16, dx[indexes[l]]);
                ++hist[bucketID];
            }
        }
    } );

    histTLS.reduce([=, &masterHist](Hist * v) -> void
    {
        if (v)
        {
            for (size_t j = 0; j < sampleCount; ++j)
            {
                masterHist[j] += v[j];
            }
            service_scalable_free<Hist, cpu>(v);
        }
    } );

    const_cast<NumericTable &>(x).releaseBlockOfColumnValues(columnBD);

    size_t sumMid = 0;
    size_t i = 0;
    for (; i < sampleCount; ++i)
    {
        if (sumMid + masterHist[i] > (end - start) / 2) { break; }
        sumMid += masterHist[i];
    }

    const algorithmFpType approximatedMedian = (i + 1 < sampleCount) ? (samples[i] + samples[i + 1]) / 2 : samples[i];

    return approximatedMedian;
}

template <typename algorithmFpType, CpuType cpu>
size_t BaseKernel<algorithmFpType, cpu>::
    computeBucketID(algorithmFpType * samples, size_t sampleCount,
                    algorithmFpType * subSamples, size_t subSampleCount,
                    size_t subSampleCount16, algorithmFpType value)
{
#if (__CPUID__(DAAL_CPU) >= __avx__) && (__FPTYPE__(DAAL_FPTYPE) == __float__) && defined(__INTEL_COMPILER_BUILD_DATE)

    __m256 vValue = _mm256_set1_ps(value);
    size_t k = 0;
    for (; k < subSampleCount16; k += __SIMDWIDTH)
    {
        __m256 mask = _mm256_cmp_ps(_mm256_loadu_ps(subSamples + k), vValue, _CMP_GE_OS);
        int maskInt = _mm256_movemask_ps(mask);
        if (maskInt)
        {
            k = k + _bit_scan_forward(_mm256_movemask_ps(mask));
            break;
        }
    }

    if (k > subSampleCount16)
    {
        for (k = subSampleCount16; k < subSampleCount; ++k)
        {
            if (subSamples[k] >= value) { break; }
        }
    }

    size_t i = k * __KDTREE_SEARCH_SKIP;
    if (i > 0)
    {
        for (size_t j = i - __KDTREE_SEARCH_SKIP + 1; j <= i; j += __SIMDWIDTH)
        {
            __m256 vSamples = _mm256_loadu_ps(samples + j);
            __m256 mask = _mm256_cmp_ps(vSamples, vValue, _CMP_GE_OS);
            int maskInt = _mm256_movemask_ps(mask);
            if (maskInt) { return j + _bit_scan_forward(_mm256_movemask_ps(mask)); }
        }
    }

    return i;

#else // #if (__CPUID__(DAAL_CPU) >= __avx__) && (__FPTYPE__(DAAL_FPTYPE) == __float__) && defined(__INTEL_COMPILER_BUILD_DATE)

    size_t k = 0;
    for (; k < subSampleCount; ++k)
    {
        if (subSamples[k] >= value) { break; }
    }
    size_t i = k * __KDTREE_SEARCH_SKIP;
    if (i > 0)
    {
        for (size_t j = i - __KDTREE_SEARCH_SKIP + 1; j <= i; ++j)
        {
            if (samples[j] >= value) { return j; }
        }
    }
    return i;

#endif // #if (__CPUID__(DAAL_CPU) >= __avx__) && (__FPTYPE__(DAAL_FPTYPE) == __float__) && defined(__INTEL_COMPILER_BUILD_DATE)
}

template <CpuType cpu, typename ForwardIterator1, typename ForwardIterator2>
static ForwardIterator2 swapRanges(ForwardIterator1 first1, ForwardIterator1 last1, ForwardIterator2 first2)
{
    while (first1 != last1)
    {
        const auto tmp = *first1; *first1 = *first2; *first2 = tmp;

        ++first1;
        ++first2;
    }
    return first2;
}

template <typename algorithmFpType, CpuType cpu>
size_t BaseKernel<algorithmFpType, cpu>::
    adjustIndexesInParallel(size_t start, size_t end, size_t dimension,
                            algorithmFpType median, const NumericTable & x, size_t * indexes)
{
    const size_t xRowCount = x.getNumberOfRows();
    data_management::BlockDescriptor<algorithmFpType> columnBD;

    const_cast<NumericTable &>(x).getBlockOfColumnValues(dimension, 0, xRowCount, readOnly, columnBD);
    const algorithmFpType * const dx = columnBD.getBlockPtr();

    const auto rowsPerBlock = 128;
    const auto blockCount = (end - start + rowsPerBlock - 1) / rowsPerBlock;
    const auto idxMultiplier = 16; // For cache line separation.

    size_t * const leftSegmentStartPerBlock = static_cast<size_t *>(daal_malloc(idxMultiplier * (blockCount + 1) * sizeof(size_t)));
    size_t * const rightSegmentStartPerBlock = static_cast<size_t *>(daal_malloc(idxMultiplier * blockCount * sizeof(size_t)));

    daal::threader_for(blockCount, blockCount, [=, &leftSegmentStartPerBlock, &rightSegmentStartPerBlock](int iBlock)
    {
        const size_t first = start + iBlock * rowsPerBlock;
        const size_t last = min<cpu>(first + rowsPerBlock, end);

        size_t left = first;
        size_t right = last - 1;

        for (;;)
        {
            while ((left <= right) && (dx[indexes[left]] <= median)) { ++left; }
            while ((left < right) && (dx[indexes[right]] > median)) { --right; }
            if ((left <= right) && (dx[indexes[right]] > median))
            {
                if (right == 0) { break; }
                --right;
            }

            if (left > right) { break; }

            swap<cpu>(indexes[left], indexes[right]);
            ++left;
            --right;
        }

        leftSegmentStartPerBlock[idxMultiplier * iBlock] = first;
        rightSegmentStartPerBlock[idxMultiplier * iBlock] = left;
    } );

    leftSegmentStartPerBlock[idxMultiplier * blockCount] = end;

    // Computes median position.
    size_t idx = start;
    for (size_t i = 0; i < blockCount; ++i)
    {
        idx += rightSegmentStartPerBlock[idxMultiplier * i] - leftSegmentStartPerBlock[idxMultiplier * i];
    }

    // Swaps the segments.
    size_t leftSegment = 0;
    size_t rightSegment = blockCount - 1;
    while (leftSegment < rightSegment)
    {
        // Find the thinner segment.
        if (leftSegmentStartPerBlock[idxMultiplier * (leftSegment + 1)] - rightSegmentStartPerBlock[idxMultiplier * leftSegment] >
            rightSegmentStartPerBlock[idxMultiplier * rightSegment] - leftSegmentStartPerBlock[idxMultiplier * rightSegment])
        { // Left chunk is bigger.
            swapRanges<cpu>(&indexes[leftSegmentStartPerBlock[idxMultiplier * rightSegment]],
                            &indexes[rightSegmentStartPerBlock[idxMultiplier * rightSegment]],
                            &indexes[rightSegmentStartPerBlock[idxMultiplier * leftSegment]]);
            rightSegmentStartPerBlock[idxMultiplier * leftSegment] += rightSegmentStartPerBlock[idxMultiplier * rightSegment]
                - leftSegmentStartPerBlock[idxMultiplier * rightSegment];
            --rightSegment;
        }
        else if (leftSegmentStartPerBlock[idxMultiplier * (leftSegment + 1)] - rightSegmentStartPerBlock[idxMultiplier * leftSegment] <
            rightSegmentStartPerBlock[idxMultiplier * rightSegment] - leftSegmentStartPerBlock[idxMultiplier * rightSegment])
        { // Right chunk is bigger.
            swapRanges<cpu>(&indexes[rightSegmentStartPerBlock[idxMultiplier * leftSegment]],
                            &indexes[leftSegmentStartPerBlock[idxMultiplier * (leftSegment + 1)]],
                            &indexes[rightSegmentStartPerBlock[idxMultiplier * rightSegment]
                                     - (leftSegmentStartPerBlock[idxMultiplier * (leftSegment + 1)]
                                        - rightSegmentStartPerBlock[idxMultiplier * leftSegment])]);
            rightSegmentStartPerBlock[idxMultiplier * rightSegment] -= leftSegmentStartPerBlock[idxMultiplier * (leftSegment + 1)]
                - rightSegmentStartPerBlock[idxMultiplier * leftSegment];
            ++leftSegment;
        }
        else
        { // Both chunks are equal.
            swapRanges<cpu>(&indexes[rightSegmentStartPerBlock[idxMultiplier * leftSegment]],
                            &indexes[leftSegmentStartPerBlock[idxMultiplier * (leftSegment + 1)]],
                            &indexes[leftSegmentStartPerBlock[idxMultiplier * rightSegment]]);
            ++leftSegment;
            --rightSegment;
        }
    }

    daal_free(leftSegmentStartPerBlock);
    daal_free(rightSegmentStartPerBlock);

    const_cast<NumericTable &>(x).releaseBlockOfColumnValues(columnBD);
    return idx;
}

template <typename algorithmFpType, CpuType cpu>
void BaseKernel<algorithmFpType, cpu>::
    copyBBox(BoundingBox<algorithmFpType> * dest, const BoundingBox<algorithmFpType> * src, size_t n)
{
    for (size_t j = 0; j < n; ++j)
    {
        dest[j] = src[j];
    }
}

template <typename algorithmFpType, CpuType cpu>
Status BaseKernel<algorithmFpType, cpu>::
    rearrangePoints(NumericTable & x, const size_t * indexes)
{
    Status status;

    const size_t xRowCount = x.getNumberOfRows();
    const size_t xColumnCount = x.getNumberOfColumns();
    const auto maxThreads = threader_get_threads_number();

    algorithmFpType * buffer = nullptr;

    data_management::BlockDescriptor<algorithmFpType> columnReadBD, columnWriteBD;

    for (size_t i = 0; i < xColumnCount; ++i)
    {
        x.getBlockOfColumnValues(i, 0, xRowCount, readOnly, columnReadBD);
        x.getBlockOfColumnValues(i, 0, xRowCount, writeOnly, columnWriteBD);
        const algorithmFpType * const rx = columnReadBD.getBlockPtr();
        algorithmFpType * const wx = columnWriteBD.getBlockPtr();
        algorithmFpType * const awx = (rx != wx) ? wx :
            (buffer ? buffer : (buffer = static_cast<algorithmFpType *>(daal_malloc(xRowCount * sizeof(algorithmFpType)))));
        if (!awx)
        {
            status.add(services::ErrorMemoryAllocationFailed);
            x.releaseBlockOfColumnValues(columnReadBD);
            x.releaseBlockOfColumnValues(columnWriteBD);
            break;
        }

        const auto rowsPerBlock = 256;
        const auto blockCount = (xRowCount + rowsPerBlock - 1) / rowsPerBlock;

        daal::threader_for(blockCount, blockCount, [=](int iBlock)
        {
            const size_t first = iBlock * rowsPerBlock;
            const size_t last = min<cpu>(static_cast<decltype(xRowCount)>(first + rowsPerBlock), xRowCount);

            size_t j = first;
            if (last > 4)
            {
                const size_t lastMinus4 = last - 4;
                for (; j < lastMinus4; ++j)
                {
                    DAAL_PREFETCH_READ_T0(&rx[indexes[j + 4]]);
                    awx[j] = rx[indexes[j]];
                }
            }
            for (; j < last; ++j)
            {
                awx[j] = rx[indexes[j]];
            }
        } );

        if (rx == wx)
        {
            daal::threader_for(blockCount, blockCount, [=](int iBlock)
            {
                const size_t first = iBlock * rowsPerBlock;
                const size_t last = min<cpu>(static_cast<decltype(xRowCount)>(first + rowsPerBlock), xRowCount);

                auto j = first;
                if (last > 4)
                {
                    const size_t lastMinus4 = last - 4;
                    for (; j < lastMinus4; ++j)
                    {
                        DAAL_PREFETCH_READ_T0(&awx[j + 4]);
                        wx[j] = awx[j];
                    }
                }
                for (; j < last; ++j)
                {
                    wx[j] = awx[j];
                }
            } );
        }

        x.releaseBlockOfColumnValues(columnReadBD);
        x.releaseBlockOfColumnValues(columnWriteBD);
    }

    daal_free(buffer);

    return status;
}

template <typename algorithmFpType, CpuType cpu>
Status BaseKernel<algorithmFpType, cpu>::
    buildSecondPartOfKDTree(Queue<BuildNode, cpu> & q, BoundingBox<algorithmFpType> * & bboxQ,
                            const NumericTable & x, SharedPtr<KDTreeTable> & kDTreeTable,
                            size_t & rootKDTreeNodeIndex, size_t & lastContiguousKDTreeNodeIndex,
                            size_t * indexes, int seed)
{
    Status status;

    typedef daal::internal::Math<algorithmFpType, cpu> Math;
    typedef BoundingBox<algorithmFpType> BBox;
    typedef IndexValuePair<algorithmFpType, cpu> IdxValue;

    if (q.size() == 0)
    {
        return status;
    }

    const size_t xRowCount = x.getNumberOfRows();
    const size_t xColumnCount = x.getNumberOfColumns();

    const algorithmFpType base = 2.0;
    const size_t expectedMaxDepth = (Math::sLog(xRowCount) / Math::sLog(base) + 1) * __KDTREE_DEPTH_MULTIPLICATION_FACTOR;
    const size_t stackSize = Math::sPowx(base, Math::sCeil(Math::sLog(expectedMaxDepth) / Math::sLog(base)));

    BuildNode * const bnQ = static_cast<BuildNode *>(daal_malloc(q.size() * sizeof(BuildNode)));
    size_t posQ = 0;
    while (q.size() > 0)
    {
        bnQ[posQ++] = q.pop();
    }

    services::Atomic<size_t> threadIndex(0);
    struct Local
    {
        Stack<BuildNode, cpu> buildStack;
        BBox * bboxes;
        size_t bboxPos;
        size_t nodeIndex;
        size_t threadIndex;
        IdxValue * inSortValues;
        IdxValue * outSortValues;
        size_t bboxesCapacity;
        KDTreeNode * extraKDTreeNodes;
        size_t extraKDTreeNodesCapacity;
        size_t * fixupQueue;
        size_t fixupQueueCapacity;
        size_t fixupQueueIndex;

        Local() : buildStack(), bboxes(nullptr), bboxPos(0), nodeIndex(0), threadIndex(0), inSortValues(nullptr), outSortValues(nullptr),
                  bboxesCapacity(0), extraKDTreeNodes(nullptr), extraKDTreeNodesCapacity(0), fixupQueue(nullptr), fixupQueueCapacity(0),
                  fixupQueueIndex(0) {}
    };

    const auto maxThreads = threader_get_threads_number();

    const auto rowsPerBlock = (posQ + maxThreads - 1) / maxThreads;
    const auto blockCount = (posQ + rowsPerBlock - 1) / rowsPerBlock;

    const size_t maxNodeCount = kDTreeTable->getNumberOfRows();
    const size_t emptyNodeCount = maxNodeCount - lastContiguousKDTreeNodeIndex;
    const size_t segment = (emptyNodeCount + maxThreads - 1) / maxThreads;
    size_t * const firstNodeIndex = static_cast<size_t *>(daal_malloc((maxThreads + 1) * sizeof(*firstNodeIndex)));
    size_t nodeIndex = lastContiguousKDTreeNodeIndex;
    for (size_t i = 0; i < maxThreads; ++i)
    {
        firstNodeIndex[i] = nodeIndex;
        nodeIndex += segment;
    }
    firstNodeIndex[maxThreads] = maxNodeCount;

    daal::tls<Local *> localTLS([=, &threadIndex, &firstNodeIndex, &stackSize, &status]()-> Local *
    {
        Local * const ptr = service_scalable_calloc<Local, cpu>(1);
        if (ptr)
        {
            ptr->bboxesCapacity = stackSize;
            ptr->fixupQueueCapacity = 1024;
            if (!(
                  ((ptr->bboxes = service_scalable_calloc<BBox, cpu>(ptr->bboxesCapacity * xColumnCount)) != nullptr) &&
                  ((ptr->inSortValues = service_scalable_calloc<IdxValue, cpu>(__KDTREE_INDEX_VALUE_PAIRS_PER_THREAD)) != nullptr) &&
                  ((ptr->outSortValues = service_scalable_calloc<IdxValue, cpu>(__KDTREE_INDEX_VALUE_PAIRS_PER_THREAD)) != nullptr) &&
                  ((ptr->fixupQueue = static_cast<size_t *>(daal_malloc(ptr->fixupQueueCapacity * sizeof(size_t)))) != nullptr) &&
                  ptr->buildStack.init(stackSize)))
            {
                status.add(services::ErrorMemoryAllocationFailed);
                service_scalable_free<IdxValue, cpu>(ptr->outSortValues);
                service_scalable_free<IdxValue, cpu>(ptr->inSortValues);
                daal_free(ptr->fixupQueue);
                service_scalable_free<BBox, cpu>(ptr->bboxes);
                service_scalable_free<Local, cpu>(ptr);
                return nullptr;
            }
            ptr->bboxPos = 0;
            ptr->threadIndex = threadIndex.inc() - 1;
            ptr->nodeIndex = firstNodeIndex[ptr->threadIndex];
        }
        else { status.add(services::ErrorMemoryAllocationFailed); }
        return ptr;
    } );
    if(!status.ok()) return status;

    SafeStatus safeStat;

    daal::threader_for(blockCount, blockCount, [=, &localTLS, &firstNodeIndex, &kDTreeTable, &x, &rowsPerBlock,
                                                   &xColumnCount, &lastContiguousKDTreeNodeIndex, &safeStat](int iBlock)
    {
        Local * const local = localTLS.local();
        if (local)
        {
            const size_t first = iBlock * rowsPerBlock;
            const size_t last = min<cpu>(first + rowsPerBlock, posQ);

            BuildNode bn, bnLeft, bnRight;
            BBox * bboxCur = nullptr, * bboxLeft = nullptr, * bboxRight = nullptr;
            KDTreeNode * curNode = nullptr;
            algorithmFpType lowerD, upperD;
            const size_t firstExtraNodeIndex = firstNodeIndex[local->threadIndex + 1];

            size_t sophisticatedSampleIndexes[__KDTREE_DIMENSION_SELECTION_SIZE];
            algorithmFpType sophisticatedSampleValues[__KDTREE_DIMENSION_SELECTION_SIZE];

            for (size_t i = first; i < last; ++i)
            {
                bn = bnQ[i];
                bboxCur = &bboxQ[bn.queueOrStackPos * xColumnCount];
                local->buildStack.push(bn);
                copyBBox(&(local->bboxes[local->bboxPos * xColumnCount]), bboxCur, xColumnCount);
                ++local->bboxPos;
                if (local->bboxPos >= local->bboxesCapacity)
                {
                    const size_t newCapacity = local->bboxesCapacity * 2;
                    BBox * const newBboxes = service_scalable_calloc<BBox, cpu>(newCapacity * xColumnCount);
                    if (!newBboxes)
                    {
                        safeStat.add(services::ErrorMemoryAllocationFailed);
                        return;
                    }
                    daal_memcpy_s(newBboxes, newCapacity * xColumnCount, local->bboxes, local->bboxesCapacity * xColumnCount);
                    BBox * const oldBboxes = local->bboxes;
                    local->bboxes = newBboxes;
                    local->bboxesCapacity = newCapacity;
                    service_scalable_free<BBox, cpu>(oldBboxes);
                }

                while (local->buildStack.size() > 0)
                {
                    bn = local->buildStack.pop();
                    --local->bboxPos;
                    bboxCur = &(local->bboxes[local->bboxPos * xColumnCount]);
                    curNode = (bn.nodePos < firstExtraNodeIndex) ? static_cast<KDTreeNode *>(kDTreeTable->getArray()) + bn.nodePos
                                                                 : &(local->extraKDTreeNodes[bn.nodePos - firstExtraNodeIndex]);

                    if (bn.end - bn.start <= __KDTREE_LEAF_BUCKET_SIZE)
                    { // Should be leaf node.
                        curNode->cutPoint = 0;
                        curNode->dimension = __KDTREE_NULLDIMENSION;
                        curNode->leftIndex = bn.start;
                        curNode->rightIndex = bn.end;
                    }
                    else // if (bn.end - bn.start <= __KDTREE_LEAF_BUCKET_SIZE)
                    {
                        if (bn.nodePos < lastContiguousKDTreeNodeIndex)
                        {
                            local->fixupQueue[local->fixupQueueIndex] = bn.nodePos;
                            ++local->fixupQueueIndex;
                            if (local->fixupQueueIndex >= local->fixupQueueCapacity)
                            {
                                const size_t newCapacity = local->fixupQueueCapacity * 2;
                                size_t * const newQueue = static_cast<size_t *>(daal_malloc(newCapacity * sizeof(size_t)));
                                daal_memcpy_s(newQueue, newCapacity * sizeof(size_t), local->fixupQueue, local->fixupQueueIndex * sizeof(size_t));
                                size_t * const oldQueue = local->fixupQueue;
                                local->fixupQueue = newQueue;
                                local->fixupQueueCapacity = newCapacity;
                                daal_free(oldQueue);
                            }
                        }

                        const auto d = selectDimensionSophisticated(bn.start, bn.end, sophisticatedSampleIndexes, sophisticatedSampleValues,
                                                                    __KDTREE_DIMENSION_SELECTION_SIZE, x, indexes, seed);
                        lowerD = bboxCur[d].lower;
                        upperD = bboxCur[d].upper;
                        const algorithmFpType approximatedMedian = computeApproximatedMedianInSerial(bn.start, bn.end, d, bboxCur[d].upper,
                                                                                                     local->inSortValues, local->outSortValues,
                                                                                                     __KDTREE_INDEX_VALUE_PAIRS_PER_THREAD, x,
                                                                                                     indexes, seed);
                        const auto idx = adjustIndexesInSerial(bn.start, bn.end, d, approximatedMedian, x, indexes);

                        curNode->cutPoint = approximatedMedian;
                        curNode->dimension = d;
                        curNode->leftIndex = (local->nodeIndex)++;
                        curNode->rightIndex = (local->nodeIndex)++;

                        if (local->nodeIndex >= firstExtraNodeIndex)
                        {
                            const size_t extraIndex = local->nodeIndex - firstExtraNodeIndex;
                            if (local->extraKDTreeNodes)
                            {
                                if (extraIndex >= local->extraKDTreeNodesCapacity)
                                {
                                    const size_t newCapacity = max<cpu>(
                                        local->extraKDTreeNodesCapacity > 0 ? local->extraKDTreeNodesCapacity * 2 : static_cast<size_t>(1024),
                                        extraIndex + 1);
                                    KDTreeNode * const newNodes = static_cast<KDTreeNode *>(daal_malloc(newCapacity * sizeof(KDTreeNode)));
                                    if (!newNodes)
                                    {
                                        safeStat.add(services::ErrorMemoryAllocationFailed);
                                        return;
                                    }
                                    daal_memcpy_s(newNodes, newCapacity * sizeof(KDTreeNode), local->extraKDTreeNodes,
                                                  local->extraKDTreeNodesCapacity * sizeof(KDTreeNode));
                                    KDTreeNode * const oldNodes = local->extraKDTreeNodes;
                                    local->extraKDTreeNodes = newNodes;
                                    local->extraKDTreeNodesCapacity = newCapacity;
                                    daal_free(oldNodes);
                                }
                            }
                            else
                            {
                                local->extraKDTreeNodesCapacity = max<cpu>(extraIndex + 1, static_cast<size_t>(1024));
                                local->extraKDTreeNodes = static_cast<KDTreeNode *>(daal_malloc(local->extraKDTreeNodesCapacity
                                                                                                * sizeof(KDTreeNode)));
                                if (!local->extraKDTreeNodes)
                                {
                                    safeStat.add(services::ErrorMemoryAllocationFailed);
                                    return;
                                }
                            }
                        }

                        // Right first to give lower node index for left.
                        bnRight.start = idx;
                        bnRight.end = bn.end;
                        bnRight.nodePos = curNode->rightIndex;
                        bnRight.queueOrStackPos = local->bboxPos;
                        ++local->bboxPos;
                        bboxRight = &local->bboxes[bnRight.queueOrStackPos * xColumnCount];
                        copyBBox(bboxRight, bboxCur, xColumnCount);
                        bboxRight[d].lower = approximatedMedian;
                        bboxRight[d].upper = upperD;
                        local->buildStack.push(bnRight);
                        bnLeft.start = bn.start;
                        bnLeft.end = idx;
                        bnLeft.nodePos = curNode->leftIndex;
                        bnLeft.queueOrStackPos = local->bboxPos;
                        ++local->bboxPos;
                        if (local->bboxPos >= local->bboxesCapacity)
                        {
                            const size_t newCapacity = local->bboxesCapacity * 2;
                            BBox * const newBboxes = service_scalable_calloc<BBox, cpu>(newCapacity * xColumnCount);
                            if (!newBboxes)
                            {
                                safeStat.add(services::ErrorMemoryAllocationFailed);
                                return;
                            }
                            daal_memcpy_s(newBboxes, newCapacity * xColumnCount, local->bboxes, local->bboxesCapacity * xColumnCount);
                            BBox * const oldBboxes = local->bboxes;
                            local->bboxes = newBboxes;
                            local->bboxesCapacity = newCapacity;
                            service_scalable_free<BBox, cpu>(oldBboxes);
                        }
                        bboxLeft = &local->bboxes[bnLeft.queueOrStackPos * xColumnCount];
                        copyBBox(bboxLeft, bboxCur, xColumnCount);
                        bboxLeft[d].lower = lowerD;
                        bboxLeft[d].upper = upperD;
                        local->buildStack.push(bnLeft);
                    } // if (bn.end - bn.start <= __KDTREE_LEAF_BUCKET_SIZE)
                } // while (local->buildStack.size() > 0)
            } // for (auto i = first; i < last; ++i)
        } // if (local)
    } );

    bool isNeedToReindex = false;
    localTLS.reduce([=, &isNeedToReindex](Local * ptr) -> void
    {
        if (ptr && ptr->extraKDTreeNodes)
        {
            isNeedToReindex = true;
        }
    } );

    DAAL_CHECK_SAFE_STATUS()

    if (isNeedToReindex)
    {
        size_t actualNodeCount = lastContiguousKDTreeNodeIndex;
        localTLS.reduce([=, &actualNodeCount](Local * ptr) -> void
        {
            if (ptr)
            {
                actualNodeCount += ptr->nodeIndex - firstNodeIndex[ptr->threadIndex];
            }
        } );

        SharedPtr<KDTreeTable> newKDTreeTable(new KDTreeTable(actualNodeCount));
        KDTreeNode * const oldRoot = static_cast<KDTreeNode *>(kDTreeTable->getArray());
        KDTreeNode * const newRoot = static_cast<KDTreeNode *>(newKDTreeTable->getArray());

        daal_memcpy_s(newRoot, actualNodeCount * sizeof(KDTreeNode), oldRoot, lastContiguousKDTreeNodeIndex * sizeof(KDTreeNode));

        size_t newNodeIndex = lastContiguousKDTreeNodeIndex;
        localTLS.reduce([=, &newNodeIndex](Local * ptr) -> void
        {
            if (ptr)
            {
                const size_t oldNodeIndex = firstNodeIndex[ptr->threadIndex];
                if (ptr->nodeIndex != oldNodeIndex)
                {
                    const size_t extraNodeIndex = firstNodeIndex[ptr->threadIndex + 1];
                    if (ptr->nodeIndex > extraNodeIndex)
                    {
                        daal_memcpy_s(&newRoot[newNodeIndex], (actualNodeCount - newNodeIndex) * sizeof(KDTreeNode),
                                      &oldRoot[oldNodeIndex], (extraNodeIndex - oldNodeIndex) * sizeof(KDTreeNode));
                        const size_t idx = newNodeIndex + (extraNodeIndex - oldNodeIndex);
                        daal_memcpy_s(&newRoot[idx], (actualNodeCount - idx) * sizeof(KDTreeNode),
                                      ptr->extraKDTreeNodes, (ptr->nodeIndex - extraNodeIndex) * sizeof(KDTreeNode));
                    }
                    else
                    {
                        daal_memcpy_s(&newRoot[newNodeIndex], (actualNodeCount - newNodeIndex) * sizeof(KDTreeNode),
                                      &oldRoot[oldNodeIndex], (ptr->nodeIndex - oldNodeIndex) * sizeof(KDTreeNode));
                    }
                    const long delta = newNodeIndex - oldNodeIndex;
                    for (size_t i = 0; i < ptr->fixupQueueIndex; ++i)
                    {
                        newRoot[ptr->fixupQueue[i]].leftIndex += delta;
                        newRoot[ptr->fixupQueue[i]].rightIndex += delta;
                    }
                    for (size_t i = newNodeIndex, end = newNodeIndex + ptr->nodeIndex - oldNodeIndex; i < end; ++i)
                    {
                        if (newRoot[i].dimension != __KDTREE_NULLDIMENSION)
                        {
                            newRoot[i].leftIndex += delta;
                            newRoot[i].rightIndex += delta;
                        }
                    }
                    newNodeIndex += ptr->nodeIndex - oldNodeIndex;
                }
            }
        } );
        kDTreeTable = newKDTreeTable;
        lastContiguousKDTreeNodeIndex = newNodeIndex;
    }

    localTLS.reduce([=](Local * ptr) -> void
    {
        if (ptr)
        {
            service_scalable_free<IdxValue, cpu>(ptr->inSortValues);
            service_scalable_free<IdxValue, cpu>(ptr->outSortValues);
            service_scalable_free<BBox, cpu>(ptr->bboxes);
            daal_free(ptr->extraKDTreeNodes);
            daal_free(ptr->fixupQueue);
            ptr->buildStack.clear();
            service_scalable_free<Local, cpu>(ptr);
        }
    } );

    daal_free(firstNodeIndex);
    daal_free(bnQ);

    return status;
}

template <typename algorithmFpType, CpuType cpu>
algorithmFpType BaseKernel<algorithmFpType, cpu>::
    computeApproximatedMedianInSerial(size_t start, size_t end, size_t dimension, algorithmFpType upper,
                                      IndexValuePair<algorithmFpType, cpu> * inSortValues,
                                      IndexValuePair<algorithmFpType, cpu> * outSortValues,
                                      size_t sortValueCount, const NumericTable & x, size_t * indexes, int seed)
{
    size_t i, j;
    const auto xRowCount = x.getNumberOfRows();
    data_management::BlockDescriptor<algorithmFpType> columnBD;
    const_cast<NumericTable &>(x).getBlockOfColumnValues(dimension, 0, xRowCount, readOnly, columnBD);
    const algorithmFpType * const dx = columnBD.getBlockPtr();
    if (end - start < sortValueCount)
    {
        const size_t elementCount = end - start;
        i = 0;
        if (elementCount > 16)
        {
            const size_t elementCountMinus16 = elementCount - 16;
            for (; i < elementCountMinus16; ++i)
            {
                DAAL_PREFETCH_READ_T0(dx + indexes[start + i + 16]);
                inSortValues[i].value = dx[indexes[start + i]];
                inSortValues[i].idx = indexes[start + i];
            }
        }
        for (; i < elementCount; ++i)
        {
            inSortValues[i].value = dx[indexes[start + i]];
            inSortValues[i].idx = indexes[start + i];
        }

        radixSort(inSortValues, elementCount, outSortValues);

        // Copy back the indexes.
        for (i = 0; i < elementCount; ++i)
        {
            indexes[start + i] = inSortValues[i].idx;
        }

        const algorithmFpType approximatedMedian = ((end - start) % 2 != 0) ? dx[indexes[start + (end - start) / 2]] :
            (dx[indexes[start + (end - start) / 2 - 1]] + dx[indexes[start + (end - start) / 2]]) / 2.0;

        const_cast<NumericTable &>(x).releaseBlockOfColumnValues(columnBD);

        return approximatedMedian;
    } // if (end - start < sortValueCount)

    size_t sampleCount = min<cpu>(static_cast<size_t>(static_cast<algorithmFpType>(end - start) * __KDTREE_SAMPLES_PERCENT / 100),
                                  static_cast<size_t>(__KDTREE_MAX_SAMPLES + 1));

    if (sampleCount < __KDTREE_MIN_SAMPLES) { sampleCount = __KDTREE_MIN_SAMPLES + 1; }

    algorithmFpType * const samples = static_cast<algorithmFpType *>(daal_malloc(sampleCount * sizeof(*samples)));

    daal::internal::BaseRNGs<cpu> brng(seed);
#ifdef KNN_INT_RANDOM_NUMBER_GENERATOR
    daal::internal::RNGs<int, cpu> rng;
    int pos;
#else
    daal::internal::RNGs<size_t, cpu> rng;
    size_t pos;
#endif
    for (i = 0; i < sampleCount - 1; ++i)
    {
        rng.uniform(1, &pos, brng, start, end);
        samples[i] = dx[indexes[pos]];
    }

    samples[i] = upper;
    daal::algorithms::internal::qSort<algorithmFpType, cpu>(sampleCount, samples);

    size_t * const hist = static_cast<size_t *>(daal_malloc(sampleCount * sizeof(*hist)));
    for (i = 0; i <sampleCount; ++i)
    {
        hist[i] = 0;
    }

    size_t subSampleCount = (end - start) / __KDTREE_SEARCH_SKIP + 1;
    algorithmFpType * const subSamples = static_cast<algorithmFpType *>(daal_malloc(subSampleCount * sizeof(*subSamples)));
    size_t subSamplesPos = 0;
    for (size_t l = 0; l < sampleCount; l += __KDTREE_SEARCH_SKIP)
    {
        subSamples[subSamplesPos++] = samples[l];
    }
    subSampleCount = subSamplesPos;
    const size_t subSampleCount16 = subSampleCount / __SIMDWIDTH * __SIMDWIDTH;
    size_t l = start;
    if (end > 2)
    {
        const size_t endMinus2 = end - 2;
        for (; l < endMinus2; ++l)
        {
            DAAL_PREFETCH_READ_T0(&dx[indexes[l + 2]]);
            const auto bucketID = computeBucketID(samples, sampleCount, subSamples, subSampleCount, subSampleCount16, dx[indexes[l]]);
            ++hist[bucketID];
        }
    }
    for (; l < end; ++l)
    {
        const auto bucketID = computeBucketID(samples, sampleCount, subSamples, subSampleCount, subSampleCount16, dx[indexes[l]]);
        ++hist[bucketID];
    }

    size_t sumMid = 0;
    for (i = 0; i < sampleCount; ++i)
    {
        if (sumMid + hist[i] >= (end - start) / 2) { break; }
        sumMid += hist[i];
    }

    const algorithmFpType approximatedMedian = (i + 1 < sampleCount) ? (samples[i] + samples[i + 1]) / 2 : samples[i];

    const_cast<NumericTable &>(x).releaseBlockOfColumnValues(columnBD);

    daal_free(samples);
    daal_free(hist);
    daal_free(subSamples);

    return approximatedMedian;
}

template <typename algorithmFpType, CpuType cpu>
size_t BaseKernel<algorithmFpType, cpu>::adjustIndexesInSerial(size_t start, size_t end, size_t dimension, algorithmFpType median,
                                                               const NumericTable & x, size_t * indexes)
{
    const size_t xRowCount = x.getNumberOfRows();
    data_management::BlockDescriptor<algorithmFpType> columnBD;
    const_cast<NumericTable &>(x).getBlockOfColumnValues(dimension, 0, xRowCount, readOnly, columnBD);
    const algorithmFpType * const dx = columnBD.getBlockPtr();

    size_t left = start;
    size_t right = end - 1;
    for (;;)
    {
        while ((left <= right) && (dx[indexes[left]] < median)) { ++left; }
        while ((left < right) && (dx[indexes[right]] >= median)) { --right; }
        if ((left <= right) && (dx[indexes[right]] >= median))
        {
            if (right == 0) { break; }
            --right;
        }

        if (left > right) { break; }

        swap<cpu>(indexes[left], indexes[right]);
        ++left;
        --right;
    }

    const size_t lim1 = left;
    right = end - 1;
    for (;;)
    {
        while ((left <= right) && (dx[indexes[left]] <= median)) { ++left; }
        while ((left < right) && (dx[indexes[right]] > median)) { --right; }
        if ((left <= right) && (dx[indexes[right]] > median))
        {
            if (right == 0) { break; }
            --right;
        }

        if (left > right) { break; }

        swap<cpu>(indexes[left], indexes[right]);
        ++left;
        --right;
    }

    const size_t lim2 = left;
    const size_t idx = (lim1 > start + (end - start) / 2) ? lim1 : (lim2 < start + (end - start) / 2) ? lim2 : start + (end - start) / 2;

    const_cast<NumericTable &>(x).releaseBlockOfColumnValues(columnBD);

    return idx;
}

template <typename algorithmFpType, CpuType cpu>
void BaseKernel<algorithmFpType, cpu>::radixSort(IndexValuePair<algorithmFpType, cpu> * inValues, size_t valueCount,
                                                 IndexValuePair<algorithmFpType, cpu> * outValues)
{
#if (__FPTYPE__(DAAL_FPTYPE) == __float__)
    typedef IndexValuePair<algorithmFpType, cpu> Item;
    typedef unsigned int IntegerType;
    const size_t histogramSize = 256;
    int histogram[histogramSize], histogramPS[histogramSize + 1];
    Item * first = inValues;
    Item * second = outValues;
    size_t valueCount4 = valueCount / 4 * 4;
    for (unsigned int pass = 0; pass < 3; ++pass)
    {
        for (size_t i = 0; i < histogramSize; ++i) { histogram[i] = 0; }
        for (size_t i = 0; i < valueCount4; i += 4)
        {
            IntegerType val1 = *reinterpret_cast<IntegerType *>(&first[i].value);
            IntegerType val2 = *reinterpret_cast<IntegerType *>(&first[i + 1].value);
            IntegerType val3 = *reinterpret_cast<IntegerType *>(&first[i + 2].value);
            IntegerType val4 = *reinterpret_cast<IntegerType *>(&first[i + 3].value);
            ++histogram[(val1 >> (pass * 8)) & 0xFF];
            ++histogram[(val2 >> (pass * 8)) & 0xFF];
            ++histogram[(val3 >> (pass * 8)) & 0xFF];
            ++histogram[(val4 >> (pass * 8)) & 0xFF];
        }
        for (size_t i = valueCount4; i < valueCount; ++i)
        {
            IntegerType val1 = *reinterpret_cast<IntegerType *>(&first[i].value);
            ++histogram[(val1 >> (pass * 8)) & 0xFF];
        }

        int sum = 0, prevSum = 0;
        for (size_t i = 0; i < histogramSize; ++i)
        {
            sum += histogram[i];
            histogramPS[i] = prevSum;
            prevSum = sum;
        }
        histogramPS[histogramSize] = prevSum;

        for (size_t i = 0; i < valueCount; ++i)
        {
            IntegerType val1 = *reinterpret_cast<IntegerType *>(&first[i].value);
            const int pos = histogramPS[(val1 >> (pass * 8)) & 0xFF]++;
            second[pos] = first[i];
        }

        Item * temp = first;
        first = second;
        second = temp;
    }
    {
        unsigned int pass = 3;
        for (size_t i = 0; i < histogramSize; ++i) { histogram[i] = 0; }
        for (size_t i = 0; i < valueCount4; i += 4)
        {
            IntegerType val1 = *reinterpret_cast<IntegerType *>(&first[i].value);
            IntegerType val2 = *reinterpret_cast<IntegerType *>(&first[i + 1].value);
            IntegerType val3 = *reinterpret_cast<IntegerType *>(&first[i + 2].value);
            IntegerType val4 = *reinterpret_cast<IntegerType *>(&first[i + 3].value);
            ++histogram[(val1 >> (pass * 8)) & 0xFF];
            ++histogram[(val2 >> (pass * 8)) & 0xFF];
            ++histogram[(val3 >> (pass * 8)) & 0xFF];
            ++histogram[(val4 >> (pass * 8)) & 0xFF];
        }
        for (size_t i = valueCount4; i < valueCount; ++i)
        {
            IntegerType val1 = *reinterpret_cast<IntegerType *>(&first[i].value);
            ++histogram[(val1 >> (pass * 8)) & 0xFF];
        }

        int sum = 0, prevSum = 0;
        for (size_t i = 0; i < histogramSize; ++i)
        {
            sum += histogram[i];
            histogramPS[i] = prevSum;
            prevSum = sum;
        }
        histogramPS[histogramSize] = prevSum;

        // Handle negative values.
        const size_t indexOfNegatives = histogramSize / 2;
        int countOfNegatives = histogramPS[histogramSize] - histogramPS[indexOfNegatives];
        // Fixes offsets for positive values.
        for (size_t i = 0; i < indexOfNegatives - 1; ++i)
        {
            histogramPS[i] += countOfNegatives;
        }
        // Fixes offsets for negative values.
        histogramPS[histogramSize - 1] = histogram[histogramSize - 1];
        for (size_t i = 0; i < indexOfNegatives - 1; ++i)
        {
            histogramPS[histogramSize - 2 - i] = histogramPS[histogramSize - 1 - i] + histogram[histogramSize - 2 - i];
        }

        for (size_t i = 0; i < valueCount; ++i)
        {
            IntegerType val1 = *reinterpret_cast<IntegerType *>(&first[i].value);
            const int bin = (val1 >> (pass * 8)) & 0xFF;
            int pos;
            if (bin >= indexOfNegatives) { pos = --histogramPS[bin]; }
            else { pos = histogramPS[bin]++; }
            second[pos] = first[i];
        }
    }
#else // #if (__FPTYPE__(DAAL_FPTYPE) == __float__)
    typedef IndexValuePair<algorithmFpType, cpu> Item;
    typedef DAAL_UINT64 IntegerType;
    const size_t histogramSize = 256;
    int histogram[histogramSize], histogramPS[histogramSize + 1];
    Item * first = inValues;
    Item * second = outValues;
    size_t valueCount4 = valueCount / 4 * 4;
    for (unsigned int pass = 0; pass < 7; ++pass)
    {
        for (size_t i = 0; i < histogramSize; ++i) { histogram[i] = 0; }
        for (size_t i = 0; i < valueCount4; i += 4)
        {
            IntegerType val1 = *reinterpret_cast<IntegerType *>(&first[i].value);
            IntegerType val2 = *reinterpret_cast<IntegerType *>(&first[i + 1].value);
            IntegerType val3 = *reinterpret_cast<IntegerType *>(&first[i + 2].value);
            IntegerType val4 = *reinterpret_cast<IntegerType *>(&first[i + 3].value);
            ++histogram[(val1 >> (pass * 8)) & 0xFF];
            ++histogram[(val2 >> (pass * 8)) & 0xFF];
            ++histogram[(val3 >> (pass * 8)) & 0xFF];
            ++histogram[(val4 >> (pass * 8)) & 0xFF];
        }
        for (size_t i = valueCount4; i < valueCount; ++i)
        {
            IntegerType val1 = *reinterpret_cast<IntegerType *>(&first[i].value);
            ++histogram[(val1 >> (pass * 8)) & 0xFF];
        }

        int sum = 0, prevSum = 0;
        for (size_t i = 0; i < histogramSize; ++i)
        {
            sum += histogram[i];
            histogramPS[i] = prevSum;
            prevSum = sum;
        }
        histogramPS[histogramSize] = prevSum;

        for (size_t i = 0; i < valueCount; ++i)
        {
            IntegerType val1 = *reinterpret_cast<IntegerType *>(&first[i].value);
            const int pos = histogramPS[(val1 >> (pass * 8)) & 0xFF]++;
            second[pos] = first[i];
        }

        Item * temp = first;
        first = second;
        second = temp;
    }
    {
        unsigned int pass = 7;
        for (size_t i = 0; i < histogramSize; ++i) { histogram[i] = 0; }
        for (size_t i = 0; i < valueCount4; i += 4)
        {
            IntegerType val1 = *reinterpret_cast<IntegerType *>(&first[i].value);
            IntegerType val2 = *reinterpret_cast<IntegerType *>(&first[i + 1].value);
            IntegerType val3 = *reinterpret_cast<IntegerType *>(&first[i + 2].value);
            IntegerType val4 = *reinterpret_cast<IntegerType *>(&first[i + 3].value);
            ++histogram[(val1 >> (pass * 8)) & 0xFF];
            ++histogram[(val2 >> (pass * 8)) & 0xFF];
            ++histogram[(val3 >> (pass * 8)) & 0xFF];
            ++histogram[(val4 >> (pass * 8)) & 0xFF];
        }
        for (size_t i = valueCount4; i < valueCount; ++i)
        {
            IntegerType val1 = *reinterpret_cast<IntegerType *>(&first[i].value);
            ++histogram[(val1 >> (pass * 8)) & 0xFF];
        }

        int sum = 0, prevSum = 0;
        for (size_t i = 0; i < histogramSize; ++i)
        {
            sum += histogram[i];
            histogramPS[i] = prevSum;
            prevSum = sum;
        }
        histogramPS[histogramSize] = prevSum;

        // Handle negative values.
        const size_t indexOfNegatives = histogramSize / 2;
        int countOfNegatives = histogramPS[histogramSize] - histogramPS[indexOfNegatives];
        // Fixes offsets for positive values.
        for (size_t i = 0; i < indexOfNegatives - 1; ++i)
        {
            histogramPS[i] += countOfNegatives;
        }
        // Fixes offsets for negative values.
        histogramPS[histogramSize - 1] = histogram[histogramSize - 1];
        for (size_t i = 0; i < indexOfNegatives - 1; ++i)
        {
            histogramPS[histogramSize - 2 - i] = histogramPS[histogramSize - 1 - i] + histogram[histogramSize - 2 - i];
        }

        for (size_t i = 0; i < valueCount; ++i)
        {
            IntegerType val1 = *reinterpret_cast<IntegerType *>(&first[i].value);
            const int bin = (val1 >> (pass * 8)) & 0xFF;
            int pos;
            if (bin >= indexOfNegatives) { pos = --histogramPS[bin]; }
            else { pos = histogramPS[bin]++; }
            second[pos] = first[i];
        }
    }
#endif // #if (__FPTYPE__(DAAL_FPTYPE) == __float__)
}

template <typename algorithmFpType, CpuType cpu>
services::Status KNNClassificationTrainBatchKernel<algorithmFpType, training::defaultDense, cpu>::
    compute(NumericTable * x, NumericTable * y, kdtree_knn_classification::Model * r, const daal::algorithms::Parameter * par)
{
    const kdtree_knn_classification::Parameter * const parameter = static_cast<const kdtree_knn_classification::Parameter *>(par);

    SharedPtr<KDTreeTable> kDTreeTable;
    size_t rootKDTreeNodeIndex, lastContiguousKDTreeNodeIndex;
    buildKDTree(*x, *y, parameter->seed, kDTreeTable, rootKDTreeNodeIndex, lastContiguousKDTreeNodeIndex);

    r->setNFeatures(x->getNumberOfColumns());
    r->impl()->setKDTreeTable(kDTreeTable);
    r->impl()->setRootNodeIndex(rootKDTreeNodeIndex);
    r->impl()->setLastNodeIndex(lastContiguousKDTreeNodeIndex);

    DAAL_RETURN_STATUS();
}

template <typename algorithmFpType, CpuType cpu>
services::Status KNNClassificationTrainDistrStep1Kernel<algorithmFpType, training::defaultDense, cpu>::
    compute(NumericTable * x, NumericTable * y, NumericTable * r, const daal::algorithms::Parameter * par)
{
    computeLocalBoundingBox(*x, *r);
    DAAL_RETURN_STATUS();
}

template <typename algorithmFpType, CpuType cpu>
void KNNClassificationTrainDistrStep1Kernel<algorithmFpType, training::defaultDense, cpu>::
    computeLocalBoundingBox(const NumericTable & x, NumericTable & r)
{
    typedef BoundingBox<algorithmFpType> BBox;
    typedef daal::data_feature_utils::internal::MaxVal<algorithmFpType, cpu> MaxVal;
    typedef BlockDescriptor<algorithmFpType> BD;

    const size_t xRowCount = x.getNumberOfRows();
    const size_t xColumnCount = x.getNumberOfColumns();

    struct Local
    {
        BBox bbox;
        BD columnBD;
    };

    BD rBD;

    const size_t rowsPerBlock = 128;
    const size_t blockCount = (xRowCount + rowsPerBlock - 1) / rowsPerBlock;
    for (size_t j = 0; j < xColumnCount; ++j)
    {
        r.getBlockOfColumnValues(j, 0, 2, writeOnly, rBD);
        algorithmFpType * const dr = rBD.getBlockPtr();

        dr[__BBOX_LOWER] = MaxVal::get();
        dr[__BBOX_UPPER] = - MaxVal::get();

        daal::tls<Local *> localTLS([=]()-> Local *
        {
            Local * const ptr = new Local;
            ptr->bbox.lower = MaxVal::get();
            ptr->bbox.upper = - MaxVal::get();
            return ptr;
        } );

        daal::threader_for(blockCount, blockCount, [=, &localTLS, &x](int iBlock)
        {
            Local * const local = localTLS.local();
            const size_t first = iBlock * rowsPerBlock;
            const size_t last = min<cpu>(first + rowsPerBlock, xRowCount);

            if (first < last)
            {
                const size_t cnt = last - first;
                const_cast<NumericTable &>(x).getBlockOfColumnValues(j, first, cnt, readOnly, local->columnBD);
                const algorithmFpType * const dx = local->columnBD.getBlockPtr();
                BBox b;
                size_t i = 0;
                b.upper = dx[i];
                b.lower = dx[i];
                PRAGMA_IVDEP
                for (++i; i < cnt; ++i)
                {
                    if (b.lower > dx[i]) { b.lower = dx[i]; }
                    if (b.upper < dx[i]) { b.upper = dx[i]; }
                }
                const_cast<NumericTable &>(x).releaseBlockOfColumnValues(local->columnBD);

                if (local->bbox.upper < b.upper) { local->bbox.upper = b.upper; }
                if (local->bbox.lower > b.lower) { local->bbox.lower = b.lower; }
            }
        } );

        localTLS.reduce([=](Local * v) -> void
        {
            if (dr[__BBOX_LOWER] > v->bbox.lower) { dr[__BBOX_LOWER] = v->bbox.lower; }
            if (dr[__BBOX_UPPER] < v->bbox.upper) { dr[__BBOX_UPPER] = v->bbox.upper; }
            delete v;
        } );

        r.releaseBlockOfColumnValues(rBD);
    }
}

template <typename algorithmFpType, CpuType cpu>
services::Status KNNClassificationTrainDistrStep2Kernel<algorithmFpType, training::defaultDense, cpu>::
    compute(size_t perNodeNTCount, NumericTable ** perNodeNTs, NumericTable * r, NumericTable * loops, const daal::algorithms::Parameter * par)
{
    size_t numberOfLoops;
    const size_t internalNodeCount = computeNumberOfLoops(perNodeNTCount, numberOfLoops);
    allocateTableData(internalNodeCount * 2, *r);
    computeGlobalBoundingBox(perNodeNTCount, perNodeNTs, *r);

    typedef int LoopsIntermediateType;
    BlockDescriptor<LoopsIntermediateType> loopsBD;
    loops->getBlockOfRows(0, 1, writeOnly, loopsBD);
    LoopsIntermediateType * const d = loopsBD.getBlockPtr();
    *d = static_cast<LoopsIntermediateType>(numberOfLoops);
    loops->releaseBlockOfRows(loopsBD);
    DAAL_RETURN_STATUS();
}

template <typename algorithmFpType, CpuType cpu>
void KNNClassificationTrainDistrStep2Kernel<algorithmFpType, training::defaultDense, cpu>::
    computeGlobalBoundingBox(size_t perNodeNTCount, NumericTable ** perNodeNTs, NumericTable & r)
{
    typedef BlockDescriptor<algorithmFpType> BD;

    if (perNodeNTCount > 0)
    {
        const size_t featureCount = perNodeNTs[0]->getNumberOfColumns();
        // const size_t internalNodeCount = r.getNumberOfRows() / 2;
        BD rBD, ntBD, restBD;
        for (size_t featureID = 0; featureID < featureCount; ++featureID)
        {
            r.getBlockOfColumnValues(featureID, 0, 2, writeOnly, rBD);
            algorithmFpType * const dr = rBD.getBlockPtr();
            perNodeNTs[0]->getBlockOfColumnValues(featureID, 0, 2, readOnly, ntBD);
            const algorithmFpType * const dnt = ntBD.getBlockPtr();
            dr[__BBOX_LOWER] = dnt[__BBOX_LOWER];
            dr[__BBOX_UPPER] = dnt[__BBOX_UPPER];
            perNodeNTs[0]->releaseBlockOfColumnValues(ntBD);
            for (size_t n = 1; n < perNodeNTCount; ++n)
            {
                perNodeNTs[n]->getBlockOfColumnValues(featureID, 0, 2, readOnly, ntBD);
                const algorithmFpType * const dnt = ntBD.getBlockPtr();
                if (dr[__BBOX_LOWER] > dnt[__BBOX_LOWER]) { dr[__BBOX_LOWER] = dnt[__BBOX_LOWER]; }
                if (dr[__BBOX_UPPER] < dnt[__BBOX_UPPER]) { dr[__BBOX_UPPER] = dnt[__BBOX_UPPER]; }
                perNodeNTs[n]->releaseBlockOfColumnValues(ntBD);
            }
            r.getBlockOfColumnValues(featureID, rBD.getNumberOfRows(), r.getNumberOfRows() - rBD.getNumberOfRows(), writeOnly, restBD);
            algorithmFpType * const drest = restBD.getBlockPtr();
            size_t pos = 0;
            for (size_t i = 0, cnt = restBD.getNumberOfRows() / 2; i < cnt; ++i)
            {
                for (size_t j = 0; j < rBD.getNumberOfRows(); ++j)
                {
                    drest[pos] = dr[j];
                    ++pos;
                }
            }
            r.releaseBlockOfColumnValues(restBD);
            r.releaseBlockOfColumnValues(rBD);
        }
    }
}

template <typename algorithmFpType, CpuType cpu>
size_t KNNClassificationTrainDistrStep2Kernel<algorithmFpType, training::defaultDense, cpu>::
    computeNumberOfLoops(size_t perNodeNTCount, size_t & numberOfLoops)
{
    size_t loops = 0;
    size_t internalNodes = 1;
    size_t i = perNodeNTCount;
    while ((i = i >> 1) > 0)
    {
        ++loops;
        internalNodes = internalNodes << 1;
    }
    internalNodes = internalNodes << 1;
    numberOfLoops = loops;
    return internalNodes;
}

template <typename algorithmFpType, CpuType cpu>
services::Status KNNClassificationTrainDistrStep3Kernel<algorithmFpType, training::defaultDense, cpu>::
    compute(NumericTable * x, NumericTable * y, NumericTable * b, NumericTable * numberOfLoops, int loop, int nodeIndex, int nodeCount,
            NumericTable * samples, NumericTable * dim, NumericTable * colorTable, const daal::algorithms::Parameter * par)
{
    const kdtree_knn_classification::Parameter * const parameter = static_cast<const kdtree_knn_classification::Parameter *>(par);

    const size_t color = calculateColor(static_cast<size_t>(loop), static_cast<size_t>(numberOfLoops->getValue<int>(0, 0)),
                                        static_cast<size_t>(nodeCount), static_cast<size_t>(nodeIndex));
    setNumericTableValue(*colorTable, color);

    const size_t idx = (static_cast<size_t>(1) << loop) + color;

    // Dimension can be calculated on group root node only, but we calculate it on every node in the group to avoid dedicated step to do it.
    const size_t d = selectDimension(*b, idx);
    setNumericTableValue(*dim, d);

    performSampling(d, *x, *b, *samples, idx, parameter->seed);
    DAAL_RETURN_STATUS();
}

template <typename algorithmFpType, CpuType cpu>
size_t KNNClassificationTrainDistrStep3Kernel<algorithmFpType, training::defaultDense, cpu>::selectDimension(const NumericTable & b, size_t idx)
{
    typedef BlockDescriptor<algorithmFpType> BD;

    const size_t bColumnCount = b.getNumberOfColumns();

    BD bBD;
    size_t bestDimension = 0;
    const_cast<NumericTable &>(b).getBlockOfColumnValues(bestDimension, idx * 2, 2, readOnly, bBD);
    const algorithmFpType * const db = bBD.getBlockPtr();
    algorithmFpType bestRange = db[__BBOX_UPPER] - db[__BBOX_LOWER];
    const_cast<NumericTable &>(b).releaseBlockOfColumnValues(bBD);
    for (size_t j = 1; j < bColumnCount; ++j)
    {
        const_cast<NumericTable &>(b).getBlockOfColumnValues(j, idx * 2, 2, readOnly, bBD);
        const algorithmFpType * const db = bBD.getBlockPtr();
        const algorithmFpType range = db[__BBOX_UPPER] - db[__BBOX_LOWER];
        if (range > bestRange)
        {
            bestRange = range;
            bestDimension = j;
        }
        const_cast<NumericTable &>(b).releaseBlockOfColumnValues(bBD);
    }
    return bestDimension;
}

template <typename algorithmFpType, CpuType cpu>
void KNNClassificationTrainDistrStep3Kernel<algorithmFpType, training::defaultDense, cpu>::
    performSampling(size_t dimension, const NumericTable & x, const NumericTable & b, NumericTable & samples, size_t idx, int seed)
{
    typedef BlockDescriptor<algorithmFpType> BD;

    const size_t sampleCount = samples.getNumberOfRows();
    const size_t xRowCount = x.getNumberOfRows();
    if (xRowCount > 0)
    {
        size_t * const sampleIndexes = static_cast<size_t *>(daal_malloc(sampleCount * sizeof(size_t)));
        daal::internal::BaseRNGs<cpu> brng(seed);
#ifdef KNN_INT_RANDOM_NUMBER_GENERATOR
        daal::internal::RNGs<int, cpu> rng;
        int * const tempSampleIndexes = static_cast<int *>(daal_malloc(sampleCount * sizeof(int)));
        rng.uniform(sampleCount, tempSampleIndexes, brng, 0, xRowCount);
        for (size_t i = 0; i < sampleCount; ++i) { sampleIndexes[i] = tempSampleIndexes[i]; }
        daal_free(tempSampleIndexes);
#else
        daal::internal::RNGs<size_t, cpu> rng;
        rng.uniform(sampleCount, sampleIndexes, brng, 0, xRowCount);
#endif

        BD samplesBD, xBD;
        samples.getBlockOfRows(0, sampleCount, writeOnly, samplesBD);
        algorithmFpType * const ds = samplesBD.getBlockPtr();
        for (size_t i = 0; i < sampleCount; ++i)
        {
            const_cast<NumericTable &>(x).getBlockOfColumnValues(dimension, sampleIndexes[i], 1, readOnly, xBD);
            const algorithmFpType * const dx = xBD.getBlockPtr();
            ds[i] = *dx;
            const_cast<NumericTable &>(x).releaseBlockOfColumnValues(xBD);
        }
        samples.releaseBlockOfRows(samplesBD);
        daal_free(sampleIndexes);
    }
    else
    { // Since no points here, use the upper as the value.
        BD bBD;
        const_cast<NumericTable &>(b).getBlockOfColumnValues(dimension, idx * 2 + __BBOX_UPPER, 1, readOnly, bBD);
        const algorithmFpType * const db = bBD.getBlockPtr();
        const algorithmFpType upper = *db;
        const_cast<NumericTable &>(b).releaseBlockOfColumnValues(bBD);

        BD samplesBD;
        samples.getBlockOfRows(0, sampleCount, writeOnly, samplesBD);
        algorithmFpType * const ds = samplesBD.getBlockPtr();
        for (size_t i = 0; i < sampleCount; ++i)
        {
            ds[i] = upper;
        }
        samples.releaseBlockOfRows(samplesBD);
    }
}

template <typename algorithmFpType, CpuType cpu>
services::Status KNNClassificationTrainDistrStep4Kernel<algorithmFpType, training::defaultDense, cpu>::
    compute(NumericTable * x, NumericTable * y, NumericTable * d, NumericTable * b, size_t nodeCount, NumericTable ** nodeSamples, NumericTable * h,
            const daal::algorithms::Parameter * par)
{
    const size_t globalSampleCount = nodeCount * __KDTREE_SAMPLES_PER_NODE + 1;
    algorithmFpType * const globalSamples = static_cast<algorithmFpType *>(daal_malloc(globalSampleCount * sizeof(algorithmFpType)));
    const size_t dimension = static_cast<size_t>(d->getValue<int>(0, 0));
    prepareGlobalSamples(nodeCount, nodeSamples, dimension, *b, globalSamples, globalSampleCount);
    buildHistogram(globalSamples, globalSampleCount, *x, dimension, *h);
    daal_free(globalSamples);
    DAAL_RETURN_STATUS();
}

template <typename algorithmFpType, CpuType cpu>
void KNNClassificationTrainDistrStep4Kernel<algorithmFpType, training::defaultDense, cpu>::
    prepareGlobalSamples(size_t nodeCount, NumericTable ** nodeSamples, size_t dimension, const NumericTable & b, algorithmFpType * globalSamples,
                         size_t globalSampleCount)
{
    typedef BlockDescriptor<algorithmFpType> BD;

    BD nBD;
    for (size_t i = 0; i < nodeCount; ++i)
    {
        nodeSamples[i]->getBlockOfRows(0, __KDTREE_SAMPLES_PER_NODE, readOnly, nBD);
        daal_memcpy_s(&globalSamples[i * __KDTREE_SAMPLES_PER_NODE], __KDTREE_SAMPLES_PER_NODE * sizeof(algorithmFpType),
                      nBD.getBlockPtr(), __KDTREE_SAMPLES_PER_NODE * sizeof(algorithmFpType));
        nodeSamples[i]->releaseBlockOfRows(nBD);
    }

    BD bBD;
    const_cast<NumericTable &>(b).getBlockOfColumnValues(dimension, __BBOX_UPPER, 1, readOnly, bBD);
    globalSamples[globalSampleCount - 1] = *bBD.getBlockPtr();
    const_cast<NumericTable &>(b).releaseBlockOfColumnValues(bBD);

    daal::algorithms::internal::qSort<algorithmFpType, cpu>(globalSampleCount - 1, globalSamples);
}

template <typename algorithmFpType, CpuType cpu>
void KNNClassificationTrainDistrStep4Kernel<algorithmFpType, training::defaultDense, cpu>::
    buildHistogram(const algorithmFpType * globalSamples, size_t globalSampleCount, const NumericTable & x, size_t dimension, NumericTable & h)
{
    typedef BlockDescriptor<algorithmFpType> BD;

    const size_t globalSampleCount16 = (globalSampleCount + 15) / 16 * 16;
    const size_t xRowCount = x.getNumberOfRows();

    struct Local
    {
        BD columnBD;
        size_t * histogram;
    };

    daal::tls<Local *> localTLS([=]()-> Local *
    {
        Local * const ptr = new Local;
        ptr->histogram = service_scalable_calloc<size_t, cpu>(globalSampleCount16);
        return ptr;
    } );

    const size_t rowsPerBlock = 64;
    const size_t blockCount = (xRowCount + rowsPerBlock - 1) / rowsPerBlock;
    daal::threader_for(blockCount, blockCount, [=, &localTLS, &globalSamples, &x](int iBlock)
    {
        Local * const local = localTLS.local();
        const size_t first = iBlock * rowsPerBlock;
        const size_t last = min<cpu>(first + rowsPerBlock, xRowCount);

        if (first < last)
        {
            const size_t cnt = last - first;
            const_cast<NumericTable &>(x).getBlockOfColumnValues(dimension, first, cnt, readOnly, local->columnBD);
            const algorithmFpType * const dx = local->columnBD.getBlockPtr();
            for (size_t i = 0; i < cnt; ++i)
            {
                const size_t bucketID = computeBucketID(globalSamples, globalSampleCount, dx[i]);
                ++local->histogram[bucketID];
            }
            const_cast<NumericTable &>(x).releaseBlockOfColumnValues(local->columnBD);
        }
    } );

    BD samplesColumnBD;
    BD countColumnBD;
    h.getBlockOfColumnValues(0, 0, globalSampleCount, writeOnly, samplesColumnBD);
    h.getBlockOfColumnValues(1, 0, globalSampleCount, writeOnly, countColumnBD);
    algorithmFpType * const ds = samplesColumnBD.getBlockPtr();
    algorithmFpType * const masterHist = countColumnBD.getBlockPtr();
    for (size_t i = 0; i < globalSampleCount; ++i)
    {
        ds[i] = globalSamples[i];
        masterHist[i] = 0;
    }
    h.releaseBlockOfColumnValues(samplesColumnBD);

    localTLS.reduce([=, &masterHist](Local * v) -> void
    {
        for (size_t j = 0; j < globalSampleCount; ++j)
        {
            masterHist[j] += v->histogram[j];
        }
        service_scalable_free<size_t, cpu>(v->histogram);
        delete v;
    } );
    h.releaseBlockOfColumnValues(countColumnBD);
}

template <typename algorithmFpType, CpuType cpu>
size_t KNNClassificationTrainDistrStep4Kernel<algorithmFpType, training::defaultDense, cpu>::
    computeBucketID(const algorithmFpType * globalSamples, size_t globalSampleCount, algorithmFpType value)
{
    size_t start = 0;
    size_t end = globalSampleCount - 1;
    size_t mid = 0;
    while (end >= start)
    {
        mid = (start + end) / 2;

        if (globalSamples[mid] < value)
        {
            start = mid + 1;
        }
        else if (globalSamples[mid] > value && mid != 0)
        {
            end = mid - 1;
        }
        else
        {
            break;
        }
    }

    while (globalSamples[mid] < value)
    {
        ++mid;
    }

    return mid;
}

template <typename algorithmFpType, CpuType cpu>
services::Status KNNClassificationTrainDistrStep5Kernel<algorithmFpType, training::defaultDense, cpu>::
    compute(NumericTable * x, NumericTable * y, NumericTable * d, bool ispg, size_t nodeCount, NumericTable ** nodeHistograms,
            const size_t * nodeIDs, NumericTable * dataToSend, NumericTable * labelsToSend, NumericTable * medianTable, NumericTable * markers,
            const daal::algorithms::Parameter * par)
{
    typedef BlockDescriptor<algorithmFpType> BD;

    const size_t sampleCount = nodeHistograms[0]->getNumberOfRows();
    size_t * const groupHistogram = static_cast<size_t *>(daal_malloc(sampleCount * sizeof(size_t)));
    const size_t groupPointCount = reduceHistogram(nodeCount, nodeHistograms, sampleCount, groupHistogram);
    BD samplesBD;
    nodeHistograms[0]->getBlockOfColumnValues(0, 0, sampleCount, readOnly, samplesBD);
    const algorithmFpType median = computeApproximatedMedian(sampleCount, samplesBD.getBlockPtr(), groupHistogram, groupPointCount);
    nodeHistograms[0]->releaseBlockOfColumnValues(samplesBD);
    daal_free(groupHistogram);

    const size_t dimension = static_cast<size_t>(d->getValue<int>(0, 0));
    prepareDataToSend(*x, *y, dimension, median, ispg, *dataToSend, *labelsToSend, *markers);

    BD medianBD;
    medianTable->getBlockOfRows(0, 1, writeOnly, medianBD);
    algorithmFpType * const dm = medianBD.getBlockPtr();
    *dm = median;
    medianTable->releaseBlockOfRows(medianBD);
    DAAL_RETURN_STATUS();
}

template <typename algorithmFpType, CpuType cpu>
size_t KNNClassificationTrainDistrStep5Kernel<algorithmFpType, training::defaultDense, cpu>::
    reduceHistogram(size_t nodeCount, const NumericTable * const * nodeHistograms, size_t sampleCount, size_t * groupHistogram)
{
    typedef BlockDescriptor<algorithmFpType> BD;

    size_t pointCount = 0;
    if (nodeCount > 0)
    {
        BD hBD;
        const_cast<NumericTable &>(**nodeHistograms).getBlockOfColumnValues(1, 0, sampleCount, readOnly, hBD);
        const algorithmFpType * const dh = hBD.getBlockPtr();
        for (size_t i = 0; i < sampleCount; ++i)
        {
            groupHistogram[i] = static_cast<size_t>(dh[i]);
            pointCount += static_cast<size_t>(dh[i]);
        }
        const_cast<NumericTable &>(**nodeHistograms).releaseBlockOfColumnValues(hBD);

        for (size_t n = 1; n < nodeCount; ++n)
        {
            const_cast<NumericTable &>(*(nodeHistograms[n])).getBlockOfColumnValues(1, 0, sampleCount, readOnly, hBD);
            const algorithmFpType * const dh = hBD.getBlockPtr();
            for (size_t i = 0; i < sampleCount; ++i)
            {
                groupHistogram[i] += static_cast<size_t>(dh[i]);
                pointCount += static_cast<size_t>(dh[i]);
            }
            const_cast<NumericTable &>(*(nodeHistograms[n])).releaseBlockOfColumnValues(hBD);
        }
    }
    else
    {
        for (size_t i = 0; i < sampleCount; ++i)
        {
            groupHistogram[i] = 0;
        }
    }

    return pointCount;
}

template <typename algorithmFpType, CpuType cpu>
algorithmFpType KNNClassificationTrainDistrStep5Kernel<algorithmFpType, training::defaultDense, cpu>::
    computeApproximatedMedian(size_t sampleCount, const algorithmFpType * samples, const size_t * groupHistogram, size_t pointCount)
{
    const size_t pointCountThreshold = pointCount / 2;
    size_t sum = 0;
    size_t i = 0;
    for (; i < sampleCount; ++i)
    {
        sum += groupHistogram[i];
        if (sum >= pointCountThreshold)
        {
            break;
        }
    }

    if (i + 1 < sampleCount)
    {
        return ((samples[i] + samples[i + 1]) / 2);
    }
    else
    {
        return samples[i];
    }
}

template <typename algorithmFpType, CpuType cpu>
void KNNClassificationTrainDistrStep5Kernel<algorithmFpType, training::defaultDense, cpu>::
    prepareDataToSend(const NumericTable & x, const NumericTable & y, size_t dimension, algorithmFpType median, bool isPartnerGreater,
                      NumericTable & xToSend, NumericTable & yToSend, NumericTable & sendMarkers)
{
    typedef BlockDescriptor<algorithmFpType> BD;

    const size_t xRowCount = x.getNumberOfRows();
    const size_t xColumnCount = x.getNumberOfColumns();
    const size_t yColumnCount = y.getNumberOfColumns();
    const size_t initialBuffersSize = __KDTREE_THREAD_LOCAL_BUFFER / xColumnCount != 0 ? __KDTREE_THREAD_LOCAL_BUFFER / xColumnCount : 1;

    struct Local
    {
        BD xBD;
        BD xRowBD;
        BD yRowBD;
        size_t sendCount;
        size_t sendCapacity;
        size_t markerCount;
        size_t markerCapacity;
        algorithmFpType * xBuffer;
        algorithmFpType * yBuffer;
        algorithmFpType * markersBuffer;
    };

    daal::tls<Local *> localTLS([=]()-> Local *
    {
        Local * const ptr = new Local;
        ptr->sendCount = 0;
        ptr->markerCount = 0;
        ptr->sendCapacity = initialBuffersSize;
        ptr->markerCapacity = initialBuffersSize;
        ptr->xBuffer = static_cast<algorithmFpType *>(daal_malloc(ptr->sendCapacity * xColumnCount * sizeof(algorithmFpType)));
        ptr->yBuffer = static_cast<algorithmFpType *>(daal_malloc(ptr->sendCapacity * yColumnCount * sizeof(algorithmFpType)));
        ptr->markersBuffer = static_cast<algorithmFpType *>(daal_malloc(ptr->markerCapacity * sizeof(algorithmFpType)));
        return ptr;
    } );

    const size_t rowsPerBlock = 512;
    const size_t blockCount = (xRowCount + rowsPerBlock - 1) / rowsPerBlock;
    daal::threader_for(blockCount, blockCount, [=, &localTLS, &x, &y](int iBlock)
    {
        Local * const local = localTLS.local();
        const size_t first = iBlock * rowsPerBlock;
        const size_t last = min<cpu>(first + rowsPerBlock, xRowCount);

        if (first < last)
        {
            const size_t cnt = last - first;
            const_cast<NumericTable &>(x).getBlockOfColumnValues(dimension, first, cnt, readOnly, local->xBD);
            const algorithmFpType * const dx = local->xBD.getBlockPtr();
            for (size_t i = 0; i < cnt; ++i)
            {
                if (isPartnerGreater)
                {
                    if (dx[i] > median)
                    {
                        const_cast<NumericTable &>(x).getBlockOfRows(first + i, 1, readOnly, local->xRowBD);
                        daal_memcpy_s(&(local->xBuffer[local->sendCount * xColumnCount]), xColumnCount * sizeof(algorithmFpType),
                                      local->xRowBD.getBlockPtr(), xColumnCount * sizeof(algorithmFpType));
                        const_cast<NumericTable &>(x).releaseBlockOfRows(local->xRowBD);

                        const_cast<NumericTable &>(y).getBlockOfRows(first + i, 1, readOnly, local->yRowBD);
                        daal_memcpy_s(&(local->yBuffer[local->sendCount * yColumnCount]), yColumnCount * sizeof(algorithmFpType),
                                      local->yRowBD.getBlockPtr(), yColumnCount * sizeof(algorithmFpType));
                        const_cast<NumericTable &>(y).releaseBlockOfRows(local->yRowBD);

                        local->markersBuffer[local->markerCount] = first + i;

                        if (++local->sendCount == local->sendCapacity)
                        {
                            local->sendCapacity *= 2;

                            algorithmFpType * const xBufferTemp = static_cast<algorithmFpType *>(daal_malloc(local->sendCapacity * xColumnCount
                                                                                                             * sizeof(algorithmFpType)));
                            daal_memcpy_s(xBufferTemp, local->sendCount * xColumnCount * sizeof(algorithmFpType),
                                          local->xBuffer, local->sendCount * xColumnCount * sizeof(algorithmFpType));
                            daal_free(local->xBuffer);
                            local->xBuffer = xBufferTemp;

                            algorithmFpType * const yBufferTemp = static_cast<algorithmFpType *>(daal_malloc(local->sendCapacity * yColumnCount
                                                                                                             * sizeof(algorithmFpType)));
                            daal_memcpy_s(yBufferTemp, local->sendCount * yColumnCount * sizeof(algorithmFpType),
                                          local->yBuffer, local->sendCount * yColumnCount * sizeof(algorithmFpType));
                            daal_free(local->yBuffer);
                            local->yBuffer = yBufferTemp;
                        }

                        if (++local->markerCount == local->markerCapacity)
                        {
                            local->markerCapacity *= 2;

                            algorithmFpType * const markersBufferTemp = static_cast<algorithmFpType *>(daal_malloc(local->markerCapacity
                                                                                                                   * sizeof(algorithmFpType)));
                            daal_memcpy_s(markersBufferTemp, local->markerCount * sizeof(algorithmFpType),
                                          local->markersBuffer, local->markerCount * sizeof(algorithmFpType));
                            daal_free(local->markersBuffer);
                            local->markersBuffer = markersBufferTemp;
                        }
                    }
                }
                else
                {
                    if (dx[i] <= median)
                    {
                        const_cast<NumericTable &>(x).getBlockOfRows(first + i, 1, readOnly, local->xRowBD);
                        daal_memcpy_s(&(local->xBuffer[local->sendCount * xColumnCount]), xColumnCount * sizeof(algorithmFpType),
                                      local->xRowBD.getBlockPtr(), xColumnCount * sizeof(algorithmFpType));
                        const_cast<NumericTable &>(x).releaseBlockOfRows(local->xRowBD);

                        const_cast<NumericTable &>(y).getBlockOfRows(first + i, 1, readOnly, local->yRowBD);
                        daal_memcpy_s(&(local->yBuffer[local->sendCount * yColumnCount]), yColumnCount * sizeof(algorithmFpType),
                                      local->yRowBD.getBlockPtr(), yColumnCount * sizeof(algorithmFpType));
                        const_cast<NumericTable &>(y).releaseBlockOfRows(local->yRowBD);

                        local->markersBuffer[local->markerCount] = first + i;

                        if (++local->sendCount == local->sendCapacity)
                        {
                            local->sendCapacity *= 2;

                            algorithmFpType * const xBufferTemp = static_cast<algorithmFpType *>(daal_malloc(local->sendCapacity * xColumnCount
                                                                                                             * sizeof(algorithmFpType)));
                            daal_memcpy_s(xBufferTemp, local->sendCount * xColumnCount * sizeof(algorithmFpType),
                                          local->xBuffer, local->sendCount * xColumnCount * sizeof(algorithmFpType));
                            daal_free(local->xBuffer);
                            local->xBuffer = xBufferTemp;

                            algorithmFpType * const yBufferTemp = static_cast<algorithmFpType *>(daal_malloc(local->sendCapacity * yColumnCount
                                                                                                             * sizeof(algorithmFpType)));
                            daal_memcpy_s(yBufferTemp, local->sendCount * yColumnCount * sizeof(algorithmFpType),
                                          local->yBuffer, local->sendCount * yColumnCount * sizeof(algorithmFpType));
                            daal_free(local->yBuffer);
                            local->yBuffer = yBufferTemp;
                        }

                        if (++local->markerCount == local->markerCapacity)
                        {
                            local->markerCapacity *= 2;

                            algorithmFpType * const markersBufferTemp = static_cast<algorithmFpType *>(daal_malloc(local->markerCapacity
                                                                                                                   * sizeof(algorithmFpType)));
                            daal_memcpy_s(markersBufferTemp, local->markerCount * sizeof(algorithmFpType),
                                          local->markersBuffer, local->markerCount * sizeof(algorithmFpType));
                            daal_free(local->markersBuffer);
                            local->markersBuffer = markersBufferTemp;
                        }
                    }
                }
            }

            const_cast<NumericTable &>(x).releaseBlockOfColumnValues(local->xBD);
        }
    } );

    size_t totalSendCount = 0;
    size_t totalMarkerCount = 0;
    localTLS.reduce([=, &totalSendCount, &totalMarkerCount](Local * v) -> void
    {
        totalSendCount += v->sendCount;
        totalMarkerCount += v->markerCount;
    } );

    allocateTableData(totalSendCount, xToSend);
    allocateTableData(totalSendCount, yToSend);
    allocateTableData(totalMarkerCount, sendMarkers);

    BD xToSendBD, yToSendBD, sendMarkersBD;
    size_t currentSendCount = 0;
    size_t currentMarkerCount = 0;
    localTLS.reduce([=, &xToSendBD, &yToSendBD, &sendMarkersBD, &currentSendCount, &currentMarkerCount, &xToSend, &yToSend, &sendMarkers]
                    (Local * v) -> void
    {
        xToSend.getBlockOfRows(currentSendCount, v->sendCount, writeOnly, xToSendBD);
        yToSend.getBlockOfRows(currentSendCount, v->sendCount, writeOnly, yToSendBD);
        sendMarkers.getBlockOfRows(currentMarkerCount, v->markerCount, writeOnly, sendMarkersBD);

        daal_memcpy_s(xToSendBD.getBlockPtr(), v->sendCount * xColumnCount * sizeof(algorithmFpType), v->xBuffer,
                      v->sendCount * xColumnCount * sizeof(algorithmFpType));
        daal_memcpy_s(yToSendBD.getBlockPtr(), v->sendCount * yColumnCount * sizeof(algorithmFpType), v->yBuffer,
                      v->sendCount * yColumnCount * sizeof(algorithmFpType));
        daal_memcpy_s(sendMarkersBD.getBlockPtr(), v->markerCount * sizeof(algorithmFpType), v->markersBuffer,
                      v->markerCount * sizeof(algorithmFpType));

        sendMarkers.releaseBlockOfRows(sendMarkersBD);
        yToSend.releaseBlockOfRows(yToSendBD);
        xToSend.releaseBlockOfRows(xToSendBD);

        currentSendCount += v->sendCount;
        currentMarkerCount += v->markerCount;

        daal_free(v->xBuffer);
        daal_free(v->yBuffer);
        daal_free(v->markersBuffer);
        delete v;
    } );
}

template <typename algorithmFpType, CpuType cpu>
services::Status KNNClassificationTrainDistrStep6Kernel<algorithmFpType, training::defaultDense, cpu>::
    compute(NumericTable * x, NumericTable * y, NumericTable * px, NumericTable * py, NumericTable * markers, NumericTable * cx,
            NumericTable * cy, const daal::algorithms::Parameter * par)
{
    concatenate(*x, *y, *px, *py, *markers, *cx, *cy);
    DAAL_RETURN_STATUS();
}

template <typename algorithmFpType, CpuType cpu>
void KNNClassificationTrainDistrStep6Kernel<algorithmFpType, training::defaultDense, cpu>::
    concatenate(const NumericTable & x, const NumericTable & y, const NumericTable & px, const NumericTable & py, const NumericTable & markers,
                NumericTable & cx, NumericTable & cy)
{
    typedef BlockDescriptor<algorithmFpType> BD;

    const size_t markersRowCount = markers.getNumberOfRows();
    const size_t partnerRowCount = px.getNumberOfRows();

    if (markersRowCount == 0 && partnerRowCount == 0)
    {
        copyNT(x, cx);
        copyNT(y, cy);
        return;
    }

    if (partnerRowCount >= markersRowCount)
    {
        const size_t xRowCount = x.getNumberOfRows();
        const size_t newPointCount = xRowCount - markersRowCount + partnerRowCount;
        allocateTableData(newPointCount, cx);
        allocateTableData(newPointCount, cy);

        // Copy existed data.
        copyNTRows(0, 0, xRowCount, x, cx);
        copyNTRows(0, 0, xRowCount, y, cy);

        // Use marked slots.
        struct Local
        {
            BD markersBD;
            BD cxBD;
            BD cyBD;
            BD pxBD;
            BD pyBD;
        };
        daal::tls<Local *> localTLS([=]()-> Local *
        {
            Local * const ptr = new Local;
            return ptr;
        } );
        const auto rowsPerBlock = 256;
        const auto blockCount = (markersRowCount + rowsPerBlock - 1) / rowsPerBlock;
        daal::threader_for(blockCount, blockCount, [=, &localTLS, &markers, &cx, &cy, &px, &py](int iBlock)
        {
            Local * const local = localTLS.local();
            const size_t first = iBlock * rowsPerBlock;
            const size_t last = min<cpu>(first + rowsPerBlock, markersRowCount);
            const size_t cnt = last - first;

            const_cast<NumericTable &>(markers).getBlockOfRows(first, cnt, readOnly, local->markersBD);
            const algorithmFpType * const dm = local->markersBD.getBlockPtr();
            const_cast<NumericTable &>(px).getBlockOfRows(first, cnt, readOnly, local->pxBD);
            const algorithmFpType * const dpx = local->pxBD.getBlockPtr();
            const_cast<NumericTable &>(py).getBlockOfRows(first, cnt, readOnly, local->pyBD);
            const algorithmFpType * const dpy = local->pyBD.getBlockPtr();
            for (size_t i = 0; i < cnt; ++i)
            {
                cx.getBlockOfRows(dm[i], 1, writeOnly, local->cxBD);
                daal_memcpy_s(local->cxBD.getBlockPtr(), local->cxBD.getNumberOfColumns() * sizeof(algorithmFpType),
                              &dpx[i * local->pxBD.getNumberOfColumns()], local->pxBD.getNumberOfColumns() * sizeof(algorithmFpType));
                cx.releaseBlockOfRows(local->cxBD);

                cy.getBlockOfRows(dm[i], 1, writeOnly, local->cyBD);
                daal_memcpy_s(local->cyBD.getBlockPtr(), local->cyBD.getNumberOfColumns() * sizeof(algorithmFpType),
                              &dpy[i * local->pyBD.getNumberOfColumns()], local->pyBD.getNumberOfColumns() * sizeof(algorithmFpType));
                cy.releaseBlockOfRows(local->cyBD);
            }
            const_cast<NumericTable &>(py).releaseBlockOfRows(local->pyBD);
            const_cast<NumericTable &>(px).releaseBlockOfRows(local->pxBD);
            const_cast<NumericTable &>(markers).releaseBlockOfRows(local->markersBD);
        } );

        localTLS.reduce([](Local * v) -> void
        {
            delete v;
        } );

        // Copy the rest if needed.
        if (partnerRowCount > markersRowCount)
        {
            const size_t restRowCount = partnerRowCount - markersRowCount;
            copyNTRows(markersRowCount, xRowCount, restRowCount, px, cx);
            copyNTRows(markersRowCount, xRowCount, restRowCount, py, cy);
        }
    }
    else // if (partnerRowCount >= markersRowCount)
    {
        const size_t xRowCount = x.getNumberOfRows();
        const size_t newPointCount = xRowCount - markersRowCount + partnerRowCount;
        allocateTableData(newPointCount, cx);
        allocateTableData(newPointCount, cy);

        // Copy existed data.
        copyNTRows(0, 0, newPointCount, x, cx);
        copyNTRows(0, 0, newPointCount, y, cy);

        // Sort markers.
        size_t * const sortedMarkers = static_cast<size_t *>(daal_malloc(markersRowCount * sizeof(size_t)));
        BD markersBD;
        const_cast<NumericTable &>(markers).getBlockOfRows(0, markersRowCount, readOnly, markersBD);
        const algorithmFpType * const dm = markersBD.getBlockPtr();
        for (size_t i = 0; i < markersRowCount; ++i)
        {
            sortedMarkers[i] = dm[i];
        }
        const_cast<NumericTable &>(markers).releaseBlockOfRows(markersBD);
        daal::algorithms::internal::qSort<size_t, cpu>(markersRowCount, sortedMarkers);

        for (size_t i = 0; i < partnerRowCount; ++i)
        {
            copyNTRows(i, sortedMarkers[i], 1, px, cx);
            copyNTRows(i, sortedMarkers[i], 1, py, cy);
        }
        const size_t restMarkerIndex = lowerBound<cpu>(sortedMarkers, &sortedMarkers[markersRowCount], newPointCount) - sortedMarkers;
        size_t xIndex = newPointCount;
        size_t markerIndex = restMarkerIndex;
        for (size_t i = partnerRowCount; i < restMarkerIndex; ++i)
        {
            // if (sortedMarkers[i] >= newPointCount)
            // {
            //     break;
            // }

            while (markerIndex < markersRowCount && xIndex >= sortedMarkers[markerIndex])
            {
                if (xIndex == sortedMarkers[markerIndex])
                {
                    ++xIndex;
                }
                ++markerIndex;
            }

            copyNTRows(xIndex, sortedMarkers[i], 1, x, cx);
            copyNTRows(xIndex, sortedMarkers[i], 1, y, cy);
            ++xIndex;
        }

        daal_free(sortedMarkers);
    } // if (partnerRowCount >= markersRowCount)
}

template <typename algorithmFpType, CpuType cpu>
void KNNClassificationTrainDistrStep6Kernel<algorithmFpType, training::defaultDense, cpu>::
    copyNT(const NumericTable & src, NumericTable & dest)
{
    const size_t rowCount= src.getNumberOfRows();
    allocateTableData(rowCount, dest);
    copyNTRows(0, 0, rowCount, src, dest);
}

template <typename algorithmFpType, CpuType cpu>
services::Status KNNClassificationTrainDistrStep7Kernel<algorithmFpType, training::defaultDense, cpu>::
    compute(NumericTable * inputb, NumericTable * loops, int loop, size_t nodeCount, NumericTable ** dimensions, NumericTable ** medians,
            size_t * dimensionNodeIDs, size_t * medianNodeIDs, PartialModel * inputpm, PartialModel * outputpm, NumericTable * outputb,
            const daal::algorithms::Parameter * par)
{
    size_t * dims = static_cast<size_t *>(daal_malloc(nodeCount * sizeof(size_t)));
    algorithmFpType * m = static_cast<algorithmFpType *>(daal_malloc(nodeCount * sizeof(algorithmFpType)));
    for (size_t i = 0; i < nodeCount; ++i)
    {
        dims[dimensionNodeIDs[i]] = dimensions[i]->getValue<int>(0, 0);
        m[medianNodeIDs[i]] = medians[i]->getValue<algorithmFpType>(0, 0);
    }

    const int numberOfLoops = loops->getValue<int>(0, 0);
    copyNTRows(0, 0, inputb->getNumberOfRows(), *inputb, *outputb);
    updateBoundingBoxes(loop, numberOfLoops, nodeCount, dims, m, *outputb);

    growPartitioningKDTree(loop, numberOfLoops, nodeCount, dims, m, inputpm, *outputpm);
    daal_free(m);
    daal_free(dims);
    DAAL_RETURN_STATUS();
}

template <typename algorithmFpType, CpuType cpu>
void KNNClassificationTrainDistrStep7Kernel<algorithmFpType, training::defaultDense, cpu>::
    growPartitioningKDTree(int loop, int loops, size_t nodeCount, const size_t * dimensions, const algorithmFpType * medians,
                           const PartialModel * inputpm, PartialModel & outputpm)
{
    const size_t p2 = static_cast<size_t>(1) << loop;
    const size_t jumpingFactor = nodeCount / p2;
    const size_t newNodeCount = nodeCount / jumpingFactor;
    const size_t firstNodePos = p2 - 1;
    SharedPtr<const PartitioningKDTreeTable> oldKDTreeTable = inputpm ? inputpm->impl()->getPartitioningKDTreeTable()
                                                                     : SharedPtr<const PartitioningKDTreeTable>();
    size_t lastPos = inputpm ? inputpm->impl()->getPartitioningLastNodeIndex() : 0;
    const size_t oldNodeCount = oldKDTreeTable ? oldKDTreeTable->getNumberOfRows() : 0;
    SharedPtr<PartitioningKDTreeTable> newKDTreeTable(new PartitioningKDTreeTable(oldNodeCount + newNodeCount));
    if (oldNodeCount > 0)
    {
        const PartitioningKDTreeNode * const oldNode = static_cast<const PartitioningKDTreeNode *>(oldKDTreeTable->getArray());
        PartitioningKDTreeNode * const newNode = static_cast<PartitioningKDTreeNode *>(newKDTreeTable->getArray());
        for (size_t i = 0; i < oldNodeCount; ++i)
        {
            newNode[i] = oldNode[i];
        }
    }
    else
    {
        ++lastPos;
    }

    size_t pos = 0;
    for (size_t l = 0; l < nodeCount; l += jumpingFactor)
    {
        PartitioningKDTreeNode & curNode = *(static_cast<PartitioningKDTreeNode *>(newKDTreeTable->getArray()) + (firstNodePos + pos));
        ++pos;
        curNode.dimension = dimensions[l];
        curNode.cutPoint = medians[l];

        if (loop + 1 == loops)
        {
            curNode.leftIndex = l;
            curNode.rightIndex = l + 1;
            curNode.isLeaf = true;
        }
        else
        {
            curNode.leftIndex = lastPos++;
            curNode.rightIndex = lastPos++;
            curNode.isLeaf = false;
        }
    }

    outputpm.impl()->setPartitioningKDTreeTable(newKDTreeTable);
    outputpm.impl()->setPartitioningLastNodeIndex(lastPos);
}

template <typename algorithmFpType, CpuType cpu>
void KNNClassificationTrainDistrStep7Kernel<algorithmFpType, training::defaultDense, cpu>::
    updateBoundingBoxes(int loop, int loops, size_t nodeCount, const size_t * dimensions, const algorithmFpType * medians, NumericTable & b)
{
    typedef BlockDescriptor<algorithmFpType> BD;

    BD srcBD, destBD;
    const size_t p2 = static_cast<size_t>(1) << loop;
    for (size_t i = 0; i < nodeCount; ++i)
    {
        const size_t color = calculateColor(loop, loops, nodeCount, i);
        const size_t idx = p2 + color;
        const size_t lower = idx << 1;
        const size_t upper = lower + 1;

        b.getBlockOfRows(idx * 2, 2, readOnly, srcBD);

        {
            b.getBlockOfRows(lower * 2, 2, writeOnly, destBD);
            algorithmFpType * const dest = destBD.getBlockPtr();
            daal_memcpy_s(dest, destBD.getNumberOfColumns() * destBD.getNumberOfRows() * sizeof(algorithmFpType),
                          srcBD.getBlockPtr(), srcBD.getNumberOfColumns() * srcBD.getNumberOfRows() * sizeof(algorithmFpType));
            dest[__BBOX_UPPER * destBD.getNumberOfColumns() + dimensions[i]] = medians[i];
            b.releaseBlockOfRows(destBD);
        }

        {
            b.getBlockOfRows(upper * 2, 2, writeOnly, destBD);
            algorithmFpType * const dest = destBD.getBlockPtr();
            daal_memcpy_s(dest, destBD.getNumberOfColumns() * destBD.getNumberOfRows() * sizeof(algorithmFpType),
                          srcBD.getBlockPtr(), srcBD.getNumberOfColumns() * srcBD.getNumberOfRows() * sizeof(algorithmFpType));
            dest[__BBOX_LOWER * destBD.getNumberOfColumns() + dimensions[i]] = medians[i];
            b.releaseBlockOfRows(destBD);
        }

        b.releaseBlockOfRows(srcBD);
    }
}

template <typename algorithmFpType, CpuType cpu>
services::Status KNNClassificationTrainDistrStep8Kernel<algorithmFpType, training::defaultDense, cpu>::
    compute(NumericTable * x, NumericTable * y, PartialModel * inputpm, PartialModel * outputpm, const daal::algorithms::Parameter * par)
{
    const kdtree_knn_classification::Parameter * const parameter = static_cast<const kdtree_knn_classification::Parameter *>(par);

    outputpm->impl()->setPartitioningKDTreeTable(inputpm->impl()->getPartitioningKDTreeTable());
    outputpm->impl()->setPartitioningLastNodeIndex(inputpm->impl()->getPartitioningLastNodeIndex());

    SharedPtr<KDTreeTable> kDTreeTable;
    size_t rootKDTreeNodeIndex, lastContiguousKDTreeNodeIndex;
    buildKDTree(*x, *y, parameter->seed, kDTreeTable, rootKDTreeNodeIndex, lastContiguousKDTreeNodeIndex);

    outputpm->setNFeatures(x->getNumberOfColumns());
    outputpm->impl()->setKDTreeTable(kDTreeTable);
    outputpm->impl()->setRootNodeIndex(rootKDTreeNodeIndex);
    outputpm->impl()->setLastNodeIndex(lastContiguousKDTreeNodeIndex);
    DAAL_RETURN_STATUS();
}

} // namespace internal
} // namespace training
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
