/* file: kdtree_knn_classification_predict_dense_default_distributed_impl.i */
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
//  Common functions for K-Nearest Neighbors predictions calculation
//--
*/

#ifndef __KDTREE_KNN_CLASSIFICATION_PREDICT_DENSE_DEFAULT_DISTRIBUTED_IMPL_I__
#define __KDTREE_KNN_CLASSIFICATION_PREDICT_DENSE_DEFAULT_DISTRIBUTED_IMPL_I__

#include "threading.h"
#include "daal_defines.h"
#include "algorithm.h"
#include "daal_atomic_int.h"
#include "service_memory.h"
#include "service_data_utils.h"
#include "service_math.h"
#include "service_rng.h"
#include "service_sort.h"
#include "service_threading.h"
#include "numeric_table.h"
#include "kdtree_knn_classification_predict_dense_default_distributed.h"
#include "kdtree_knn_classification_model_impl.h"
#include "kdtree_knn_impl.i"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace prediction
{
namespace internal
{

using namespace daal::services::internal;
using namespace daal::services;
using namespace daal::internal;
using namespace kdtree_knn_classification::internal;

template <typename algorithmFPType, CpuType cpu>
struct GlobalNeighbors
{
    algorithmFPType distance;
    size_t index;
    size_t classLabel;

    inline bool operator< (const GlobalNeighbors & rhs) const { return (distance < rhs.distance); }
};

template <typename algorithmFPType>
struct SearchNode
{
    size_t nodeIndex;
    algorithmFPType minDistance;
};

template <typename algorithmFPType, CpuType cpu>
class ColumnsArray
{
public:
    ColumnsArray(NumericTable &numericTable) :
        sourceTable(numericTable),
        dataForEachColumn(numericTable.getNumberOfColumns()),
        descriptorsForEachColumn(numericTable.getNumberOfColumns())
    {
        const size_t rowsNumber    = getNumberOfRows();
        const size_t columnsNumber = getNumberOfColumns();

        for (size_t j = 0; j < columnsNumber; j++)
        {
            BlockDescriptor<algorithmFPType> &columnBlock = descriptorsForEachColumn[j];
            sourceTable.getBlockOfColumnValues(j, 0, rowsNumber, readOnly, columnBlock);
            dataForEachColumn[j] = columnBlock.getBlockPtr();
        }
    }

    ~ColumnsArray()
    {
        const size_t columnsNumber = getNumberOfColumns();
        for (size_t j = 0; j < columnsNumber; j++)
        {
            sourceTable.releaseBlockOfColumnValues(descriptorsForEachColumn[j]);
        }
    }

    inline algorithmFPType *operator[] (size_t columnIndex) const
    {
        return dataForEachColumn[columnIndex];
    }

    inline size_t getNumberOfRows() const
    {
        return sourceTable.getNumberOfRows();
    }

    inline size_t getNumberOfColumns() const
    {
        return sourceTable.getNumberOfColumns();
    }

    inline algorithmFPType **getColumnsData() const
    {
        return const_cast<algorithmFPType **>(dataForEachColumn.get());
    }

    ColumnsArray(const ColumnsArray &) = delete;
    ColumnsArray &operator=(const ColumnsArray &) = delete;

private:

    NumericTable &sourceTable;
    TArray<algorithmFPType *, cpu> dataForEachColumn;
    TArray<BlockDescriptor<algorithmFPType>, cpu> descriptorsForEachColumn;
};

template<typename algorithmFPType, CpuType cpu>
services::Status KNNClassificationPredictDistrStep1Kernel<algorithmFPType, defaultDense, cpu>::
    compute(const NumericTable * x, const PartialModel * m, NumericTable * keys, const daal::algorithms::Parameter * par)
{
    DAAL_ASSERT(x);
    DAAL_ASSERT(m);
    DAAL_ASSERT(keys);
    DAAL_ASSERT(m->impl());

    const size_t xRowCount = x->getNumberOfRows();
    const size_t xColumnCount = x->getNumberOfColumns();
    DAAL_ASSERT(xRowCount == keys->getNumberOfRows());
    PartitioningKDTreeTableConstPtr partitioningKDTreeTable = m->impl()->getPartitioningKDTreeTable();
    const PartitioningKDTreeNode * const treeNodes = static_cast<const PartitioningKDTreeNode *>(partitioningKDTreeTable->getArray());
    DAAL_ASSERT(treeNodes);
    const size_t treeNodeCount = partitioningKDTreeTable->getNumberOfRows();

    SafeStatus safeStat;
    const size_t rowsPerBlock = 512;
    const size_t blockCount = (xRowCount + rowsPerBlock - 1) / rowsPerBlock;
    daal::threader_for(blockCount, blockCount, [=, &safeStat](int iBlock)
    {
        const size_t first = iBlock * rowsPerBlock;
        const size_t last = min<cpu>(first + rowsPerBlock, xRowCount);

        ReadRows<algorithmFPType, cpu> xBD(const_cast<NumericTable *>(x), first, last - first);
        DAAL_CHECK_BLOCK_STATUS_THR(xBD);
        const algorithmFPType * const dx = xBD.get();

        WriteRows<algorithmFPType, cpu> keysBD(keys, first, last - first);
        DAAL_CHECK_BLOCK_STATUS_THR(keysBD);
        auto * const dkeys  = keysBD.get();

        algorithmFPType val, diff;
        const PartitioningKDTreeNode * treeNode;
        for (size_t i = 0; i < last - first; ++i)
        {
            const algorithmFPType * const xRow = &dx[i * xColumnCount];
            size_t nodeIndex = 0;
            DAAL_ASSERT(nodeIndex < treeNodeCount);
            for (;;)
            {
                treeNode = &treeNodes[nodeIndex];
                val = xRow[treeNode->dimension];
                if (treeNode->isLeaf)
                {
                    if (val - treeNode->cutPoint < 0)
                    {
                        dkeys[i] = treeNode->leftIndex;
                        break;
                    }
                    else
                    {
                        dkeys[i] = treeNode->rightIndex;
                        break;
                    }
                }
                else
                {
                    diff = val - treeNode->cutPoint;
                    nodeIndex = (diff < 0) ? treeNode->leftIndex : treeNode->rightIndex;
                    DAAL_ASSERT(nodeIndex < treeNodeCount);
                }
            }
        }
    } );

    return safeStat.detach();
}

template<typename algorithmFPType, CpuType cpu>
services::Status KNNClassificationPredictDistrStep2Kernel<algorithmFPType, defaultDense, cpu>::
    compute(const NumericTable * x, const NumericTable * interm, int key, int round, const PartialModel * m,
            size_t respCount, const NumericTable * const * resp, const size_t * respIDs,
            size_t inputQueryCount, const NumericTable * const * commInputQueries, const size_t * commInputQueriesNodeIDs,
            size_t outputResponsesCount, NumericTable * const * commOutputResponses, const size_t * commOutputResponsesNodeIDs,
            size_t queryCount, NumericTable * const * queries, const size_t * queryIDs,
            NumericTable * y, const daal::algorithms::Parameter * par)
{
    typedef GlobalNeighbors<algorithmFPType, cpu> Neighbors;
    typedef Heap<Neighbors, cpu> MaxHeap;
    typedef kdtree_knn_classification::internal::Stack<SearchNode<algorithmFPType>, cpu> SearchStack;
    typedef daal::data_feature_utils::internal::MaxVal<algorithmFPType, cpu> MaxVal;
    typedef daal::internal::Math<algorithmFPType, cpu> Math;

    DAAL_ASSERT(dynamic_cast<const kdtree_knn_classification::Parameter *>(par));
    const kdtree_knn_classification::Parameter * const parameter = static_cast<const kdtree_knn_classification::Parameter *>(par);

    const auto k = parameter->k;
    const auto & kdTreeTable = *(m->impl()->getKDTreeTable());
    const auto & partitioningKDTreeTable = *(m->impl()->getPartitioningKDTreeTable());
    const auto rootTreeNodeIndex = m->impl()->getRootNodeIndex();
    const NumericTable & data = *(m->impl()->getData());
    const NumericTable & labels = *(m->impl()->getLabels());

    // Copying intermediate result from previous rounds.
    if (interm && interm->getNumberOfRows() > 0)
    {
        const size_t intermRowCount = interm->getNumberOfRows();
        const size_t intermColCount = interm->getNumberOfColumns();

        WriteRows<algorithmFPType, cpu> destBD(y, 0, intermRowCount);
        DAAL_CHECK_BLOCK_STATUS(destBD);

        ReadRows<algorithmFPType, cpu> srcBD(const_cast<NumericTable *>(interm), 0, intermRowCount);
        DAAL_CHECK_BLOCK_STATUS(srcBD);

        daal_memcpy_s(destBD.get(), intermColCount * intermRowCount * sizeof(algorithmFPType),
                      srcBD.get(), intermColCount * intermRowCount * sizeof(algorithmFPType));
    }

    // Generates queries of current round.

    const size_t xColumnCount = x->getNumberOfColumns();
    const size_t xRowCount = x->getNumberOfRows();
    const size_t yColumnCount = y->getNumberOfColumns();

    const size_t queryItemCount = 2 + xColumnCount; // ID, radius, feature values.
    const size_t responseItemCount = 2 + 3 * k; // ID, count, (ID, distance, class) * k;

    size_t iSize = 1;
    while (iSize < k) { iSize *= 2; }
    const size_t heapSize = (iSize / 16 + 1) * 16;

    MaxHeap * const heaps = static_cast<MaxHeap *>(services::daal_malloc(__KDTREE_MAX_QUERIES_PER_ROUND * sizeof(MaxHeap)));
    DAAL_ASSERT((service_memset<unsigned char, cpu>((unsigned char *)heaps, (unsigned char)0, __KDTREE_MAX_QUERIES_PER_ROUND * sizeof(MaxHeap))));
    for (size_t i = 0; i < __KDTREE_MAX_QUERIES_PER_ROUND; ++i)
    {
        heaps[i].init(heapSize);
    }

    const algorithmFPType base = 2.0;
    const size_t expectedMaxDepth = (Math::sLog(xRowCount) / Math::sLog(base) + 1) * __KDTREE_DEPTH_MULTIPLICATION_FACTOR;
    const size_t stackSize = Math::sPowx(base, Math::sCeil(Math::sLog(expectedMaxDepth) / Math::sLog(base)));
    const size_t qCount = max<cpu>(max<cpu>(queryCount, inputQueryCount), max<cpu>(respCount, outputResponsesCount));

    size_t * const bufferCount = static_cast<size_t *>(services::daal_malloc(qCount * sizeof(size_t)));
    DAAL_CHECK_MALLOC(bufferCount);

    size_t * const bufferCapacity = static_cast<size_t *>(services::daal_malloc(qCount * sizeof(size_t)));
    if (!bufferCapacity)
    {
        daal_free(bufferCount);
        return services::Status(services::ErrorMemoryAllocationFailed);
    }

    algorithmFPType ** const buffer = static_cast<algorithmFPType **>(services::daal_malloc(qCount * sizeof(algorithmFPType *)));
    if (!buffer)
    {
        daal_free(bufferCount);
        daal_free(bufferCapacity);
        return services::Status(services::ErrorMemoryAllocationFailed);
    }

    for (size_t i = 0; i < qCount; ++i)
    {
        buffer[i] = nullptr;
        bufferCapacity[i] = 0;
        bufferCount[i] = 0;
    }

    const size_t mutexCount = max<cpu>(queryCount, respCount);
    Mutex * const mutexs = new Mutex[mutexCount];

    struct Local
    {
        SearchStack stack;
        size_t * list;
        algorithmFPType ** buffer;
        size_t * bufferCapacity;
        size_t * bufferCount;
    };
    daal::tls<Local *> localTLS([=]()-> Local *
    {
        Local * const ptr = service_scalable_calloc<Local, cpu>(1);
        if (ptr)
        {
            if (!ptr->stack.init(stackSize))
            {
                _errors->add(services::ErrorMemoryAllocationFailed);
                service_scalable_free<Local, cpu>(ptr);
                return nullptr;
            }
            if ((ptr->list = static_cast<size_t *>(services::daal_malloc(queryCount * sizeof(size_t)))) == nullptr)
            {
                _errors->add(services::ErrorMemoryAllocationFailed);
                ptr->stack.clear();
                service_scalable_free<Local, cpu>(ptr);
                return nullptr;
            }
            if ((ptr->buffer = static_cast<algorithmFPType **>(services::daal_malloc(queryCount * sizeof(algorithmFPType *)))) == nullptr)
            {
                _errors->add(services::ErrorMemoryAllocationFailed);
                daal_free(ptr->list);
                ptr->stack.clear();
                service_scalable_free<Local, cpu>(ptr);
                return nullptr;
            }
            if ((ptr->bufferCapacity = static_cast<size_t *>(services::daal_malloc(queryCount * sizeof(size_t)))) == nullptr)
            {
                _errors->add(services::ErrorMemoryAllocationFailed);
                daal_free(ptr->buffer);
                daal_free(ptr->list);
                ptr->stack.clear();
                service_scalable_free<Local, cpu>(ptr);
                return nullptr;
            }
            if ((ptr->bufferCount = static_cast<size_t *>(services::daal_malloc(queryCount * sizeof(size_t)))) == nullptr)
            {
                _errors->add(services::ErrorMemoryAllocationFailed);
                daal_free(ptr->bufferCapacity);
                daal_free(ptr->buffer);
                daal_free(ptr->list);
                ptr->stack.clear();
                service_scalable_free<Local, cpu>(ptr);
                return nullptr;
            }
            for (size_t i = 0; i < queryCount; ++i)
            {
                ptr->buffer[i] = nullptr;
                ptr->bufferCapacity[i] = 64;
                ptr->bufferCount[i] = 0;
            }
            for (size_t i = 0; i < queryCount; ++i)
            {
                if ((ptr->buffer[i] = static_cast<algorithmFPType *>(services::daal_malloc(ptr->bufferCapacity[i] * queryItemCount
                    * sizeof(algorithmFPType)))) == nullptr)
                {
                    _errors->add(services::ErrorMemoryAllocationFailed);
                    for (size_t j = 0; j < i; ++j)
                    {
                        daal_free(ptr->buffer[j]);
                    }
                    daal_free(ptr->bufferCount);
                    daal_free(ptr->bufferCapacity);
                    daal_free(ptr->buffer);
                    daal_free(ptr->list);
                    ptr->stack.clear();
                    service_scalable_free<Local, cpu>(ptr);
                    return nullptr;
                }
            }
        }
        else { _errors->add(services::ErrorMemoryAllocationFailed); }
        return ptr;
    } );

    /* Unpack data to "column-wise" representation */
    ColumnsArray<algorithmFPType, cpu> dataColumnsArray(const_cast<NumericTable &>(data));

    const auto maxThreads = threader_get_threads_number();
    const size_t firstRow = __KDTREE_MAX_QUERIES_PER_ROUND * round;
    const size_t lastRow = min<cpu>(firstRow + __KDTREE_MAX_QUERIES_PER_ROUND, xRowCount);
    if (firstRow < lastRow)
    {
        const size_t roundRowCount = lastRow - firstRow;

        // Prepare response for itself.
        size_t selfIdx = static_cast<size_t>(-1);
        for (size_t i = 0; i < outputResponsesCount; ++i)
        {
            if (commOutputResponsesNodeIDs[i] == key)
            {
                selfIdx = i;
                break;
            }
        }
        DAAL_ASSERT(selfIdx != static_cast<size_t>(-1));
        DAAL_ASSERT(selfIdx < outputResponsesCount);
        NumericTable & selfResp = *(commOutputResponses[selfIdx]);
        selfResp.resize(roundRowCount);

        const size_t rowsPerBlock = (roundRowCount + maxThreads - 1) / maxThreads;
        const size_t blockCount = (roundRowCount + rowsPerBlock - 1) / rowsPerBlock;
        daal::threader_for(blockCount, blockCount, [=, &localTLS, &kdTreeTable, &data, &labels, &rowsPerBlock, &k, &partitioningKDTreeTable,
                                                    &buffer, &bufferCapacity, &bufferCount, &xColumnCount, &selfResp, &dataColumnsArray](int iBlock)
        {
            Local * const local = localTLS.local();
            if (local)
            {
                const size_t first = firstRow + iBlock * rowsPerBlock;
                const size_t last = min<cpu>(first + rowsPerBlock, lastRow);

                const algorithmFPType radius = MaxVal::get();
                data_management::BlockDescriptor<algorithmFPType> xBD, selfRespBD, yBD;
                DAAL_ASSERT(x->getNumberOfRows() >= last);
                const_cast<NumericTable &>(*x).getBlockOfRows(first, last - first, readOnly, xBD);
                const algorithmFPType * const dx = xBD.getBlockPtr();

                selfResp.getBlockOfRows(first - firstRow, last - first, writeOnly, selfRespBD);
                DAAL_ASSERT(selfRespBD.getNumberOfColumns() == responseItemCount);
                algorithmFPType * const dself = selfRespBD.getBlockPtr();

                for (size_t i = 0; i < last - first; ++i)
                {
                    DAAL_ASSERT(first - firstRow + i < __KDTREE_MAX_QUERIES_PER_ROUND);
                    findNearestNeighbors(&dx[i * xColumnCount], heaps[first - firstRow + i], local->stack,
                                         k, radius, kdTreeTable, rootTreeNodeIndex, dataColumnsArray);

                    const algorithmFPType maxRadius = (heaps[first - firstRow + i].size() > 0 ? heaps[first - firstRow + i].getMax()->distance
                                                                                              : radius);

                    // Generates response for itself.
                    {
                        algorithmFPType * const p = &(dself[i * responseItemCount]);
                        const MaxHeap & h = heaps[first - firstRow + i];
                        const size_t foundCount = h.size();
                        DAAL_ASSERT(foundCount <= k);
                        size_t idx = 0;
                        p[idx++] = first + i;
                        p[idx++] = foundCount;

                        size_t j = 0;
                        for (; j < foundCount; ++j)
                        {
                            const size_t rowIndex = h[j].index;
                            p[idx++] = static_cast<algorithmFPType>(rowIndex);
                            p[idx++] = h[j].distance;

                            const_cast<NumericTable &>(labels).getBlockOfRows(rowIndex, 1, readOnly, yBD);
                            const auto * const dy = yBD.getBlockPtr();
                            p[idx++] = dy[0];
                            const_cast<NumericTable &>(labels).releaseBlockOfRows(yBD);
                        }
                        for (; j < k; ++j)
                        {
                            p[idx++] = 0;
                            p[idx++] = 0;
                            p[idx++] = 0;
                        }
                        DAAL_ASSERT(idx == responseItemCount);
                    }

                    const size_t listCount = findList(&dx[i * xColumnCount], xColumnCount, maxRadius, local->list, queryCount, key, local->stack,
                                                      partitioningKDTreeTable);

                    for (size_t l = 0; l < listCount; ++l)
                    {
                        const size_t destKey = local->list[l];
                        size_t destIdx = static_cast<size_t>(-1);
                        for (size_t j = 0; j < queryCount; ++j)
                        {
                            if (queryIDs[j] == destKey)
                            {
                                destIdx = j;
                                break;
                            }
                        }
                        DAAL_ASSERT(destIdx != static_cast<size_t>(-1));
                        DAAL_ASSERT(destIdx < queryCount);
                        DAAL_ASSERT(local->bufferCount[destIdx] < local->bufferCapacity[destIdx]);
                        algorithmFPType * const p = &(local->buffer[destIdx][(local->bufferCount[destIdx]++) * queryItemCount]);
                        p[0] = static_cast<algorithmFPType>(first + i);
                        p[1] = maxRadius;
                        for (size_t f = 0; f < xColumnCount; ++f)
                        {
                            p[2 + f] = dx[i * xColumnCount + f];
                        }

                        if (local->bufferCount[destIdx] >= local->bufferCapacity[destIdx])
                        {
                            {
                                DAAL_ASSERT(destIdx < mutexCount);
                                AUTOLOCK(mutexs[destIdx]);

                                if (!buffer[destIdx])
                                {
                                    bufferCapacity[destIdx] = max<cpu>(static_cast<size_t>(1024), local->bufferCount[destIdx]);
                                    buffer[destIdx] = static_cast<algorithmFPType *>(services::daal_malloc(
                                         bufferCapacity[destIdx] * queryItemCount * sizeof(algorithmFPType)));
                                }
                                else if (bufferCapacity[destIdx] < bufferCount[destIdx] + local->bufferCount[destIdx])
                                {
                                    bufferCapacity[destIdx] = max<cpu>(bufferCount[destIdx] + local->bufferCount[destIdx],
                                                                            2 * bufferCapacity[destIdx]);
                                    algorithmFPType * const newBuffer = static_cast<algorithmFPType *>(services::daal_malloc(
                                        bufferCapacity[destIdx] * queryItemCount * sizeof(algorithmFPType)));
                                    if (!newBuffer)
                                    {
                                        _errors->add(services::ErrorMemoryAllocationFailed);
                                        break;
                                    }
                                    services::daal_memcpy_s(newBuffer, bufferCapacity[destIdx] * queryItemCount * sizeof(algorithmFPType),
                                                            buffer[destIdx], bufferCount[destIdx] * queryItemCount * sizeof(algorithmFPType));
                                    algorithmFPType * const oldBuffer = buffer[destIdx];
                                    buffer[destIdx] = newBuffer;
                                    daal_free(oldBuffer);
                                }

                                services::daal_memcpy_s(&(buffer[destIdx][bufferCount[destIdx] * queryItemCount]),
                                                        (bufferCapacity[destIdx] - bufferCount[destIdx]) * queryItemCount
                                                            * sizeof(algorithmFPType),
                                                        local->buffer[destIdx], local->bufferCount[destIdx] * queryItemCount
                                                            * sizeof(algorithmFPType));
                                bufferCount[destIdx] += local->bufferCount[destIdx];
                            }
                            local->bufferCount[destIdx] = 0;
                        }
                    }
                }
                selfResp.releaseBlockOfRows(selfRespBD);
                const_cast<NumericTable &>(*x).releaseBlockOfRows(xBD);
            }
            } );

        localTLS.reduce([=](Local * ptr) -> void
        {
            if (ptr)
            {
                // Copy the rest.
                daal::threader_for(queryCount, queryCount, [=, &buffer, &bufferCapacity, &bufferCount, &ptr](int destIdx)
                {
                    if (ptr->bufferCount[destIdx] > 0)
                    {
                        if (!buffer[destIdx])
                        {
                            bufferCapacity[destIdx] = max<cpu>(static_cast<size_t>(1024), ptr->bufferCount[destIdx]);
                            buffer[destIdx] = static_cast<algorithmFPType *>(services::daal_malloc(
                                bufferCapacity[destIdx] * queryItemCount * sizeof(algorithmFPType)));
                        }
                        else if (bufferCapacity[destIdx] < bufferCount[destIdx] + ptr->bufferCount[destIdx])
                        {
                            bufferCapacity[destIdx] = max<cpu>(bufferCount[destIdx] + ptr->bufferCount[destIdx],
                                                                    2 * bufferCapacity[destIdx]);
                            algorithmFPType * const newbuffer = static_cast<algorithmFPType *>(services::daal_malloc(
                                bufferCapacity[destIdx] * queryItemCount * sizeof(algorithmFPType)));
                            if (!newbuffer)
                            {
                                _errors->add(services::ErrorMemoryAllocationFailed);
                                return;
                            }
                            services::daal_memcpy_s(newbuffer, bufferCapacity[destIdx] * queryItemCount * sizeof(algorithmFPType),
                                                    buffer[destIdx], bufferCount[destIdx] * queryItemCount * sizeof(algorithmFPType));
                            algorithmFPType * const oldbuffer = buffer[destIdx];
                            buffer[destIdx] = newbuffer;
                            daal_free(oldbuffer);
                        }

                        services::daal_memcpy_s(&(buffer[destIdx][bufferCount[destIdx] * queryItemCount]),
                                                (bufferCapacity[destIdx] - bufferCount[destIdx]) * queryItemCount
                                                    * sizeof(algorithmFPType),
                                                ptr->buffer[destIdx], ptr->bufferCount[destIdx] * queryItemCount * sizeof(algorithmFPType));
                        bufferCount[destIdx] += ptr->bufferCount[destIdx];
                    }
                } );

                for (size_t j = 0; j < queryCount; ++j)
                {
                    daal_free(ptr->buffer[j]);
                }
                ptr->stack.clear();
                daal_free(ptr->list);
                daal_free(ptr->buffer);
                daal_free(ptr->bufferCapacity);
                daal_free(ptr->bufferCount);
                service_scalable_free<Local, cpu>(ptr);
            }
        } );

        BlockDescriptor<algorithmFPType> qBD;
        for (size_t i = 0; i < queryCount; ++i)
        {
            DAAL_ASSERT(queries[i]->getNumberOfColumns() == queryItemCount);
            queries[i]->resize(bufferCount[i]);
            queries[i]->getBlockOfRows(0, bufferCount[i], writeOnly, qBD);
            services::daal_memcpy_s(qBD.getBlockPtr(), qBD.getNumberOfRows() * qBD.getNumberOfColumns() * sizeof(algorithmFPType),
                                    buffer[i], bufferCount[i] * queryItemCount * sizeof(algorithmFPType));
            queries[i]->releaseBlockOfRows(qBD);
        }
        for (size_t i = 0; i < qCount; ++i)
        {
            bufferCount[i] = 0;
            bufferCapacity[i] = 0;
            if (buffer[i])
            {
                daal_free(buffer[i]);
                buffer[i] = nullptr;
            }
        }
    }

    // Processes input queries, generates appropriate output responses.

    struct QLocal
    {
        MaxHeap heap;
        SearchStack stack;
        size_t * list;
        algorithmFPType ** buffer;
        size_t * bufferCapacity;
        size_t * bufferCount;
    };

    for (size_t inputQueryIdx = 0; inputQueryIdx < inputQueryCount; ++inputQueryIdx)
    {
        if ((!commInputQueries[inputQueryIdx]) || commInputQueries[inputQueryIdx]->getNumberOfRows() == 0)
        {
            continue;
        }
        size_t destIdx = static_cast<size_t>(-1);
        for (size_t j = 0; j < outputResponsesCount; ++j)
        {
            if (commOutputResponsesNodeIDs[j] == commInputQueriesNodeIDs[inputQueryIdx])
            {
                destIdx = j;
                break;
            }
        }
        DAAL_ASSERT(destIdx != static_cast<size_t>(-1));
        DAAL_ASSERT(destIdx < outputResponsesCount);

        daal::tls<QLocal *> localTLS([=]()-> QLocal *
        {
            QLocal * const ptr = service_scalable_calloc<QLocal, cpu>(1);
            if (ptr)
            {
                if (!ptr->heap.init(heapSize))
                {
                    _errors->add(services::ErrorMemoryAllocationFailed);
                    service_scalable_free<QLocal, cpu>(ptr);
                    return nullptr;
                }
                if (!ptr->stack.init(stackSize))
                {
                    _errors->add(services::ErrorMemoryAllocationFailed);
                    ptr->heap.clear();
                    service_scalable_free<QLocal, cpu>(ptr);
                    return nullptr;
                }
                if ((ptr->list = static_cast<size_t *>(services::daal_malloc(qCount * sizeof(size_t)))) == nullptr)
                {
                    _errors->add(services::ErrorMemoryAllocationFailed);
                    ptr->heap.clear();
                    ptr->stack.clear();
                    service_scalable_free<QLocal, cpu>(ptr);
                    return nullptr;
                }
                if ((ptr->buffer = static_cast<algorithmFPType **>(services::daal_malloc(qCount * sizeof(algorithmFPType *)))) == nullptr)
                {
                    _errors->add(services::ErrorMemoryAllocationFailed);
                    daal_free(ptr->list);
                    ptr->heap.clear();
                    ptr->stack.clear();
                    service_scalable_free<QLocal, cpu>(ptr);
                    return nullptr;
                }
                if ((ptr->bufferCapacity = static_cast<size_t *>(services::daal_malloc(qCount * sizeof(size_t)))) == nullptr)
                {
                    _errors->add(services::ErrorMemoryAllocationFailed);
                    daal_free(ptr->buffer);
                    daal_free(ptr->list);
                    ptr->heap.clear();
                    ptr->stack.clear();
                    service_scalable_free<QLocal, cpu>(ptr);
                    return nullptr;
                }
                if ((ptr->bufferCount = static_cast<size_t *>(services::daal_malloc(qCount * sizeof(size_t)))) == nullptr)
                {
                    _errors->add(services::ErrorMemoryAllocationFailed);
                    daal_free(ptr->bufferCapacity);
                    daal_free(ptr->buffer);
                    daal_free(ptr->list);
                    ptr->heap.clear();
                    ptr->stack.clear();
                    service_scalable_free<QLocal, cpu>(ptr);
                    return nullptr;
                }
                for (size_t i = 0; i < qCount; ++i)
                {
                    ptr->buffer[i] = nullptr;
                    ptr->bufferCapacity[i] = 64;
                    ptr->bufferCount[i] = 0;
                }
                for (size_t i = 0; i < qCount; ++i)
                {
                    if ((ptr->buffer[i] = static_cast<algorithmFPType *>(services::daal_malloc(ptr->bufferCapacity[i] * responseItemCount
                        * sizeof(algorithmFPType)))) == nullptr)
                    {
                        _errors->add(services::ErrorMemoryAllocationFailed);
                        for (size_t j = 0; j < i; ++j)
                        {
                            daal_free(ptr->buffer[j]);
                        }
                        daal_free(ptr->bufferCount);
                        daal_free(ptr->bufferCapacity);
                        daal_free(ptr->buffer);
                        daal_free(ptr->list);
                        ptr->heap.clear();
                        ptr->stack.clear();
                        service_scalable_free<QLocal, cpu>(ptr);
                        return nullptr;
                    }
                }
            }
            else { _errors->add(services::ErrorMemoryAllocationFailed); }
            return ptr;
        } );

        const NumericTable & commInputQuery = *(commInputQueries[inputQueryIdx]);
        const size_t inputQueryRowCount = commInputQuery.getNumberOfRows();
        const size_t inputQueryColumnCount = commInputQuery.getNumberOfColumns();
        const size_t rowsPerBlock = (inputQueryRowCount + maxThreads - 1) / maxThreads;
        const size_t blockCount = (inputQueryRowCount + rowsPerBlock - 1) / rowsPerBlock;
        daal::threader_for(blockCount, blockCount, [=, &localTLS, &kdTreeTable, &commInputQuery, &rowsPerBlock, &k, &buffer, &bufferCapacity,
                                                    &bufferCount, &inputQueryColumnCount, &data, &labels, &dataColumnsArray](int iBlock)
        {
            QLocal * const local = localTLS.local();
            if (local)
            {
                const size_t first = iBlock * rowsPerBlock;
                const size_t last = min<cpu>(first + rowsPerBlock, inputQueryRowCount);

                data_management::BlockDescriptor<algorithmFPType> xBD;
                const_cast<NumericTable &>(*commInputQueries[inputQueryIdx]).getBlockOfRows(first, last - first, readOnly, xBD);
                const algorithmFPType * const dx = xBD.getBlockPtr();
                data_management::BlockDescriptor<algorithmFPType> yBD;

                for (size_t i = 0; i < last - first; ++i)
                {
                    const algorithmFPType * const row = &dx[i * xBD.getNumberOfColumns()];
                    const algorithmFPType qID = row[0];
                    const algorithmFPType radius = row[1];

                    findNearestNeighbors(&row[2], local->heap, local->stack, k, radius, kdTreeTable,
                                         rootTreeNodeIndex, dataColumnsArray);

                    if(local->heap.size() > 0)
                    {
                        DAAL_ASSERT(local->bufferCount[destIdx] < local->bufferCapacity[destIdx]);
                        algorithmFPType * const p = &(local->buffer[destIdx][(local->bufferCount[destIdx]++) * responseItemCount]);
                        size_t idx = 0;
                        p[idx++] = qID;
                        p[idx++] = local->heap.size();

                        size_t j = 0;
                        for (; j < local->heap.size(); ++j)
                        {
                            const size_t rowIndex = local->heap[j].index;
                            p[idx++] = static_cast<algorithmFPType>(rowIndex);
                            p[idx++] = local->heap[j].distance;

                            const_cast<NumericTable &>(labels).getBlockOfRows(rowIndex, 1, readOnly, yBD);
                            const auto * const dy = yBD.getBlockPtr();
                            p[idx++] = dy[0];
                            const_cast<NumericTable &>(labels).releaseBlockOfRows(yBD);
                        }
                        for (; j < k; ++j)
                        {
                            p[idx++] = 0;
                            p[idx++] = 0;
                            p[idx++] = 0;
                        }
                        DAAL_ASSERT(idx == responseItemCount);

                        if (local->bufferCount[destIdx] >= local->bufferCapacity[destIdx])
                        {
                            {
                                DAAL_ASSERT(destIdx < mutexCount);
                                AUTOLOCK(mutexs[destIdx]);

                                if (!buffer[destIdx])
                                {
                                    bufferCapacity[destIdx] = max<cpu>(static_cast<size_t>(1024), local->bufferCount[destIdx]);
                                    buffer[destIdx] = static_cast<algorithmFPType *>(services::daal_malloc(bufferCapacity[destIdx]
                                        * responseItemCount * sizeof(algorithmFPType)));
                                }
                                else if (bufferCapacity[destIdx] < bufferCount[destIdx] + local->bufferCount[destIdx])
                                {
                                    bufferCapacity[destIdx] = max<cpu>(bufferCount[destIdx] + local->bufferCount[destIdx],
                                                                            2 * bufferCapacity[destIdx]);
                                    algorithmFPType * const newbuffer = static_cast<algorithmFPType *>(services::daal_malloc(
                                        bufferCapacity[destIdx] * responseItemCount * sizeof(algorithmFPType)));
                                    if (!newbuffer)
                                    {
                                        _errors->add(services::ErrorMemoryAllocationFailed);
                                        break;
                                    }
                                    services::daal_memcpy_s(newbuffer, bufferCapacity[destIdx] * responseItemCount * sizeof(algorithmFPType),
                                                            buffer[destIdx], bufferCount[destIdx] * responseItemCount * sizeof(algorithmFPType));
                                    algorithmFPType * const oldbuffer = buffer[destIdx];
                                    buffer[destIdx] = newbuffer;
                                    daal_free(oldbuffer);
                                }

                                services::daal_memcpy_s(&(buffer[destIdx][bufferCount[destIdx] * responseItemCount]),
                                                        (bufferCapacity[destIdx] - bufferCount[destIdx]) * responseItemCount
                                                            * sizeof(algorithmFPType),
                                                        local->buffer[destIdx], local->bufferCount[destIdx] * responseItemCount
                                                            * sizeof(algorithmFPType));
                                bufferCount[destIdx] += local->bufferCount[destIdx];
                            }
                            local->bufferCount[destIdx] = 0;
                        }
                    }
                }
                const_cast<NumericTable &>(*x).releaseBlockOfRows(xBD);
            }
        } );

        localTLS.reduce([=](QLocal * ptr) -> void
        {
            if (ptr)
            {
                // Copy the rest.
                daal::threader_for(outputResponsesCount, outputResponsesCount, [=, &buffer, &bufferCapacity, &bufferCount, &ptr](int destIdx)
                {
                    if (ptr->bufferCount[destIdx] > 0)
                    {
                        if (!buffer[destIdx])
                        {
                            bufferCapacity[destIdx] = max<cpu>(static_cast<size_t>(1024), ptr->bufferCount[destIdx]);
                            buffer[destIdx] = static_cast<algorithmFPType *>(services::daal_malloc(
                                bufferCapacity[destIdx] * responseItemCount * sizeof(algorithmFPType)));
                        }
                        else if (bufferCapacity[destIdx] < bufferCount[destIdx] + ptr->bufferCount[destIdx])
                        {
                            bufferCapacity[destIdx] = max<cpu>(bufferCount[destIdx] + ptr->bufferCount[destIdx],
                                                                    2 * bufferCapacity[destIdx]);
                            algorithmFPType * const newbuffer = static_cast<algorithmFPType *>(services::daal_malloc(
                                bufferCapacity[destIdx] * responseItemCount * sizeof(algorithmFPType)));
                            if (!newbuffer)
                            {
                                _errors->add(services::ErrorMemoryAllocationFailed);
                                return;
                            }
                            services::daal_memcpy_s(newbuffer, bufferCapacity[destIdx] * responseItemCount * sizeof(algorithmFPType),
                                                    buffer[destIdx], bufferCount[destIdx] * responseItemCount * sizeof(algorithmFPType));
                            algorithmFPType * const oldbuffer = buffer[destIdx];
                            buffer[destIdx] = newbuffer;
                            daal_free(oldbuffer);
                        }

                        services::daal_memcpy_s(&(buffer[destIdx][bufferCount[destIdx] * responseItemCount]),
                                                (bufferCapacity[destIdx] - bufferCount[destIdx]) * responseItemCount
                                                    * sizeof(algorithmFPType),
                                                ptr->buffer[destIdx], ptr->bufferCount[destIdx] * responseItemCount * sizeof(algorithmFPType));
                        bufferCount[destIdx] += ptr->bufferCount[destIdx];
                    }
                } );

                for (size_t j = 0; j < inputQueryCount; ++j)
                {
                    if (ptr->buffer[j])
                    {
                        daal_free(ptr->buffer[j]);
                    }
                }
                ptr->stack.clear();
                ptr->heap.clear();
                daal_free(ptr->list);
                daal_free(ptr->buffer);
                daal_free(ptr->bufferCapacity);
                daal_free(ptr->bufferCount);
                service_scalable_free<QLocal, cpu>(ptr);
            }
        } );

        BlockDescriptor<algorithmFPType> respBD;
        DAAL_ASSERT(commOutputResponses[destIdx]->getNumberOfColumns() == responseItemCount);
        commOutputResponses[destIdx]->resize(bufferCount[destIdx]);
        commOutputResponses[destIdx]->getBlockOfRows(0, bufferCount[destIdx], writeOnly, respBD);
        services::daal_memcpy_s(respBD.getBlockPtr(), respBD.getNumberOfRows() * respBD.getNumberOfColumns() * sizeof(algorithmFPType),
                                buffer[destIdx], bufferCount[destIdx] * responseItemCount * sizeof(algorithmFPType));
        commOutputResponses[destIdx]->releaseBlockOfRows(respBD);
    }

    // Processes responses.

    for (size_t i = 0; i < __KDTREE_MAX_QUERIES_PER_ROUND; ++i)
    {
        heaps[i].reset();
    }

    size_t respHandledCount = 0;
    for (size_t respIdx = 0; respIdx < respCount; ++respIdx)
    {
        if ((!resp[respIdx]) || resp[respIdx]->getNumberOfRows() == 0)
        {
            continue;
        }

        // In the round #0, the queries are generated. In the round #1, responses for queries are generated. Responses to itself are generated at
        // round #0, but handled in round #2.

        DAAL_ASSERT(round >= 2);

        ++respHandledCount;

        const NumericTable & respTable = *(resp[respIdx]);

        const size_t respRowCount = respTable.getNumberOfRows();

        const size_t rowsPerBlock = 512;
        const size_t blockCount = (respRowCount + rowsPerBlock - 1) / rowsPerBlock;
        daal::threader_for(blockCount, blockCount, [=, &respTable, &heaps, &rowsPerBlock, &k, &y, &responseItemCount, &round](int iBlock)
        {
            const size_t first = iBlock * rowsPerBlock;
            const size_t last = min<cpu>(first + rowsPerBlock, respRowCount);
            BlockDescriptor<algorithmFPType> respBD;
            const_cast<NumericTable &>(respTable).getBlockOfRows(first, last - first, readOnly, respBD);
            Neighbors neighbor;
            const algorithmFPType * const dresp = respBD.getBlockPtr();
            for (size_t i = 0; i < last - first; ++i)
            {
                const algorithmFPType * const respRow = &dresp[i * respBD.getNumberOfColumns()];
                size_t idx = 0;
                const size_t qID = static_cast<size_t>(respRow[idx++]);
                DAAL_ASSERT(qID < y->getNumberOfRows());
                DAAL_ASSERT(qID >= (round - 2) * __KDTREE_MAX_QUERIES_PER_ROUND);
                DAAL_ASSERT(qID < (round - 1) * __KDTREE_MAX_QUERIES_PER_ROUND);
                const size_t neighbourCount = static_cast<size_t>(respRow[idx++]);
                DAAL_ASSERT(neighbourCount <= k);
                MaxHeap & heap = heaps[qID % __KDTREE_MAX_QUERIES_PER_ROUND];
                for (size_t l = 0; l < neighbourCount; ++l)
                {
                    neighbor.index = static_cast<size_t>(respRow[idx++]);
                    neighbor.distance = respRow[idx++];
                    neighbor.classLabel = static_cast<size_t>(respRow[idx++]);

                    if (heap.size() == 0)
                    {
                        heap.push(neighbor, k);
                    }
                    else if (heap.getMax()->distance > neighbor.distance)
                    {
                        heap.replaceMax(neighbor);
                    }
                }
                DAAL_ASSERT(idx + (k - neighbourCount) * 3 == responseItemCount);
            }
            const_cast<NumericTable &>(respTable).releaseBlockOfRows(respBD);
        } );
    }

    // Prepare results.

    if (respHandledCount > 0)
    {
        const size_t firstRow = (round - 2) * __KDTREE_MAX_QUERIES_PER_ROUND;
        const size_t lastRow = min<cpu>(firstRow + __KDTREE_MAX_QUERIES_PER_ROUND, y->getNumberOfRows());
        const size_t rowCount = lastRow - firstRow;
        const size_t rowsPerBlock = 128;
        const size_t blockCount = (rowCount + rowsPerBlock - 1) / rowsPerBlock;
        daal::threader_for(blockCount, blockCount, [=, &heaps, &rowsPerBlock, &k, &y, &round](int iBlock)
        {
            const size_t first = iBlock * rowsPerBlock;
            const size_t last = min<cpu>(first + rowsPerBlock, rowCount);
            BlockDescriptor<algorithmFPType> yBD;
            y->getBlockOfRows(firstRow + first, last - first, writeOnly, yBD);
            const size_t yColumnCount = y->getNumberOfColumns();
            auto * const dy = yBD.getBlockPtr();
            for (size_t i = 0; i < last - first; ++i)
            {
                const MaxHeap & heap = heaps[first + i];
                predict(dy[i * yColumnCount], heap, k);
            }
            y->releaseBlockOfRows(yBD);
        } );
    }


    for (size_t i = 0; i < qCount; ++i)
    {
        if (buffer[i])
        {
            daal_free(buffer[i]);
        }
    }
    daal_free(buffer);
    daal_free(bufferCount);
    daal_free(bufferCapacity);

    for (size_t i = 0; i < __KDTREE_MAX_QUERIES_PER_ROUND; ++i)
    {
        heaps[i].clear();
    }
    daal_free(heaps);

    delete[] mutexs;

    DAAL_RETURN_STATUS();
}

template<typename algorithmFPType, CpuType cpu>
void KNNClassificationPredictDistrStep2Kernel<algorithmFPType, defaultDense, cpu>::
    findNearestNeighbors(const algorithmFPType * query, Heap<GlobalNeighbors<algorithmFPType, cpu>, cpu> & heap,
                         Stack<SearchNode<algorithmFPType>, cpu> & stack, size_t k, algorithmFPType radius,
                         const KDTreeTable & kdTreeTable, size_t rootTreeNodeIndex,
                         const ColumnsArray<algorithmFPType, cpu> &data)
{
    DAAL_ASSERT(query);

    heap.reset();
    stack.reset();

    if (kdTreeTable.getNumberOfRows() == 0)
        return;

    size_t i, j;
    size_t start, end;
    algorithmFPType dist, diff, val;

    GlobalNeighbors<algorithmFPType, cpu> curNeighbor;
    SearchNode<algorithmFPType> cur, toPush, interCur;

    cur.minDistance = 0;
    cur.nodeIndex   = rootTreeNodeIndex;

    const size_t xColumnCount     = data.getNumberOfColumns();
    algorithmFPType **dataColumns = data.getColumnsData();

    const KDTreeNode *node;
    const KDTreeNode *treeNodes = static_cast<const KDTreeNode *>(kdTreeTable.getArray());

    DAAL_ALIGNAS(256) algorithmFPType distance[__KDTREE_LEAF_BUCKET_SIZE + 1];

    for (;;)
    {
        DAAL_ASSERT(cur.nodeIndex < kdTreeTable.getNumberOfRows());
        node = treeNodes + cur.nodeIndex;

        if (node->dimension == __KDTREE_NULLDIMENSION)
        {
            start = node->leftIndex;
            end = node->rightIndex;

            for (i = start; i < end; ++i)
            {
                distance[i - start] = 0;
            }

            for (j = 0; j < xColumnCount; ++j)
            {
                const algorithmFPType *dx = dataColumns[j] + start;

                if (j != xColumnCount - 1) {
                    const algorithmFPType *nx = dataColumns[j + 1] + start;
                    DAAL_PREFETCH_READ_T0(nx);
                    DAAL_PREFETCH_READ_T0(nx + 16);
                }

                for (i = 0; i < end - start; ++i)
                {
                    distance[i] += (query[j] - dx[i]) * (query[j] - dx[i]);
                }
            }

            for (i = start; i < end; ++i)
            {
                if (distance[i - start] <= radius)
                {
                    curNeighbor.distance = distance[i - start];
                    curNeighbor.index = i;
                    if (heap.size() < k)
                    {
                        heap.push(curNeighbor, k);

                        if (heap.size() == k)
                        {
                            radius = heap.getMax()->distance;
                        }
                    }
                    else
                    {
                        if (heap.getMax()->distance > curNeighbor.distance)
                        {
                            heap.replaceMax(curNeighbor);
                            radius = heap.getMax()->distance;
                        }
                    }
                }
            }

            if (!stack.empty())
            {
                cur = stack.pop();
                DAAL_PREFETCH_READ_T0(treeNodes + cur.nodeIndex);
            }
            else { break; }
        }
        else
        {
            DAAL_ASSERT(node->dimension < data.getNumberOfColumns());
            val = query[node->dimension];
            diff = val - node->cutPoint;

            if (cur.minDistance <= radius)
            {
                cur.nodeIndex    = (diff < 0) ? node->leftIndex : node->rightIndex;
                toPush.nodeIndex = (diff < 0) ? node->rightIndex : node->leftIndex;

                val -= node->cutPoint;
                toPush.minDistance = cur.minDistance + val * val;
                stack.push(toPush);
            }
            else if (!stack.empty())
            {
                cur = stack.pop();
                DAAL_PREFETCH_READ_T0(treeNodes + cur.nodeIndex);
            }
            else { break; }
        }
    }
}

template<typename algorithmFPType, CpuType cpu>
size_t KNNClassificationPredictDistrStep2Kernel<algorithmFPType, defaultDense, cpu>::
    findList(const algorithmFPType * query, size_t featureCount, algorithmFPType radius, size_t * list,
             size_t listCapacity, size_t key, Stack<SearchNode<algorithmFPType>, cpu> & stack,
             const PartitioningKDTreeTable & kdTreeTable)
{
    DAAL_ASSERT(query);

    size_t procCount = 0;

    stack.reset();
    size_t i, j, closeIdx, farIdx;
    SearchNode<algorithmFPType> cur, toPush;
    const PartitioningKDTreeNode * node = nullptr;
    cur.nodeIndex = 0;
    cur.minDistance = 0;

    algorithmFPType diff, val;

    for (;;)
    {
        node = static_cast<const PartitioningKDTreeNode *>(kdTreeTable.getArray()) + cur.nodeIndex;
        if (node->isLeaf)
        {
            DAAL_ASSERT(node->dimension < featureCount);
            diff = query[node->dimension] - node->cutPoint;
            closeIdx = (diff < 0) ? node->leftIndex : node->rightIndex;
            farIdx = (diff >= 0) ? node->leftIndex : node->rightIndex;

            if (closeIdx != key)
            {
                DAAL_ASSERT(procCount + 1 < listCapacity);
                list[procCount++] = closeIdx;
            }

            if (farIdx != key && (diff < 0 ? -diff : diff) <= radius)
            {
                DAAL_ASSERT(procCount + 1 < listCapacity);
                list[procCount++] = farIdx;
            }

            if (!stack.empty())
            {
                cur = stack.pop();
                DAAL_PREFETCH_READ_T0(static_cast<const KDTreeNode *>(kdTreeTable.getArray()) + cur.nodeIndex);
            }
            else { break; }

        }
        else
        {
            DAAL_ASSERT(node->dimension < featureCount);
            val = query[node->dimension];
            diff = val - node->cutPoint;

            if (cur.minDistance <= radius)
            {
                cur.nodeIndex = (diff < 0) ? node->leftIndex : node->rightIndex;
                toPush.nodeIndex = (diff < 0) ? node->rightIndex : node->leftIndex;
                val -= node->cutPoint;
                toPush.minDistance = cur.minDistance + val * val;
                if (toPush.minDistance <= radius)
                {
                    stack.push(toPush);
                }
            }
            else if (!stack.empty())
            {
                cur = stack.pop();
                DAAL_PREFETCH_READ_T0(static_cast<const KDTreeNode *>(kdTreeTable.getArray()) + cur.nodeIndex);
            }
            else { break; }
        }
    }

    return procCount;
}

template<typename algorithmFPType, CpuType cpu>
Status KNNClassificationPredictDistrStep2Kernel<algorithmFPType, defaultDense, cpu>::
    predict(algorithmFPType & predictedClass, const Heap<GlobalNeighbors<algorithmFPType, cpu>, cpu> & heap, size_t k)
{
    const size_t heapSize = heap.size();
    if (heapSize < 1)
    {
        predictedClass = 0;
        return Status();
    }

    data_management::BlockDescriptor<algorithmFPType> labelBD;
    algorithmFPType * const classes = static_cast<algorithmFPType *>(daal_malloc(heapSize * sizeof(*classes)));
    DAAL_CHECK_MALLOC(classes);

    for (size_t i = 0; i < heapSize; ++i)
    {
        classes[i] = heap[i].classLabel;
    }
    daal::algorithms::internal::qSort<algorithmFPType, cpu>(heapSize, classes);
    algorithmFPType currentClass = classes[0];
    algorithmFPType winnerClass = currentClass;
    size_t currentWeight = 1;
    size_t winnerWeight = currentWeight;
    for (size_t i = 1; i < heapSize; ++i)
    {
        if (classes[i] == currentClass)
        {
            if((++currentWeight) > winnerWeight)
            {
                winnerWeight = currentWeight;
                winnerClass = currentClass;
            }
        }
        else
        {
            currentWeight = 1;
            currentClass = classes[i];
        }
    }
    predictedClass = winnerClass;
    daal_free(classes);

    return Status();
}

} // namespace internal
} // namespace prediction
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
