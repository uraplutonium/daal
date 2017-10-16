/* file: kdtree_knn_classification_train_kernel.h */
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
//  Declaration of structure containing kernels for K-Nearest Neighbors training.
//--
*/

#ifndef __KDTREE_KNN_CLASSIFICATION_TRAIN_KERNEL_H__
#define __KDTREE_KNN_CLASSIFICATION_TRAIN_KERNEL_H__

#include "numeric_table.h"
#include "algorithm_base_common.h"
#include "kdtree_knn_classification_training_types.h"
#include "service_error_handling.h"
#include "kdtree_knn_classification_model_impl.h"

#if defined(_MSC_VER)
    #define DAAL_FORCEINLINE __forceinline
    #define DAAL_FORCENOINLINE __declspec(noinline)
#else
    #define DAAL_FORCEINLINE inline __attribute__((always_inline))
    #define DAAL_FORCENOINLINE __attribute__((noinline))
#endif

#define __KDTREE_SAMPLES_PER_NODE (1024 + 1)

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

using namespace daal::data_management;
using namespace daal::services;

template <typename T, CpuType cpu> class Queue;
struct BuildNode;
template <typename T> struct BoundingBox;
template <typename algorithmFpType, CpuType cpu> struct IndexValuePair;

template <typename algorithmFpType, CpuType cpu>
class BaseKernel : public daal::algorithms::Kernel
{
protected:
    void allocateTableData(size_t rowCount, NumericTable & table);

    void setNumericTableValue(NumericTable & table, size_t value);

    size_t calculateColor(size_t loop, size_t loops, size_t nodeCount, size_t nodeIndex);

    void copyNTRows(size_t firstSrcRow, size_t firstDestRow, size_t rowCount, const NumericTable & src, NumericTable & dest);

    Status buildKDTree(NumericTable & x, NumericTable & y, int seed,
                       SharedPtr<KDTreeTable> & kDTreeTable,
                       size_t & rootKDTreeNodeIndex,
                       size_t & lastContiguousKDTreeNodeIndex);

    Status buildFirstPartOfKDTree(Queue<BuildNode, cpu> & q, BoundingBox<algorithmFpType> * & bboxQ,
                                  const NumericTable & x, SharedPtr<KDTreeTable> & kDTreeTable,
                                  size_t & rootKDTreeNodeIndex, size_t & lastContiguousKDTreeNodeIndex,
                                  size_t * indexes, int seed);

    Status buildSecondPartOfKDTree(Queue<BuildNode, cpu> & q, BoundingBox<algorithmFpType> * & bboxQ,
                                   const NumericTable & x, SharedPtr<KDTreeTable> & kDTreeTable,
                                   size_t & rootKDTreeNodeIndex, size_t & lastContiguousKDTreeNodeIndex,
                                   size_t * indexes, int seed);

    Status computeLocalBoundingBoxOfKDTree(BoundingBox<algorithmFpType> * localBBox, const NumericTable & x, const size_t * indexes);

    size_t selectDimensionSophisticated(size_t start, size_t end, size_t * sampleIndexes, algorithmFpType * sampleValues, size_t sampleCount,
                                        const NumericTable & x, const size_t * indexes, int seed);

    algorithmFpType computeApproximatedMedianInParallel(size_t start, size_t end, size_t dimension, algorithmFpType upper,
                                                        const NumericTable & x, const size_t * indexes, int seed, algorithmFpType * subSamples,
                                                        size_t subSampleCapacity, Status &status);

    DAAL_FORCEINLINE size_t computeBucketID(algorithmFpType * samples, size_t sampleCount, algorithmFpType * subSamples,
                                                size_t subSampleCount, size_t subSampleCount16, algorithmFpType value);

    size_t adjustIndexesInParallel(size_t start, size_t end, size_t dimension, algorithmFpType median, const NumericTable & x, size_t * indexes);

    void copyBBox(BoundingBox<algorithmFpType> * dest, const BoundingBox<algorithmFpType> * src, size_t n);

    Status rearrangePoints(NumericTable & x, const size_t * indexes);

    Status buildSecondPartOfKDTree(Queue<BuildNode, cpu> & q, BoundingBox<algorithmFpType> * & bboxQ, const NumericTable & x,
                                 kdtree_knn_classification::Model & r, size_t * indexes, int seed);

    algorithmFpType computeApproximatedMedianInSerial(size_t start, size_t end, size_t dimension, algorithmFpType upper,
                                                      IndexValuePair<algorithmFpType, cpu> * inSortValues,
                                                      IndexValuePair<algorithmFpType, cpu> * outSortValues,
                                                      size_t sortValueCount,
                                                      const NumericTable & x,
                                                      size_t * indexes, int seed);

    size_t adjustIndexesInSerial(size_t start, size_t end, size_t dimension, algorithmFpType median, const NumericTable & x, size_t * indexes);

    DAAL_FORCEINLINE void radixSort(IndexValuePair<algorithmFpType, cpu> * inValues, size_t valueCount,
                                    IndexValuePair<algorithmFpType, cpu> * outValues);
};

template <typename algorithmFpType, training::Method method, CpuType cpu>
class KNNClassificationTrainBatchKernel
{};

template <typename algorithmFpType, CpuType cpu>
class KNNClassificationTrainBatchKernel<algorithmFpType, training::defaultDense, cpu> : public BaseKernel<algorithmFpType, cpu>
{
public:
    services::Status compute(NumericTable * x, NumericTable * y, kdtree_knn_classification::Model * r, const daal::algorithms::Parameter * par);

protected:
    using BaseKernel<algorithmFpType, cpu>::buildKDTree;
};

template <typename algorithmFpType, training::Method method, CpuType cpu>
class KNNClassificationTrainDistrStep1Kernel
{};

template <typename algorithmFpType, CpuType cpu>
class KNNClassificationTrainDistrStep1Kernel<algorithmFpType, training::defaultDense, cpu> : public BaseKernel<algorithmFpType, cpu>
{
public:
    services::Status compute(NumericTable * x, NumericTable * y, NumericTable * r, const daal::algorithms::Parameter * par);

protected:
    void computeLocalBoundingBox(const NumericTable & x, NumericTable & r);
};

template <typename algorithmFpType, training::Method method, CpuType cpu>
class KNNClassificationTrainDistrStep2Kernel
{};

template <typename algorithmFpType, CpuType cpu>
class KNNClassificationTrainDistrStep2Kernel<algorithmFpType, training::defaultDense, cpu> : public BaseKernel<algorithmFpType, cpu>
{
public:
    services::Status compute(size_t perNodeNTCount, NumericTable ** perNodeNTs, NumericTable * r, NumericTable * loops,
                             const daal::algorithms::Parameter * par);

protected:
    using BaseKernel<algorithmFpType, cpu>::allocateTableData;

    void computeGlobalBoundingBox(size_t perNodeNTCount, NumericTable ** perNodeNTs, NumericTable & r);
    size_t computeNumberOfLoops(size_t perNodeNTCount, size_t & numberOfLoops);
};

template <typename algorithmFpType, training::Method method, CpuType cpu>
class KNNClassificationTrainDistrStep3Kernel
{};

template <typename algorithmFpType, CpuType cpu>
class KNNClassificationTrainDistrStep3Kernel<algorithmFpType, training::defaultDense, cpu> : public BaseKernel<algorithmFpType, cpu>
{
public:
    services::Status compute(NumericTable * x, NumericTable * y, NumericTable * b, NumericTable * numberOfLoops, int loop, int nodeIndex,
                             int nodeCount, NumericTable * samples, NumericTable * dim, NumericTable * colorTable,
                             const daal::algorithms::Parameter * par);

protected:
    using BaseKernel<algorithmFpType, cpu>::setNumericTableValue;
    using BaseKernel<algorithmFpType, cpu>::calculateColor;

    size_t selectDimension(const NumericTable & b, size_t idx);
    void performSampling(size_t dimension, const NumericTable & x, const NumericTable & b, NumericTable & samples, size_t idx, int seed);
};

template <typename algorithmFpType, training::Method method, CpuType cpu>
class KNNClassificationTrainDistrStep4Kernel
{};

template <typename algorithmFpType, CpuType cpu>
class KNNClassificationTrainDistrStep4Kernel<algorithmFpType, training::defaultDense, cpu> : public BaseKernel<algorithmFpType, cpu>
{
public:
    services::Status compute(NumericTable * x, NumericTable * y, NumericTable * d, NumericTable * b, size_t nodeCount, NumericTable ** nodeSamples,
                             NumericTable * h, const daal::algorithms::Parameter * par);

protected:
    void prepareGlobalSamples(size_t nodeCount, NumericTable ** nodeSamples, size_t dimension, const NumericTable & b,
                              algorithmFpType * globalSamples, size_t globalSampleCount);

    void buildHistogram(const algorithmFpType * globalSamples, size_t globalSampleCount, const NumericTable & x, size_t dimension, NumericTable & h);

    size_t computeBucketID(const algorithmFpType * globalSamples, size_t globalSampleCount, algorithmFpType value);
};

template <typename algorithmFpType, training::Method method, CpuType cpu>
class KNNClassificationTrainDistrStep5Kernel
{};

template <typename algorithmFpType, CpuType cpu>
class KNNClassificationTrainDistrStep5Kernel<algorithmFpType, training::defaultDense, cpu> : public BaseKernel<algorithmFpType, cpu>
{
public:
    services::Status compute(NumericTable * x, NumericTable * y, NumericTable * d, bool ispg, size_t nodeCount, NumericTable ** nodeHistograms,
                             const size_t * nodeIDs, NumericTable * dataToSend, NumericTable * labelsToSend, NumericTable * medianTable,
                             NumericTable * markers, const daal::algorithms::Parameter * par);

protected:
    using BaseKernel<algorithmFpType, cpu>::allocateTableData;

    size_t reduceHistogram(size_t nodeCount, const NumericTable * const * nodeHistograms, size_t sampleCount, size_t * groupHistogram);

    algorithmFpType computeApproximatedMedian(size_t sampleCount, const algorithmFpType * samples, const size_t * groupHistogram, size_t pointCount);

    void prepareDataToSend(const NumericTable & x, const NumericTable & y, size_t dimension, algorithmFpType median, bool isPartnerGreater,
                           NumericTable & xToSend, NumericTable & yToSend, NumericTable & sendMarkers);
};

template <typename algorithmFpType, training::Method method, CpuType cpu>
class KNNClassificationTrainDistrStep6Kernel
{};

template <typename algorithmFpType, CpuType cpu>
class KNNClassificationTrainDistrStep6Kernel<algorithmFpType, training::defaultDense, cpu> : public BaseKernel<algorithmFpType, cpu>
{
public:
    services::Status compute(NumericTable * x, NumericTable * y, NumericTable * px, NumericTable * py, NumericTable * markers, NumericTable * cx,
                             NumericTable * cy, const daal::algorithms::Parameter * par);

protected:
    using BaseKernel<algorithmFpType, cpu>::allocateTableData;
    using BaseKernel<algorithmFpType, cpu>::copyNTRows;

    void concatenate(const NumericTable & x, const NumericTable & y, const NumericTable & px, const NumericTable & py, const NumericTable & markers,
                     NumericTable & cx, NumericTable & cy);
    void copyNT(const NumericTable & src, NumericTable & dest);
};

template <typename algorithmFpType, training::Method method, CpuType cpu>
class KNNClassificationTrainDistrStep7Kernel
{};

template <typename algorithmFpType, CpuType cpu>
class KNNClassificationTrainDistrStep7Kernel<algorithmFpType, training::defaultDense, cpu> : public BaseKernel<algorithmFpType, cpu>
{
public:
    services::Status compute(NumericTable * inputb, NumericTable * loops, int loop, size_t nodeCount, NumericTable ** dimensions,
                             NumericTable ** medians, size_t * dimensionNodeIDs, size_t * medianNodeIDs, PartialModel * inputpm,
                             PartialModel * outputpm, NumericTable * outputb, const daal::algorithms::Parameter * par);

protected:
    using BaseKernel<algorithmFpType, cpu>::calculateColor;
    using BaseKernel<algorithmFpType, cpu>::copyNTRows;

    void growPartitioningKDTree(int loop, int loops, size_t nodeCount, const size_t * dimensions, const algorithmFpType * medians,
                                const PartialModel * inputpm, PartialModel & outputpm);

    void updateBoundingBoxes(int loop, int loops, size_t nodeCount, const size_t * dimensions, const algorithmFpType * medians, NumericTable & b);
};

template <typename algorithmFpType, training::Method method, CpuType cpu>
class KNNClassificationTrainDistrStep8Kernel
{};

template <typename algorithmFpType, CpuType cpu>
class KNNClassificationTrainDistrStep8Kernel<algorithmFpType, training::defaultDense, cpu> : public BaseKernel<algorithmFpType, cpu>
{
public:
    services::Status compute(NumericTable * x, NumericTable * y, PartialModel * inputpm, PartialModel * outputpm,
                             const daal::algorithms::Parameter * par);

protected:
    using BaseKernel<algorithmFpType, cpu>::buildKDTree;
};

} // namespace internal
} // namespace training
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
