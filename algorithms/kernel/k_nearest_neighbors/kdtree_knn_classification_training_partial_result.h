/* file: kdtree_knn_classification_training_partial_result.h */
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
//  Implementation of the class defining the K-Nearest Neighbors (kNN) model
//--
*/

#ifndef __KDTREE_KNN_CLASSIFICATION_TRAINING_PARTIAL_RESULT_
#define __KDTREE_KNN_CLASSIFICATION_TRAINING_PARTIAL_RESULT_

#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_training_types.h"
#include "data_management/data/homogen_numeric_table.h"
#include "kdtree_knn_impl.i"
#include "kdtree_knn_classification_train_kernel.h"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace training
{
namespace interface1
{

using namespace daal::services;
using namespace daal::data_management;

template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep1::allocate(const daal::algorithms::Input * input,
                                                                     const daal::algorithms::Parameter * parameter, const int method)
{
    const DistributedInput<step1Local> * const in = static_cast<const DistributedInput<step1Local> *>(input);
    const size_t featureCount = in->getNumberOfFeatures();
    Argument::set(boundingBoxes, HomogenNumericTable<algorithmFPType>::create(featureCount, 2, NumericTable::doAllocate));
    return services::Status();
}

template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep2::allocate(const daal::algorithms::Input * input,
                                                                     const daal::algorithms::Parameter * parameter, const int method)
{
    const DistributedInput<step2Master> * const in = static_cast<const DistributedInput<step2Master> *>(input);
    const size_t featureCount = in->getNumberOfFeatures();
    Argument::set(globalBoundingBoxes, HomogenNumericTable<algorithmFPType>::create(featureCount, 0, NumericTable::notAllocate));
    Argument::set(numberOfLoops, HomogenNumericTable<size_t>::create(1, 1, NumericTable::doAllocate));
    return services::Status();
}

template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep3::allocate(const daal::algorithms::Input * input,
                                                                     const daal::algorithms::Parameter * parameter, const int method)
{
    Argument::set(localSamples, HomogenNumericTable<algorithmFPType>::create(1, __KDTREE_SAMPLES_PER_NODE, NumericTable::doAllocate));
    Argument::set(dimension, HomogenNumericTable<size_t>::create(1, 1, NumericTable::doAllocate));
    Argument::set(color, HomogenNumericTable<size_t>::create(1, 1, NumericTable::doAllocate));
    return services::Status();
}

template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep4::allocate(const daal::algorithms::Input * input,
                                                                     const daal::algorithms::Parameter * parameter, const int method)
{
    const DistributedInput<step4Local> * const in = static_cast<const DistributedInput<step4Local> *>(input);
    const KeyValueDataCollectionConstPtr perNodePartials = in->get(samplesForStep4);
    const size_t nodeCount = perNodePartials->size();
    const size_t globalSampleCount = nodeCount * __KDTREE_SAMPLES_PER_NODE + 1;
    Argument::set(localHistogram, HomogenNumericTable<algorithmFPType>::create(2, globalSampleCount, NumericTable::doAllocate));
    return services::Status();
}

template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep5::allocate(const daal::algorithms::Input * input,
                                                                     const daal::algorithms::Parameter * parameter, const int method)
{
    const DistributedInput<step5Local> * const in = static_cast<const DistributedInput<step5Local> *>(input);
    const size_t featureCount = in->getNumberOfFeatures();
    Argument::set(dataForPartner, HomogenNumericTable<algorithmFPType>::create(featureCount, 0,
                                                                               NumericTable::notAllocate));
    Argument::set(labelsForPartner, HomogenNumericTable<algorithmFPType>::create(1, 0, NumericTable::notAllocate));
    Argument::set(median, HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate));
    Argument::set(markers, HomogenNumericTable<algorithmFPType>::create(1, 0, NumericTable::notAllocate));
    return services::Status();
}

template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep6::allocate(const daal::algorithms::Input * input,
                                                                     const daal::algorithms::Parameter * parameter, const int method)
{
    const DistributedInput<step5Local> * const in = static_cast<const DistributedInput<step5Local> *>(input);
    const size_t featureCount = in->getNumberOfFeatures();

    data_management::NumericTableFeature temp;
    temp.setType<algorithmFPType>();

    SOANumericTablePtr cx = SOANumericTable::create(featureCount);
    cx->setArray(static_cast<algorithmFPType *>(0), 0); // Just to create the dictionary.
    cx->freeDataMemory(); // Just to mark it as notAllocated.
    cx->getDictionary()->setNumberOfFeatures(featureCount); // Sadly, setArray() hides number of features from the dictionary.
    cx->getDictionary()->setAllFeatures(temp); // Just to set type of all features. Also, no way to use featuresEqual flag.
    Argument::set(concatenatedData, cx);

    const size_t labelsColumnCount = 1;
    SOANumericTablePtr cy = SOANumericTable::create(labelsColumnCount);
    cy->setArray(static_cast<algorithmFPType *>(0), 0); // Just to create the dictionary.
    cy->freeDataMemory(); // Just to mark it as notAllocated.
    cy->getDictionary()->setNumberOfFeatures(labelsColumnCount); // Sadly, setArray() hides number of features from the dictionary.
    cy->getDictionary()->setAllFeatures(temp); // Just to set type of all features. Also, no way to use featuresEqual flag.
    Argument::set(concatenatedLabels, cy);

    return services::Status();
}

template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep7::allocate(const daal::algorithms::Input * input,
                                                                     const daal::algorithms::Parameter * parameter, const int method)
{
    const DistributedInput<step7Master> * const in = static_cast<const DistributedInput<step7Master> *>(input);
    const NumericTableConstPtr b = in->get(boundingBoxesForStep7);
    const size_t numberOfFeatures = in->getNumberOfFeatures();

    Argument::set(boundingBoxesOfStep7ForStep3, HomogenNumericTable<algorithmFPType>::create(b->getNumberOfColumns(),
                                                                                             b->getNumberOfRows(),
                                                                                             NumericTable::doAllocate));
    Argument::set(partialModelOfStep7, PartialModel::create(numberOfFeatures));
    return services::Status();
}

template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep8::allocate(const daal::algorithms::Input * input,
                                                                     const daal::algorithms::Parameter * parameter, const int method)
{
    // TODO: Replace to create
    const DistributedInput<step8Local> * const in = static_cast<const DistributedInput<step8Local> *>(input);
    Argument::set(partialModel, PartialModel::create(in->getNumberOfFeatures()));
    return services::Status();
}

} // namespace interface1
} // namespace training
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
