/* file: kdtree_knn_classification_train_container.h */
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
//  Implementation of K-Nearest Neighbors container.
//--
*/

#ifndef __KDTREE_KNN_CLASSIFICATION_TRAIN_CONTAINER_H__
#define __KDTREE_KNN_CLASSIFICATION_TRAIN_CONTAINER_H__

#include "kernel.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/data_collection.h"
#include "services/daal_shared_ptr.h"
#include "kdtree_knn_classification_training_batch.h"
#include "kdtree_knn_classification_training_distributed.h"
#include "kdtree_knn_classification_train_kernel.h"
#include "kdtree_knn_classification_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace training
{

using namespace daal::data_management;

/**
 *  \brief Initialize list of K-Nearest Neighbors kernels with implementations for supported architectures
 */
template <typename algorithmFpType, training::Method method, CpuType cpu>
BatchContainer<algorithmFpType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KNNClassificationTrainBatchKernel, algorithmFpType, method);
}

template <typename algorithmFpType, training::Method method, CpuType cpu>
BatchContainer<algorithmFpType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

/**
 *  \brief Choose appropriate kernel to calculate K-Nearest Neighbors model.
 */
template <typename algorithmFpType, training::Method method, CpuType cpu>
services::Status BatchContainer<algorithmFpType, method, cpu>::compute()
{
    const classifier::training::Input * const input = static_cast<classifier::training::Input *>(_in);
    Result * const result = static_cast<Result *>(_res);

    const NumericTablePtr x = input->get(classifier::training::data);
    const NumericTablePtr y = input->get(classifier::training::labels);

    const kdtree_knn_classification::ModelPtr r = result->get(classifier::training::model);

    const daal::algorithms::Parameter * const par = _par;
    daal::services::Environment::env & env = *_env;

    const bool copy = (static_cast<const kdtree_knn_classification::Parameter *>(par)->dataUseInModel == doNotUse);
    r->impl()->setData<algorithmFpType>(x, copy);
    r->impl()->setLabels<algorithmFpType>(y, copy);

    __DAAL_CALL_KERNEL(env, internal::KNNClassificationTrainBatchKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFpType, method),    \
                       compute, r->impl()->getData().get(), r->impl()->getLabels().get(), r.get(), par);
}

template <typename algorithmFpType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFpType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KNNClassificationTrainDistrStep1Kernel, algorithmFpType, method);
}

template <typename algorithmFpType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFpType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFpType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFpType, method, cpu>::compute()
{
    const DistributedInput<step1Local> * const input = static_cast<DistributedInput<step1Local> *>(_in);
    DistributedPartialResultStep1 * const partialResult = static_cast<DistributedPartialResultStep1 *>(_pres);
    const daal::algorithms::Parameter * const par = _par;
    daal::services::Environment::env & env = *_env;

    const NumericTablePtr x = input->get(classifier::training::data);
    const NumericTablePtr y = input->get(classifier::training::labels);
    const NumericTablePtr r = partialResult->get(boundingBoxes);

    __DAAL_CALL_KERNEL(env, internal::KNNClassificationTrainDistrStep1Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFpType, method),    \
                       compute, x.get(), y.get(), r.get(), par);
}

template <typename algorithmFpType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFpType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFpType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFpType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KNNClassificationTrainDistrStep2Kernel, algorithmFpType, method);
}

template <typename algorithmFpType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFpType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFpType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFpType, method, cpu>::compute()
{
    const DistributedInput<step2Master> * const input = static_cast<DistributedInput<step2Master> *>(_in);
    DistributedPartialResultStep2 * const partialResult = static_cast<DistributedPartialResultStep2 *>(_pres);
    const daal::algorithms::Parameter * const par = _par;
    daal::services::Environment::env & env = *_env;

    const KeyValueDataCollectionConstPtr perNodePartials = input->get(inputOfStep2);
    const size_t nodeCount = perNodePartials->size();
    NumericTable ** const nts = new NumericTable * [nodeCount];
    for (size_t i = 0; i < nodeCount; ++i)
    {
        nts[i] = static_cast<NumericTable *>(perNodePartials->getValueByIndex(i).get());
    }

    const NumericTablePtr r = partialResult->get(globalBoundingBoxes);
    const NumericTablePtr loops = partialResult->get(numberOfLoops);

    const services::Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::KNNClassificationTrainDistrStep2Kernel,                                 \
                                                         __DAAL_KERNEL_ARGUMENTS(algorithmFpType, method), compute,                             \
                                                         nodeCount, nts, r.get(), loops.get(), par);

    delete[] nts;

    return s;
}

template <typename algorithmFpType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFpType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFpType, Method method, CpuType cpu>
DistributedContainer<step3Local, algorithmFpType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KNNClassificationTrainDistrStep3Kernel, algorithmFpType, method);
}

template <typename algorithmFpType, Method method, CpuType cpu>
DistributedContainer<step3Local, algorithmFpType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFpType, Method method, CpuType cpu>
services::Status DistributedContainer<step3Local, algorithmFpType, method, cpu>::compute()
{
    const DistributedInput<step3Local> * const input = static_cast<DistributedInput<step3Local> *>(_in);
    DistributedPartialResultStep3 * const partialResult = static_cast<DistributedPartialResultStep3 *>(_pres);
    const daal::algorithms::Parameter * const par = _par;
    daal::services::Environment::env & env = *_env;

    const NumericTablePtr x = input->get(dataForStep3);
    const NumericTablePtr y = input->get(labelsForStep3);
    const NumericTablePtr b = input->get(boundingBoxesForStep3);
    const NumericTablePtr numberOfLoops = input->get(numberOfLoopsForStep3);
    const int loopNumber = input->get(loopNumberForStep3);
    const int nodeIndex = input->get(nodeIndexForStep3);
    const int nodeCount = input->get(nodeCountForStep3);

    const NumericTablePtr s = partialResult->get(localSamples);
    const NumericTablePtr d = partialResult->get(dimension);
    const NumericTablePtr colorTable = partialResult->get(color);

    __DAAL_CALL_KERNEL(env, internal::KNNClassificationTrainDistrStep3Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFpType, method),    \
                       compute, x.get(), y.get(), b.get(), numberOfLoops.get(), loopNumber, nodeIndex, nodeCount, s.get(), d.get(), \
                       colorTable.get(), par);
}

template <typename algorithmFpType, Method method, CpuType cpu>
services::Status DistributedContainer<step3Local, algorithmFpType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFpType, Method method, CpuType cpu>
DistributedContainer<step4Local, algorithmFpType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KNNClassificationTrainDistrStep4Kernel, algorithmFpType, method);
}

template <typename algorithmFpType, Method method, CpuType cpu>
DistributedContainer<step4Local, algorithmFpType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFpType, Method method, CpuType cpu>
services::Status DistributedContainer<step4Local, algorithmFpType, method, cpu>::compute()
{
    const DistributedInput<step4Local> * const input = static_cast<DistributedInput<step4Local> *>(_in);
    DistributedPartialResultStep4 * const partialResult = static_cast<DistributedPartialResultStep4 *>(_pres);
    const daal::algorithms::Parameter * const par = _par;
    daal::services::Environment::env & env = *_env;

    const NumericTablePtr x = input->get(dataForStep4);
    const NumericTablePtr y = input->get(labelsForStep4);
    const NumericTablePtr d = input->get(dimensionForStep4);
    const NumericTablePtr b = input->get(boundingBoxesForStep4);

    const KeyValueDataCollectionConstPtr perNodePartials = input->get(samplesForStep4);
    const size_t nodeCount = perNodePartials->size();
    NumericTable ** const nts = new NumericTable * [nodeCount];
    for (size_t i = 0; i < nodeCount; ++i)
    {
        nts[i] = static_cast<NumericTable *>(perNodePartials->getValueByIndex(i).get());
    }

    const NumericTablePtr h = partialResult->get(localHistogram);

    const services::Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::KNNClassificationTrainDistrStep4Kernel,                                 \
                                                         __DAAL_KERNEL_ARGUMENTS(algorithmFpType, method), compute,                             \
                                                         x.get(), y.get(), d.get(), b.get(), nodeCount, nts, h.get(), par);

    delete[] nts;

    return s;
}

template <typename algorithmFpType, Method method, CpuType cpu>
services::Status DistributedContainer<step4Local, algorithmFpType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFpType, Method method, CpuType cpu>
DistributedContainer<step5Local, algorithmFpType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KNNClassificationTrainDistrStep5Kernel, algorithmFpType, method);
}

template <typename algorithmFpType, Method method, CpuType cpu>
DistributedContainer<step5Local, algorithmFpType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFpType, Method method, CpuType cpu>
services::Status DistributedContainer<step5Local, algorithmFpType, method, cpu>::compute()
{
    const DistributedInput<step5Local> * const input = static_cast<DistributedInput<step5Local> *>(_in);
    DistributedPartialResultStep5 * const partialResult = static_cast<DistributedPartialResultStep5 *>(_pres);
    const daal::algorithms::Parameter * const par = _par;
    daal::services::Environment::env & env = *_env;

    const NumericTablePtr x = input->get(dataForStep5);
    const NumericTablePtr y = input->get(labelsForStep5);
    const NumericTablePtr d = input->get(dimensionForStep5);
    const bool ispg = input->get(isPartnerGreaterForStep5);

    const KeyValueDataCollectionConstPtr perNodePartials = input->get(histogramForStep5);
    const size_t nodeCount = perNodePartials->size();
    NumericTable ** const nts = new NumericTable * [nodeCount];
    size_t * nodeIDs = new size_t[nodeCount];
    for (size_t i = 0; i < nodeCount; ++i)
    {
        nts[i] = static_cast<NumericTable *>(perNodePartials->getValueByIndex(i).get());
        nodeIDs[i] = perNodePartials->getKeyByIndex(i);
    }

    NumericTablePtr dataToSend = partialResult->get(dataForPartner);
    NumericTablePtr labelsToSend = partialResult->get(labelsForPartner);
    NumericTablePtr medianTable = partialResult->get(median);
    NumericTablePtr markersTable = partialResult->get(markers);

    const services::Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::KNNClassificationTrainDistrStep5Kernel,                                 \
                                                         __DAAL_KERNEL_ARGUMENTS(algorithmFpType, method), compute,                             \
                                                         x.get(), y.get(), d.get(), ispg, nodeCount, nts, nodeIDs, dataToSend.get(),            \
                                                         labelsToSend.get(), medianTable.get(), markersTable.get(), par);

    delete[] nodeIDs;
    delete[] nts;

    return s;
}

template <typename algorithmFpType, Method method, CpuType cpu>
services::Status DistributedContainer<step5Local, algorithmFpType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFpType, Method method, CpuType cpu>
DistributedContainer<step6Local, algorithmFpType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KNNClassificationTrainDistrStep6Kernel, algorithmFpType, method);
}

template <typename algorithmFpType, Method method, CpuType cpu>
DistributedContainer<step6Local, algorithmFpType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFpType, Method method, CpuType cpu>
services::Status DistributedContainer<step6Local, algorithmFpType, method, cpu>::compute()
{
    const DistributedInput<step6Local> * const input = static_cast<DistributedInput<step6Local> *>(_in);
    DistributedPartialResultStep6 * const partialResult = static_cast<DistributedPartialResultStep6 *>(_pres);
    const daal::algorithms::Parameter * const par = _par;
    daal::services::Environment::env & env = *_env;

    const NumericTablePtr x = input->get(dataForStep6);
    const NumericTablePtr y = input->get(labelsForStep6);
    const NumericTablePtr px = input->get(dataFromPartnerForStep6);
    const NumericTablePtr py = input->get(labelsFromPartnerForStep6);
    const NumericTablePtr markers = input->get(markersForStep6);

    NumericTablePtr cx = partialResult->get(concatenatedData);
    NumericTablePtr cy = partialResult->get(concatenatedLabels);

    __DAAL_CALL_KERNEL(env, internal::KNNClassificationTrainDistrStep6Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFpType, method),       \
                       compute, x.get(), y.get(), px.get(), py.get(), markers.get(), cx.get(), cy.get(), par);
}

template <typename algorithmFpType, Method method, CpuType cpu>
services::Status DistributedContainer<step6Local, algorithmFpType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFpType, Method method, CpuType cpu>
DistributedContainer<step7Master, algorithmFpType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KNNClassificationTrainDistrStep7Kernel, algorithmFpType, method);
}

template <typename algorithmFpType, Method method, CpuType cpu>
DistributedContainer<step7Master, algorithmFpType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFpType, Method method, CpuType cpu>
services::Status DistributedContainer<step7Master, algorithmFpType, method, cpu>::compute()
{
    const DistributedInput<step7Master> * const input = static_cast<DistributedInput<step7Master> *>(_in);
    DistributedPartialResultStep7 * const partialResult = static_cast<DistributedPartialResultStep7 *>(_pres);
    const daal::algorithms::Parameter * const par = _par;
    daal::services::Environment::env & env = *_env;

    const NumericTablePtr inputb = input->get(boundingBoxesForStep7);
    const NumericTablePtr loops = input->get(numberOfLoopsForStep7);
    const int loop = input->get(loopNumberForStep7);
    const daal::services::SharedPtr<PartialModel> inputpm = input->get(partialModelForStep7);

    const KeyValueDataCollectionConstPtr perNodeDimensions = input->get(dimensionForStep7);
    const size_t nodeCount = perNodeDimensions->size();
    NumericTable ** const dimensions = new NumericTable * [nodeCount];
    size_t * dimensionNodeIDs = new size_t[nodeCount];
    for (size_t i = 0; i < nodeCount; ++i)
    {
        dimensions[i] = static_cast<NumericTable *>(perNodeDimensions->getValueByIndex(i).get());
        dimensionNodeIDs[i] = perNodeDimensions->getKeyByIndex(i);
    }

    const KeyValueDataCollectionConstPtr perNodeMedians = input->get(medianForStep7);
    NumericTable ** const medians = new NumericTable * [nodeCount];
    size_t * medianNodeIDs = new size_t[nodeCount];
    for (size_t i = 0; i < nodeCount; ++i)
    {
        medians[i] = static_cast<NumericTable *>(perNodeMedians->getValueByIndex(i).get());
        medianNodeIDs[i] = perNodeMedians->getKeyByIndex(i);
    }

    const daal::services::SharedPtr<PartialModel> outputpm = partialResult->get(partialModelOfStep7);
    const NumericTablePtr outputb = partialResult->get(boundingBoxesOfStep7ForStep3);

    const services::Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::KNNClassificationTrainDistrStep7Kernel,                                 \
                                                         __DAAL_KERNEL_ARGUMENTS(algorithmFpType, method), compute,                             \
                                                         inputb.get(), loops.get(), loop, nodeCount, dimensions, medians, dimensionNodeIDs,     \
                                                         medianNodeIDs, inputpm.get(), outputpm.get(), outputb.get(), par);

    delete[] medians;
    delete[] medianNodeIDs;
    delete[] dimensionNodeIDs;
    delete[] dimensions;

    return s;
}

template <typename algorithmFpType, Method method, CpuType cpu>
services::Status DistributedContainer<step7Master, algorithmFpType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFpType, Method method, CpuType cpu>
DistributedContainer<step8Local, algorithmFpType, method, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KNNClassificationTrainDistrStep8Kernel, algorithmFpType, method);
}

template <typename algorithmFpType, Method method, CpuType cpu>
DistributedContainer<step8Local, algorithmFpType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFpType, Method method, CpuType cpu>
services::Status DistributedContainer<step8Local, algorithmFpType, method, cpu>::compute()
{
    const DistributedInput<step8Local> * const input = static_cast<DistributedInput<step8Local> *>(_in);
    DistributedPartialResultStep8 * const partialResult = static_cast<DistributedPartialResultStep8 *>(_pres);
    const daal::algorithms::Parameter * const par = _par;
    daal::services::Environment::env & env = *_env;

    const NumericTablePtr x = input->get(dataForStep8);
    const NumericTablePtr y = input->get(labelsForStep8);
    const daal::services::SharedPtr<PartialModel> inputpm = input->get(partialModelForStep8);

    const daal::services::SharedPtr<PartialModel> outputpm = partialResult->get(partialModel);

    const bool copyTablesToModel = true;
    outputpm->impl()->setData<algorithmFpType>(x, copyTablesToModel);
    outputpm->impl()->setLabels<algorithmFpType>(y, copyTablesToModel);

    NumericTable *xCopy = outputpm->impl()->getData().get();
    NumericTable *yCopy = outputpm->impl()->getLabels().get();

    __DAAL_CALL_KERNEL(env, internal::KNNClassificationTrainDistrStep8Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFpType, method),    \
                       compute, xCopy, yCopy, inputpm.get(), outputpm.get(), par);
}

template <typename algorithmFpType, Method method, CpuType cpu>
services::Status DistributedContainer<step8Local, algorithmFpType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

} // namespace training
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
