/* file: kdtree_knn_classification_predict_dense_default_distributed_container.h */
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
//  Implementation of K-Nearest Neighbors algorithm container - a class that contains fast K-Nearest Neighbors prediction kernels for supported
//  architectures.
//--
*/

#include "kdtree_knn_classification_predict_distributed.h"
#include "kdtree_knn_classification_predict_dense_default_distributed.h"
#include "data_management/data/data_collection.h"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace prediction
{

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::
    DistributedContainer(daal::services::Environment::env * daalEnv) : DistributedPredictionContainerIface()
{
    __DAAL_INITIALIZE_KERNELS(internal::KNNClassificationPredictDistrStep1Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::compute()
{
    const DistributedInput<step1Local> * const input = static_cast<const DistributedInput<step1Local> *>(_in);
    DistributedPartialResultStep1 * const result = static_cast<DistributedPartialResultStep1 *>(_pres);

    const data_management::NumericTableConstPtr a = input->get(kdtree_knn_classification::prediction::data);
    const services::SharedPtr<const PartialModel> m = input->get(kdtree_knn_classification::prediction::partialModel);
    const data_management::NumericTablePtr r = result->get(kdtree_knn_classification::prediction::keys);

    const daal::algorithms::Parameter * const par = _par;
    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::KNNClassificationPredictDistrStep1Kernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), \
                       compute, a.get(), m.get(), r.get(), par);
}

template <typename algorithmFpType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFpType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Local, algorithmFPType, method, cpu>::
    DistributedContainer(daal::services::Environment::env * daalEnv) : DistributedPredictionContainerIface()
{
    __DAAL_INITIALIZE_KERNELS(internal::KNNClassificationPredictDistrStep2Kernel, algorithmFPType, method);
}

template <typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Local, algorithmFPType, method, cpu>::compute()
{
    const DistributedInput<step2Local> * const input = static_cast<const DistributedInput<step2Local> *>(_in);
    DistributedPartialResultStep2 * const result = static_cast<DistributedPartialResultStep2 *>(_pres);

    const data_management::NumericTableConstPtr a = input->get(kdtree_knn_classification::prediction::arrangedData);
    const data_management::NumericTableConstPtr interm = input->get(kdtree_knn_classification::prediction::intermediatePrediction);
    const int key = input->get(kdtree_knn_classification::prediction::key);
    const int round = input->get(kdtree_knn_classification::prediction::round);
    const services::SharedPtr<const PartialModel> m = input->get(kdtree_knn_classification::prediction::partialModel);
    const data_management::NumericTablePtr r = result->get(kdtree_knn_classification::prediction::prediction);

    const data_management::KeyValueDataCollectionConstPtr perNodeCommResponses = input->get(communicationResponses);
    const size_t respCount = perNodeCommResponses->size();
    data_management::NumericTable ** const commResponses = new data_management::NumericTable * [respCount];
    size_t * const commResponsesNodeIDs = new size_t[respCount];
    for (size_t i = 0; i < respCount; ++i)
    {
        commResponses[i] = static_cast<data_management::NumericTable *>(perNodeCommResponses->getValueByIndex(i).get());
        commResponsesNodeIDs[i] = perNodeCommResponses->getKeyByIndex(i);
    }

    const data_management::KeyValueDataCollectionConstPtr perNodeCommInputQueries = input->get(communicationInputQueries);
    const size_t inputQueryCount = perNodeCommInputQueries->size();
    data_management::NumericTable ** const commInputQueries = new data_management::NumericTable * [inputQueryCount];
    size_t * const commInputQueriesNodeIDs = new size_t[inputQueryCount];
    for (size_t i = 0; i < inputQueryCount; ++i)
    {
        commInputQueries[i] = static_cast<data_management::NumericTable *>(perNodeCommInputQueries->getValueByIndex(i).get());
        commInputQueriesNodeIDs[i] = perNodeCommInputQueries->getKeyByIndex(i);
    }

    const data_management::KeyValueDataCollectionPtr perNodeCommQueries = result->get(communicationQueries);
    const size_t queryCount = perNodeCommQueries->size();
    data_management::NumericTable ** const commQueries = new data_management::NumericTable * [queryCount];
    size_t * const commQueriesNodeIDs = new size_t[queryCount];
    for (size_t i = 0; i < queryCount; ++i)
    {
        commQueries[i] = static_cast<data_management::NumericTable *>(perNodeCommQueries->getValueByIndex(i).get());
        commQueriesNodeIDs[i] = perNodeCommQueries->getKeyByIndex(i);
    }

    const data_management::KeyValueDataCollectionPtr perNodeCommOutputResponses = result->get(communicationOutputResponses);
    const size_t outputResponsesCount = perNodeCommOutputResponses->size();
    data_management::NumericTable ** const commOutputResponses = new data_management::NumericTable * [outputResponsesCount];
    size_t * const commOutputResponsesNodeIDs = new size_t[outputResponsesCount];
    for (size_t i = 0; i < outputResponsesCount; ++i)
    {
        commOutputResponses[i] = static_cast<data_management::NumericTable *>(perNodeCommOutputResponses->getValueByIndex(i).get());
        commOutputResponsesNodeIDs[i] = perNodeCommOutputResponses->getKeyByIndex(i);
    }

    const daal::algorithms::Parameter * const par = _par;
    daal::services::Environment::env & env = *_env;

    const services::Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::KNNClassificationPredictDistrStep2Kernel,                               \
                                                         __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute,                             \
                                                         a.get(), interm.get(), key, round, m.get(),                                            \
                                                         respCount, commResponses, commResponsesNodeIDs,                                        \
                                                         inputQueryCount, commInputQueries, commInputQueriesNodeIDs,                            \
                                                         outputResponsesCount, commOutputResponses, commOutputResponsesNodeIDs,                 \
                                                         queryCount, commQueries, commQueriesNodeIDs, r.get(), par);

    delete[] commOutputResponsesNodeIDs;
    delete[] commOutputResponses;
    delete[] commQueriesNodeIDs;
    delete[] commQueries;
    delete[] commInputQueriesNodeIDs;
    delete[] commInputQueries;
    delete[] commResponsesNodeIDs;
    delete[] commResponses;

    return s;
}

template <typename algorithmFpType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Local, algorithmFpType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

} // namespace prediction
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal
