/* file: kmeans_init_container_v1.h */
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
//  Implementation of initializing the K-Means algorithm container.
//--
*/

#ifndef __KMEANS_INIT_CONTAINER_V1_H__
#define __KMEANS_INIT_CONTAINER_V1_H__

#include "kmeans/inner/kmeans_init_batch_v1.h"
#include "kmeans/inner/kmeans_init_distributed_v1.h"
#include "kmeans_init_batch.h"
#include "kmeans_init_distributed.h"
#include "kmeans_init_kernel.h"
#include "kmeans_init_impl.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{

namespace interface1
{

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansInitKernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    const size_t na = 1;
    NumericTable *a[na];
    a[0] = static_cast<NumericTable *>(input->get(data).get());

    const size_t nr = 1;
    NumericTable *r[nr];
    r[0] = static_cast<NumericTable *>(result->get(centroids).get());

    interface1::Parameter *par = static_cast<interface1::Parameter *>(_par);
    interface2::Parameter internalPar(par->nClusters, par->offset, par->seed);

    internalPar.nRowsTotal = par->nRowsTotal;
    internalPar.oversamplingFactor = par->oversamplingFactor;
    internalPar.nRounds = par->nRounds;
    internalPar.engine = par->engine;
    internalPar.nTrials = 1;

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::KMeansInitKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, na, a, nr, r,
        &internalPar, *par->engine);
}


template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansInitStep1LocalKernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step1Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::compute()
{
    NumericTable* pData = static_cast<Input *>(_in)->get(data).get();
    PartialResult *pRes = static_cast<PartialResult *>(_pres);

    NumericTablePtr pPartialClusters = pRes->get(partialClusters);
    NumericTable* pNumPartialClusters = pRes->get(partialClustersNumber).get();
    interface1::Parameter *par = static_cast<interface1::Parameter *>(_par);

    interface2::Parameter internalPar(par->nClusters, par->offset, par->seed);

    internalPar.nRowsTotal = par->nRowsTotal;
    internalPar.oversamplingFactor = par->oversamplingFactor;
    internalPar.nRounds = par->nRounds;
    internalPar.engine = par->engine;
    internalPar.nTrials = 1;

    daal::services::Environment::env &env = *_env;
    services::Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::KMeansInitStep1LocalKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType),
        compute, pData, &internalPar, pNumPartialClusters, pPartialClusters, *par->engine);
    static_cast<PartialResult *>(_pres)->set(partialClusters, pPartialClusters); //can be null
    return s;
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step1Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansInitStep2MasterKernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::compute()
{
    return services::Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    DistributedStep2MasterInput *input = static_cast<DistributedStep2MasterInput *>(_in);
    Result *result = static_cast<Result *>(_res);
    data_management::DataCollection *dcInput = input->get(partialResults).get();

    size_t nPartials = dcInput->size();

    size_t na = nPartials * 2;
    NumericTable **a = new NumericTable*[na];
    for(size_t i = 0; i < nPartials; i++)
    {
        PartialResult *inPres = static_cast<PartialResult *>((*dcInput)[i].get());
        a[i * 2 + 0] = static_cast<NumericTable *>(inPres->get(partialClustersNumber).get());
        a[i * 2 + 1] = static_cast<NumericTable *>(inPres->get(partialClusters).get());
    }

    NumericTable* ntClusters = static_cast<NumericTable *>(result->get(centroids).get());

    interface1::Parameter *par = static_cast<interface1::Parameter *>(_par);

    interface2::Parameter internalPar(par->nClusters, par->offset, par->seed);

    internalPar.nRowsTotal = par->nRowsTotal;
    internalPar.oversamplingFactor = par->oversamplingFactor;
    internalPar.nRounds = par->nRounds;
    internalPar.engine = par->engine;
    internalPar.nTrials = 1;

    daal::services::Environment::env &env = *_env;

    services::Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::KMeansInitStep2MasterKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType),
        finalizeCompute, na, a, ntClusters, &internalPar);

    delete[] a;

    dcInput->clear();
    return s;
}

/////////////////////////////// init plusPlus/parallelPlus distributed containers ///////////////////////////////////////////////
template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansInitStep2LocalKernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step2Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Local, algorithmFPType, method, cpu>::compute()
{
    const interface1::DistributedStep2LocalPlusPlusParameter* par = (const interface1::DistributedStep2LocalPlusPlusParameter*)(_par);
    DistributedStep2LocalPlusPlusInput *input = static_cast<DistributedStep2LocalPlusPlusInput *>(_in);
    const NumericTable* pData = input->get(data).get();
    const NumericTable* pNewCenters = input->get(inputOfStep2).get();
    DistributedStep2LocalPlusPlusPartialResult *pPartRes = static_cast<DistributedStep2LocalPlusPlusPartialResult *>(_pres);
    NumericTable* pRes = pPartRes->get(outputOfStep2ForStep3).get();
    DataCollectionPtr pLocalData = (par->firstIteration ? pPartRes->get(internalResult) : input->get(internalInput));
    NumericTable* aLocalData[internal::localDataSize] = { 0, 0, 0, 0 };
    for(size_t i = 0; i < pLocalData->size(); ++i)
        aLocalData[i] = NumericTable::cast((*pLocalData)[i]).get();

    NumericTable* pOutputForStep5 = (isParallelPlusMethod(method) && par->outputForStep5Required ?
        pPartRes->get(outputOfStep2ForStep5).get() : nullptr);

    __DAAL_CALL_KERNEL(env, internal::KMeansInitStep2LocalKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute,
        par, pData, pNewCenters, aLocalData, pRes, pOutputForStep5);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step2Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step3Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansInitStep3MasterKernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step3Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step3Master, algorithmFPType, method, cpu>::compute()
{
    DistributedStep3MasterPlusPlusInput* input = static_cast<DistributedStep3MasterPlusPlusInput*>(_in);
    DistributedStep3MasterPlusPlusPartialResult* pr = static_cast<DistributedStep3MasterPlusPlusPartialResult *>(_pres);
    data_management::MemoryBlock* pRngState = dynamic_cast<data_management::MemoryBlock*>(pr->get(rngState).get());
    const interface1::Parameter *par = (const interface1::Parameter*)(_par);

    interface2::Parameter internalPar(par->nClusters, par->offset, par->seed);

    internalPar.nRowsTotal = par->nRowsTotal;
    internalPar.oversamplingFactor = par->oversamplingFactor;
    internalPar.nRounds = par->nRounds;
    internalPar.engine = par->engine;
    internalPar.nTrials = 1;

    __DAAL_CALL_KERNEL(env, internal::KMeansInitStep3MasterKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute,
        &internalPar, input->get(inputOfStep3FromStep2).get(), pRngState, pr->get(outputOfStep3ForStep4).get(), *par->engine);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step3Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step4Local, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansInitStep4LocalKernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step4Local, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step4Local, algorithmFPType, method, cpu>::compute()
{
    const interface1::Parameter* par = (const interface1::Parameter*)(_par);
    DistributedStep4LocalPlusPlusInput *input = static_cast<DistributedStep4LocalPlusPlusInput *>(_in);
    const NumericTable* pData = input->get(data).get();
    const NumericTable* pInput = input->get(inputOfStep4FromStep3).get();
    DistributedStep4LocalPlusPlusPartialResult *pPartRes = static_cast<DistributedStep4LocalPlusPlusPartialResult *>(_pres);
    NumericTable* pOutput = pPartRes->get(outputOfStep4).get();
    DataCollectionPtr pLocalData = input->get(internalInput);
    NumericTable* aLocalData[internal::localDataSize] = { 0, 0, 0, 0 };
    for(size_t i = 0; i < pLocalData->size(); ++i)
        aLocalData[i] = NumericTable::cast((*pLocalData)[i]).get();

    __DAAL_CALL_KERNEL(env, internal::KMeansInitStep4LocalKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute,
        pData, pInput, aLocalData, pOutput);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step4Local, algorithmFPType, method, cpu>::finalizeCompute()
{
    return services::Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step5Master, algorithmFPType, method, cpu>::DistributedContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::KMeansInitStep5MasterKernel, method, algorithmFPType);
}

template<typename algorithmFPType, Method method, CpuType cpu>
DistributedContainer<step5Master, algorithmFPType, method, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step5Master, algorithmFPType, method, cpu>::compute()
{
    DistributedStep5MasterPlusPlusInput *input = static_cast<DistributedStep5MasterPlusPlusInput*>(_in);
    const DataCollection* pCandidates = input->get(inputCentroids).get();
    const DataCollection* pRating = input->get(inputOfStep5FromStep2).get();
    DistributedStep5MasterPlusPlusPartialResult *pPartRes = static_cast<DistributedStep5MasterPlusPlusPartialResult *>(_pres);
    NumericTable* pResCand = pPartRes->get(candidates).get();
    NumericTable* pResWeights = pPartRes->get(weights).get();
    __DAAL_CALL_KERNEL(env, internal::KMeansInitStep5MasterKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute,
        pCandidates, pRating, pResCand, pResWeights);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistributedContainer<step5Master, algorithmFPType, method, cpu>::finalizeCompute()
{
    const interface1::Parameter* par = (const interface1::Parameter*)(_par);
    DistributedStep5MasterPlusPlusPartialResult *pPartRes = static_cast<DistributedStep5MasterPlusPlusPartialResult *>(_pres);
    DistributedStep5MasterPlusPlusInput *input = static_cast<DistributedStep5MasterPlusPlusInput*>(_in);
    NumericTable* ntCandidates = pPartRes->get(candidates).get();
    NumericTable* ntWeights = pPartRes->get(weights).get();
    data_management::MemoryBlock* pRngState = dynamic_cast<data_management::MemoryBlock*>(input->get(inputOfStep5FromStep3).get());
    Result *pRes = static_cast<Result *>(_res);

    interface2::Parameter internalPar(par->nClusters, par->offset, par->seed);

    internalPar.nRowsTotal = par->nRowsTotal;
    internalPar.oversamplingFactor = par->oversamplingFactor;
    internalPar.nRounds = par->nRounds;
    internalPar.engine = par->engine;
    internalPar.nTrials = 1;

    __DAAL_CALL_KERNEL(env, internal::KMeansInitStep5MasterKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), finalizeCompute,
        &internalPar, ntCandidates, ntWeights, pRngState, pRes->get(centroids).get(), *par->engine);
}


} // interface1

} // namespace daal::algorithms::kmeans::init
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal

#endif
