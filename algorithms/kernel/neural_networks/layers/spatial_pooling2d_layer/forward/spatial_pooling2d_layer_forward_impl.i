/* file: spatial_pooling2d_layer_forward_impl.i */
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
//  Implementation of forward pooling layer
//--
*/

#ifndef __SPATIAL_POOLING2D_LAYER_FORWARD_IMPL_I__
#define __SPATIAL_POOLING2D_LAYER_FORWARD_IMPL_I__

#include "service_memory.h"
#include "service_blas.h"
#include "service_tensor.h"
#include "service_numeric_table.h"

#include "tensor.h"
#include "homogen_numeric_table.h"
#include "maximum_pooling2d_layer_forward.h"
#include "stochastic_pooling2d_layer_forward.h"
#include "average_pooling2d_layer_forward.h"

#include "average_pooling2d_layer_forward_kernel.h"
#include "stochastic_pooling2d_layer_forward_kernel.h"
#include "maximum_pooling2d_layer_forward_kernel.h"

using namespace daal::services;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace spatial_pooling2d
{
namespace forward
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu>
services::Status BasePoolingKernel<algorithmFPType, cpu>::compute(const Tensor &dataTensor, Tensor &valueTensor,
        Tensor *selectedPosTensor, const spatial_pooling2d::Parameter &spatialParameter)
{
    Status s;
    const size_t L = spatialParameter.pyramidHeight;

    size_t nDims = 4;
    Collection<size_t> extractLayoutCollection(nDims);
    for(size_t i = 0; i < nDims; i++)
    {
        extractLayoutCollection[i] = i;
    }

    daal::services::internal::swap<cpu, size_t>(extractLayoutCollection[spatialParameter.indices.size[0]], extractLayoutCollection[nDims - 2]);
    daal::services::internal::swap<cpu, size_t>(extractLayoutCollection[spatialParameter.indices.size[1]], extractLayoutCollection[nDims - 1]);

    TensorOffsetLayout targetInLayout = dataTensor.createDefaultSubtensorLayout();
    targetInLayout.shuffleDimensions(extractLayoutCollection);

    const Collection<size_t> &dims(targetInLayout.getDimensions());
    const Collection<size_t> &valueDims = valueTensor.getDimensions();

    ReadSubtensor<algorithmFPType, cpu, Tensor> dataSubtensor;
    WriteOnlySubtensor<algorithmFPType, cpu, Tensor> valueSubtensor;
    WriteOnlySubtensor<int, cpu, Tensor> selectedPosSubtensor;

    size_t poolDimSize1 = dims[2];
    size_t poolDimSize2 = dims[3];

    Collection<size_t> dataSliceDims(dims);
    dataSliceDims[0] = 1;

    pooling2d::forward::Result dummyResult;
    for(size_t slice = 0; slice < dims[0]; slice++)
    {
        dataSubtensor.set(const_cast<Tensor &>(dataTensor), 0, 0, slice, 1, targetInLayout);
        DAAL_CHECK_BLOCK_STATUS(dataSubtensor);
        const algorithmFPType *dataSlice = dataSubtensor.get();
        SharedPtr<HomogenTensor<algorithmFPType> > dataSliceTensor = HomogenTensor<algorithmFPType>::create(dataSliceDims, const_cast<algorithmFPType *>(dataSlice),&s);
        DAAL_CHECK_STATUS_VAR(s);

        valueSubtensor.set(valueTensor, 0, 0, slice, 1);
        DAAL_CHECK_BLOCK_STATUS(valueSubtensor);
        algorithmFPType *valueSlice = valueSubtensor.get();

        int *selectedPosSlice = nullptr;
        if(selectedPosTensor)
        {
            selectedPosSubtensor.set(*selectedPosTensor, 0, 0, slice, 1);
            DAAL_CHECK_BLOCK_STATUS(selectedPosSubtensor);
            selectedPosSlice = selectedPosSubtensor.get();
        }

        size_t accumulatedFlattenOffset = 0;
        for(size_t level = 0, pow2 = 1; level < L ; level++, pow2 *= 2)
        {
            const size_t kernelSize1 = (poolDimSize1 % pow2 == 0) ? poolDimSize1 / pow2 : poolDimSize1 / pow2 + 1;
            const size_t kernelSize2 = (poolDimSize2 % pow2 == 0) ? poolDimSize2 / pow2 : poolDimSize2 / pow2 + 1;
            const size_t stride1 = kernelSize1;
            const size_t stride2 = kernelSize2;
            const size_t padding1 = (kernelSize1 * pow2 - poolDimSize1) / 2;
            const size_t padding2 = (kernelSize2 * pow2 - poolDimSize2) / 2;

            const pooling2d::Parameter poolingPar(2, 3, kernelSize1, kernelSize2, stride1, stride2, padding1, padding2);

            const Collection<size_t> &valueDims = dummyResult.getValueSize(dataSliceTensor->getDimensions(), &poolingPar, 0);
            algorithmFPType *value = &valueSlice[accumulatedFlattenOffset];
            SharedPtr<HomogenTensor<algorithmFPType> > poolingValueTensor = HomogenTensor<algorithmFPType>::create(valueDims, value, &s);
            DAAL_CHECK_STATUS_VAR(s);

            TensorPtr poolingSelectedPosTensor;
            if(selectedPosTensor)
            {
                int *selectedPos = &selectedPosSlice[accumulatedFlattenOffset];
                poolingSelectedPosTensor = HomogenTensor<int>::create(valueDims, selectedPos, &s);
            }

            DAAL_CHECK_STATUS(s, computePooling(poolingPar, spatialParameter, *dataSliceTensor, *poolingValueTensor, poolingSelectedPosTensor.get()));
            accumulatedFlattenOffset += dims[1] * pow2 * pow2;
        }
    }
    return s;
}

template<typename algorithmFPType, spatial_pooling2d::internal::Method method, CpuType cpu>
Status PoolingKernel<algorithmFPType, method, cpu>::computePooling(
    const pooling2d::Parameter &poolingPar,
    const spatial_pooling2d::Parameter &spatialParameter,
    const Tensor &dataTensor,
    Tensor &poolingValueTensor,
    Tensor *poolingSelectedPosTensor)
{
    return Status();
}


template<typename algorithmFPType, CpuType cpu>
Status PoolingKernel<algorithmFPType, spatial_pooling2d::internal::maximum, cpu>::computePooling(
    const pooling2d::Parameter &poolingPar,
    const spatial_pooling2d::Parameter &spatialParameter,
    const Tensor &dataTensor,
    Tensor &poolingValueTensor,
    Tensor *poolingSelectedPosTensor)
{
    const size_t nInputDims = dataTensor.getNumberOfDimensions();
    maximum_pooling2d::Parameter forwardMaximumParameter(nInputDims - 2, nInputDims - 1);
    forwardMaximumParameter.predictionStage = spatialParameter.predictionStage;
    spatial_pooling2d::internal::CommonSpatialPoolingFunctions<cpu>::setParameter(poolingPar, forwardMaximumParameter);
    maximum_pooling2d::forward::internal::PoolingKernel<algorithmFPType, maximum_pooling2d::defaultDense, cpu> poolKernel;
    return poolKernel.compute(dataTensor, poolingValueTensor, poolingSelectedPosTensor, forwardMaximumParameter);
}

template<typename algorithmFPType, CpuType cpu>
Status PoolingKernel<algorithmFPType, spatial_pooling2d::internal::stochastic, cpu>::computePooling(
    const pooling2d::Parameter &poolingPar,
    const spatial_pooling2d::Parameter &spatialParameter,
    const Tensor &dataTensor,
    Tensor &poolingValueTensor,
    Tensor *poolingSelectedPosTensor)
{
    const size_t nInputDims = dataTensor.getNumberOfDimensions();
    stochastic_pooling2d::Parameter forwardStochasticParameter(nInputDims - 2, nInputDims - 1);
    forwardStochasticParameter.predictionStage = spatialParameter.predictionStage;
    auto engine = ((const spatial_stochastic_pooling2d::Parameter &)spatialParameter).engine;
    spatial_pooling2d::internal::CommonSpatialPoolingFunctions<cpu>::setParameter(poolingPar, forwardStochasticParameter);
    stochastic_pooling2d::forward::internal::PoolingKernel<algorithmFPType, stochastic_pooling2d::defaultDense, cpu> poolKernel;
    return poolKernel.compute(dataTensor, poolingValueTensor, poolingSelectedPosTensor, forwardStochasticParameter, *engine);
}

template<typename algorithmFPType, CpuType cpu>
Status PoolingKernel<algorithmFPType, spatial_pooling2d::internal::average, cpu>::computePooling(
    const pooling2d::Parameter &poolingPar,
    const spatial_pooling2d::Parameter &spatialParameter,
    const Tensor &dataTensor,
    Tensor &poolingValueTensor,
    Tensor *poolingSelectedPosTensor)
{
    const size_t nInputDims = dataTensor.getNumberOfDimensions();
    average_pooling2d::Parameter forwardAverageParameter(nInputDims - 2, nInputDims - 1);
    forwardAverageParameter.predictionStage = spatialParameter.predictionStage;
    spatial_pooling2d::internal::CommonSpatialPoolingFunctions<cpu>::setParameter(poolingPar, forwardAverageParameter);
    average_pooling2d::forward::internal::PoolingKernel<algorithmFPType, average_pooling2d::defaultDense, cpu> poolKernel;
    return poolKernel.compute(dataTensor, forwardAverageParameter, poolingValueTensor);
}

} // namespace internal
} // namespace forward
} // namespace spatial_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
