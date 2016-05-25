/* file: average_pooling3d_layer_backward_impl.i */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of backward average pooling layer
//--
*/

#ifndef __AVERAGE_POOLING3D_LAYER_BACKWARD_IMPL_I__
#define __AVERAGE_POOLING3D_LAYER_BACKWARD_IMPL_I__

#include "service_sort.h"
#include "service_memory.h"
#include "service_blas.h"

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace average_pooling3d
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::compute(
    const average_pooling3d::backward::Input *input, const average_pooling3d::Parameter *parameter,
    average_pooling3d::backward::Result *result)
{
    SharedPtr<Tensor> inputTensor = input->get(layers::backward::inputGradient);
    SharedPtr<Tensor> gradTensor = result->get(layers::backward::gradient);

    const Collection<size_t> &inputDims = inputTensor->getDimensions();
    const Collection<size_t> &gradDims = gradTensor->getDimensions();
    SubtensorDescriptor<algorithmFPType> inputBlock, maskBlock, gradBlock;

    inputTensor->getSubtensor(0, 0, 0, inputDims[0], readOnly, inputBlock);
    gradTensor->getSubtensor(0, 0, 0, gradDims[0], writeOnly, gradBlock);

    algorithmFPType *inputGrad = inputBlock.getPtr();
    algorithmFPType *grad = gradBlock.getPtr();

    algorithmFPType zero = 0.0;
    daal::services::internal::service_memset<algorithmFPType, cpu>(grad, zero, gradBlock.getSize());

    /* Get indices from the parameters of the layer */
    size_t indices[nKernelDims];
    size_t indicesOrder[nKernelDims];
    for (size_t i = 0; i < nKernelDims; i++)
    {
        indices[i] = parameter->indices.size[i];
        indicesOrder[i] = i;
    }

    /* Arrange indices in the ascending order and re-order other parameters respectively */
    daal::algorithms::internal::qSort<size_t, size_t, cpu>(nKernelDims, indices, indicesOrder);

    MKL_INT padding[nKernelDims], stride[nKernelDims], kernelSize[nKernelDims];
    for (size_t i = 0; i < nKernelDims; i++)
    {
        padding[i]    = (MKL_INT)(parameter->padding   .size[indicesOrder[i]]);
        stride[i]     = (MKL_INT)(parameter->stride    .size[indicesOrder[i]]);
        kernelSize[i] = (MKL_INT)(parameter->kernelSize.size[indicesOrder[i]]);
    }

    /*
     * Resulting gradient tensor is viewed by this method as a 7-dimensional tensor of size:
     * offset[0] * gradSize[0] * offset[1] * gradSize[1] * offset[2] * gradSize[2] * offset[3]
     */
    MKL_INT offset[nKernelDims + 1];
    MKL_INT gradSize[nKernelDims];      // size of the resulting 3D subtensor over which the kernel is applied
    MKL_INT inputSize[nKernelDims];     // size of the input 3D subtensor over which the kernel is applied
    for (size_t d = 0; d < nKernelDims; d++)
    {
        gradSize[d]  = (MKL_INT)gradDims[indices[d]];
        inputSize[d] = (MKL_INT)inputDims[indices[d]];
    }
    offset[0] = 1;
    for (size_t i = 0; i < indices[0]; i++)
    {
        offset[0] *= (MKL_INT)gradDims[i];
    }
    for (size_t d = 1; d < nKernelDims; d++)
    {
        offset[d] = 1;
        for (size_t i = indices[d-1] + 1; i < indices[d]; i++)
        {
            offset[d] *= (MKL_INT)gradDims[i];
        }
    }
    offset[nKernelDims] = 1;
    for (size_t i = indices[nKernelDims-1] + 1; i < gradDims.size(); i++)
    {
        offset[nKernelDims] *= (MKL_INT)gradDims[i];
    }

    const algorithmFPType one = 1.0;
    algorithmFPType gradMultiplier = 1.0;
    for (size_t d = 0; d < nKernelDims; d++)
    {
        gradMultiplier *= (algorithmFPType)kernelSize[d];
    }
    gradMultiplier = 1.0 / gradMultiplier;

    MKL_INT ii[nKernelDims + 1];    // index of the input data
    MKL_INT ik[nKernelDims];        // index of the kernel
    MKL_INT iv[nKernelDims];        // index of the value
    MKL_INT inputOffset[nKernelDims + 1];
    MKL_INT gradOffset[nKernelDims + 1];

    for (ii[0] = 0; ii[0] < (MKL_INT)offset[0]; ii[0]++)
    {
        inputOffset[0] = ii[0];
        gradOffset[0]  = ii[0];
        recurrentCompute(0, ii, ik, iv, padding, stride, kernelSize, gradSize, inputSize,
            offset, gradOffset, inputOffset, inputGrad, grad, gradMultiplier);
    }

    inputTensor->releaseSubtensor(inputBlock);
    gradTensor->releaseSubtensor(gradBlock);
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::recurrentCompute(size_t d,
    MKL_INT *ii, MKL_INT *ik, MKL_INT *iv, const MKL_INT *padding, const MKL_INT *stride, const MKL_INT *kernelSize,
    const MKL_INT* gradSize, const MKL_INT* inputSize, const MKL_INT* offset, MKL_INT* gradOffset, MKL_INT* inputOffset,
    const algorithmFPType *inputGrad, algorithmFPType *grad, algorithmFPType gradMultiplier)
{
    const algorithmFPType zero = 0.0;
    if (d < nKernelDims)
    {
        /*
         * Loops over the d-th kernel dimension
         */
        for (ik[d] = -padding[d], iv[d] = 0; iv[d] < inputSize[d]; ik[d] += stride[d], iv[d]++)
        {
            inputOffset[d+1] = offset[d+1] * (iv[d] + inputSize[d] * inputOffset[d]);
            gradOffset[d+1] = offset[d+1] * (ik[d] + gradSize[d] * gradOffset[d]);
            for (ii[d+1] = 0; ii[d+1] < offset[d+1]; ii[d+1]++)
            {
                recurrentCompute(d + 1, ii, ik, iv, padding, stride, kernelSize, gradSize, inputSize,
                    offset, gradOffset, inputOffset, inputGrad, grad, gradMultiplier);
            }
        }
    }
    else
    {
        /*
         * Input gradient index
         */
        MKL_INT inputIndex = ii[nKernelDims] + inputOffset[nKernelDims];
        algorithmFPType inputValue = gradMultiplier * inputGrad[inputIndex];

        MKL_INT iwk[nKernelDims];              // index of the gradient within kernel
        MKL_INT iwkShifted[nKernelDims];
        MKL_INT gradKernelOffset[nKernelDims];
        bool paddingFlags[nKernelDims];
        /*
         * Loops over the kernel
         */
        for (iwk[0] = 0, iwkShifted[0] = ik[0]; iwk[0] < kernelSize[0]; iwk[0]++, iwkShifted[0]++)
        {
            paddingFlags[0] = (iwkShifted[0] < 0) || (iwkShifted[0] >= gradSize[0]);
            gradKernelOffset[0] = offset[1] * iwk[0];
            for (iwk[1] = 0, iwkShifted[1] = ik[1]; iwk[1] < kernelSize[1]; iwk[1]++, iwkShifted[1]++)
            {
                paddingFlags[1] = (iwkShifted[1] < 0) || (iwkShifted[1] >= gradSize[1]);
                gradKernelOffset[1] = offset[2] * (iwk[1] + gradSize[1] * gradKernelOffset[0]);
                for (iwk[2] = 0, iwkShifted[2] = ik[2]; iwk[2] < kernelSize[2]; iwk[2]++, iwkShifted[2]++)
                {
                    paddingFlags[2] = (iwkShifted[2] < 0) || (iwkShifted[2] >= gradSize[2]);
                    gradKernelOffset[2] = offset[3] * (iwk[2] + gradSize[2] * gradKernelOffset[1]);
                    MKL_INT gradIndex = ii[3] + gradOffset[3] + gradKernelOffset[2];
                    bool paddingFlag = paddingFlags[0] || paddingFlags[1] || paddingFlags[2];

                    if (!paddingFlag)
                    {
                        grad[gradIndex] += inputValue;
                    }
                }
            }
        }
    }
}

} // namespace internal
} // namespace backward
} // namespace average_pooling3d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
