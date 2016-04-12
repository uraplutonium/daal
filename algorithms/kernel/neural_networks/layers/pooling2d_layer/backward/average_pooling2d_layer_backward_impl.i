/* file: average_pooling2d_layer_backward_impl.i */
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

#ifndef __AVERAGE_POOLING2D_LAYER_BACKWARD_IMPL_I__
#define __AVERAGE_POOLING2D_LAYER_BACKWARD_IMPL_I__

#include "service_utils.h"
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
namespace average_pooling2d
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::compute(
    const average_pooling2d::backward::Input *input, const average_pooling2d::Parameter *parameter,
    average_pooling2d::backward::Result *result)
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

    size_t firstIndex    = parameter->indices.size[0];
    size_t secondIndex   = parameter->indices.size[1];
    size_t firstPadding  = parameter->padding.size[0];
    size_t secondPadding = parameter->padding.size[1];
    size_t firstStride   = parameter->stride.size[0];
    size_t secondStride  = parameter->stride.size[1];
    MKL_INT firstKernelSize  = (MKL_INT)(parameter->kernelSize.size[0]);
    MKL_INT secondKernelSize = (MKL_INT)(parameter->kernelSize.size[1]);

    if (firstIndex > secondIndex)
    {
        swap<size_t, cpu>(firstIndex,   secondIndex);
        swap<size_t, cpu>(firstPadding, secondPadding);
        swap<size_t, cpu>(firstStride,  secondStride);
        swap<MKL_INT, cpu>(firstKernelSize, secondKernelSize);
    }

    /*
     * Tensor is viewed by this method as a 5-dimensional tensor of size:
     * offsetBefore * firstSize * offsetBetween * secondSize * offsetAfter
     */
    size_t offsetBefore = 1;
    for (size_t i = 0; i < firstIndex; i++)
    {
        offsetBefore *= gradDims[i];
    }
    size_t firstSize = gradDims[firstIndex];
    size_t firstOutSize = inputDims[firstIndex];
    size_t offsetBetween = 1;
    for (size_t i = firstIndex + 1; i < secondIndex; i++)
    {
        offsetBetween *= gradDims[i];
    }
    size_t secondSize = gradDims[secondIndex];
    size_t secondOutSize = inputDims[secondIndex];
    size_t offsetAfter = 1;
    for (size_t i = secondIndex + 1; i < gradDims.size(); i++)
    {
        offsetAfter *= gradDims[i];
    }

    const algorithmFPType one = 1.0;
    algorithmFPType gradMultiplier = one / (algorithmFPType)(firstKernelSize * secondKernelSize);
    for (MKL_INT i = 0; i < (MKL_INT)offsetBefore; i++)
    {
        /*
         * Loop by the first kernel dimension
         * f - index of the left upper corner of the kernel
         * fo - index of the output value
         */
        for (MKL_INT f = -firstPadding, fo = 0; fo < (MKL_INT)firstOutSize; f += firstStride, fo++)
        {
            for (MKL_INT k = 0; k < (MKL_INT)offsetBetween; k++)
            {
                /*
                 * Loop by the second kernel dimension
                 * s - index of the left upper corner of the kernel
                 * so - index of the output value
                 */
                for (MKL_INT s = -secondPadding, so = 0; so < (MKL_INT)secondOutSize; s += secondStride, so++)
                {
                    for (MKL_INT j = 0; j < (MKL_INT)offsetAfter; j++)
                    {
                        /*
                         * Input value index
                         */
                        MKL_INT inputIndex = j + offsetAfter * (so + secondOutSize * (k + offsetBetween * (fo + firstOutSize * i)));
                        algorithmFPType inputValue = gradMultiplier * inputGrad[inputIndex];

                        /*
                         * Loops over the kernel
                         */
                        for (MKL_INT fi = f; fi < f + firstKernelSize; fi++)
                        {
                            for (MKL_INT si = s; si < s + secondKernelSize; si++)
                            {
                                MKL_INT gradIndex = j + offsetAfter * (si + secondSize * (k + offsetBetween * (fi + firstSize * i)));
                                bool paddingFlag = ((fi < 0) || (fi >= (MKL_INT)firstSize) || (si < 0) || (si >= (MKL_INT)secondSize));

                                if (!paddingFlag)
                                {
                                    grad[gradIndex] += inputValue;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    inputTensor->releaseSubtensor(inputBlock);
    gradTensor->releaseSubtensor(gradBlock);
}
} // namespace internal
} // namespace backward
} // namespace average_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
