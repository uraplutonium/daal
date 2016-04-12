/* file: average_pooling2d_layer_forward_impl.i */
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
//  Implementation of forward average pooling layer
//--
*/

#ifndef __AVERAGE_POOLING2D_LAYER_FORWARD_IMPL_I__
#define __AVERAGE_POOLING2D_LAYER_FORWARD_IMPL_I__

#include "service_utils.h"
#include "service_memory.h"
#include "service_data_utils.h"
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
namespace forward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void PoolingKernel<algorithmFPType, method, cpu>::compute(
    const average_pooling2d::forward::Input *input, const average_pooling2d::Parameter *parameter,
    average_pooling2d::forward::Result *result)
{
    const algorithmFPType zero = 0.0;
    const algorithmFPType one = 1.0;

    SharedPtr<Tensor> dataTensor = input->get(layers::forward::data);
    SharedPtr<Tensor> valueTensor = result->get(layers::forward::value);

    const Collection<size_t> &dims = dataTensor->getDimensions();
    const Collection<size_t> &valueDims = valueTensor->getDimensions();
    SubtensorDescriptor<algorithmFPType> dataBlock, valueBlock;
    dataTensor->getSubtensor(0, 0, 0, dims[0], readOnly, dataBlock);
    valueTensor->getSubtensor(0, 0, 0, valueDims[0], writeOnly, valueBlock);

    algorithmFPType *data = dataBlock.getPtr();
    algorithmFPType *value = valueBlock.getPtr();

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

    SharedPtr<NumericTable> inputDimensionsTable = result->get(auxInputDimensions);
    data_management::BlockDescriptor<int> block;
    inputDimensionsTable->getBlockOfRows(0, 1, data_management::writeOnly, block);
    int *inputDimensionsInt = block.getBlockPtr();
    for(size_t i = 0; i < inputDimensionsTable->getNumberOfColumns(); i++)
    {
        inputDimensionsInt[i] = (int)dims[i];
    }
    inputDimensionsTable->releaseBlockOfRows(block);
    /*
     * Input data tensor is viewed by this method as a 5-dimensional tensor of size:
     * offsetBefore * firstSize * offsetBetween * secondSize * offsetAfter
     */
    size_t offsetBefore = 1;
    for (size_t i = 0; i < firstIndex; i++)
    {
        offsetBefore *= dims[i];
    }
    size_t firstSize = dims[firstIndex];
    size_t firstOutSize = valueDims[firstIndex];
    size_t offsetBetween = 1;
    for (size_t i = firstIndex + 1; i < secondIndex; i++)
    {
        offsetBetween *= dims[i];
    }
    size_t secondSize = dims[secondIndex];
    size_t secondOutSize = valueDims[secondIndex];
    size_t offsetAfter = 1;
    for (size_t i = secondIndex + 1; i < dims.size(); i++)
    {
        offsetAfter *= dims[i];
    }

    algorithmFPType divisor = 1.0 / (firstKernelSize * secondKernelSize);
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
                         * Resulting value index
                         */
                        MKL_INT valueIndex = j + offsetAfter * (so + secondOutSize * (k + offsetBetween * (fo + firstOutSize * i)));

                        algorithmFPType average = zero;

                        /*
                         * Loops over the kernel
                         */
                        for (MKL_INT fi = f; fi < f + firstKernelSize; fi++)
                        {
                            for (MKL_INT si = s; si < s + secondKernelSize; si++)
                            {
                                MKL_INT dataIndex = j + offsetAfter * (si + secondSize * (k + offsetBetween * (fi + firstSize * i)));
                                bool paddingFlag = ((fi < 0) || (fi >= (MKL_INT)firstSize) || (si < 0) || (si >= (MKL_INT)secondSize));
                                algorithmFPType dataValue = (paddingFlag ? zero : data[dataIndex]);

                                average += dataValue;
                            }
                        }
                        value[valueIndex] = average * divisor;
                    }
                }
            }
        }
    }
    dataTensor->releaseSubtensor(dataBlock);
    valueTensor->releaseSubtensor(valueBlock);
}

} // namespace internal
} // namespace forward
} // namespace average_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
