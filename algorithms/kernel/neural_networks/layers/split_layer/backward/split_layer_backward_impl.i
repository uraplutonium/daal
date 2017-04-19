/* file: split_layer_backward_impl.i */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of split algorithm
//--
*/

#ifndef __SPLIT_LAYER_BACKWARD_IMPL_I__
#define __SPLIT_LAYER_BACKWARD_IMPL_I__

#include "threading.h"

#include "mkl_tensor.h"

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace split
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status SplitKernel<algorithmFPType, method, cpu>::compute(Tensor *inputTensors[], Tensor *resultTensor, size_t nInputs)
{
    if (nInputs == 0) { return services::Status(); }

    MklTensor<algorithmFPType> *resultMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(resultTensor);
    MklTensor<algorithmFPType> *firstInputMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(inputTensors[0]);

    bool canUseMklTensor = (resultMklTensor != 0 && firstInputMklTensor != 0);

    for (size_t i = 1; i < nInputs && canUseMklTensor; i++)
    {
        MklTensor<algorithmFPType> *inputMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(inputTensors[i]);
        if (!inputMklTensor)
        {
            canUseMklTensor = false;
        }
        else if (!dnn::xLayoutCompare((dnnLayout_t)firstInputMklTensor->getDnnLayout(), (dnnLayout_t)inputMklTensor->getDnnLayout()))
        {
            canUseMklTensor = false;
        }
    }

    if (canUseMklTensor)
    {
        const services::Collection<size_t> &dims = inputTensors[0]->getDimensions();
        size_t dstChannelSize[1] = {dims[1]};
        dnnPrimitive_t splitPrim;
        dnnError_t err;
        dnnLayout_t inputLayout = (dnnLayout_t)firstInputMklTensor->getDnnLayout();
        err = dnn::xSplitCreate(&splitPrim, 1, inputLayout, dstChannelSize); ON_ERR(err);

        dnnLayout_t resultLayout;
        err = dnn::xLayoutCreateFromPrimitive(&resultLayout, splitPrim, dnnResourceSrc); ON_ERR(err);

        resultMklTensor->setDnnLayout(resultLayout);
        size_t nDataElements = firstInputMklTensor->getSize();
        algorithmFPType *resultArray = resultMklTensor->getDnnArray();
        algorithmFPType *firstInputArray = firstInputMklTensor->getDnnArray();
        for (size_t i = 0; i < nDataElements; i++)
        {
            resultArray[i] = firstInputArray[i];
        }
        for (size_t i = 1; i < nInputs; i++)
        {
            MklTensor<algorithmFPType> *inputMklTensor = dynamic_cast<MklTensor<algorithmFPType>*>(inputTensors[i]);
            algorithmFPType *inputArray = inputMklTensor->getDnnArray();
            for (size_t j = 0; j < nDataElements; j++)
            {
                resultArray[j] += inputArray[j];
            }
        }

        dnn::xDelete(splitPrim);
    }
    else
    {
        const services::Collection<size_t> &dims = inputTensors[0]->getDimensions();
        size_t nInputRows = dims[0];

        size_t nBlocks = nInputRows / _nRowsInBlock;
        size_t nRowsInLastBlock = nInputRows - nBlocks * _nRowsInBlock;

        for(size_t block = 0; block < nBlocks; block++)
        {
            processBlockInit(inputTensors[0], block * _nRowsInBlock, _nRowsInBlock, resultTensor);
        }
        if(nRowsInLastBlock > 0)
        {
            processBlockInit(inputTensors[0], nBlocks * _nRowsInBlock, nRowsInLastBlock, resultTensor);
        }

        for(int i = 1; i < nInputs; i++)
        {
            Tensor *inputTensor = inputTensors[i];

            for(size_t block = 0; block < nBlocks; block++)
            {
                processBlock(inputTensor, block * _nRowsInBlock, _nRowsInBlock, resultTensor);
            }
            if(nRowsInLastBlock > 0)
            {
                processBlock(inputTensor, nBlocks * _nRowsInBlock, nRowsInLastBlock, resultTensor);
            }
        }
    }
    DAAL_RETURN_STATUS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline void SplitKernel<algorithmFPType, method, cpu>::processBlock(Tensor *inputTensor,
                                                                    size_t nProcessedRows,
                                                                    size_t nRowsInCurrentBlock,
                                                                    Tensor *resultTensor)
{
    SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, readWrite, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    size_t nDataElements = inputBlock.getSize();
    for(size_t i = 0; i < nDataElements; i++)
    {
        resultArray[i] += inputArray[i];
    }

    inputTensor->releaseSubtensor(inputBlock);
    resultTensor->releaseSubtensor(resultBlock);
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline void SplitKernel<algorithmFPType, method, cpu>::processBlockInit(Tensor *inputTensor,
                                                                        size_t nProcessedRows,
                                                                        size_t nRowsInCurrentBlock,
                                                                        Tensor *resultTensor)
{
    SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTensor->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    size_t nDataElements = inputBlock.getSize();
    for(size_t i = 0; i < nDataElements; i++)
    {
        resultArray[i] = inputArray[i];
    }

    inputTensor->releaseSubtensor(inputBlock);
    resultTensor->releaseSubtensor(resultBlock);
}

} // namespace internal
} // namespace backward
} // namespace split
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
