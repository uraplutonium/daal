/* file: prelu_layer_forward_impl.i */
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
//  Implementation of prelu calculation functions.
//--
*/

#ifndef __PRELU_LAYER_FORWARD_IMPL_I__
#define __PRELU_LAYER_FORWARD_IMPL_I__

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace prelu
{
namespace forward
{
namespace internal
{

using namespace daal::services;

template<typename algorithmFPType, Method method, CpuType cpu>
Status PReLUKernel<algorithmFPType, method, cpu>::compute(const Tensor &inputTensor, const Tensor &wTensor, Tensor &resultTensor, const prelu::Parameter &parameter)
{
    Status s;

    size_t wStart  = parameter.dataDimension;
    size_t wLen    = parameter.weightsDimension;
    size_t wSize   = wTensor.getSize();
    size_t fDimN   = 0;
    size_t wOffset = 1;
    size_t wEndDim = wStart + wLen;

    const Collection<size_t> &inDims = inputTensor.getDimensions();
    const Collection<size_t> &wDims  = wTensor.getDimensions();

    ReadSubtensor<algorithmFPType, cpu, Tensor> wBlock(const_cast<Tensor &>(wTensor), 0, 0, 0, wTensor.getDimensions()[0]);
    DAAL_CHECK_BLOCK_STATUS(wBlock);
    const algorithmFPType *wArray = wBlock.get();

    TensorOffsetLayout inputLayout = inputTensor.createDefaultSubtensorLayout();

    Collection<size_t> wOffsets(wLen);
    wOffsets[wLen - 1] = 1;
    for(size_t i = 1; i < wLen; i++)
    {
        wOffsets[wLen - i - 1] = wOffsets[wLen - i] * wDims[wLen - i];
    }

    getNumberOfFixedDimensions(inputLayout, inDims, wEndDim, fDimN, wOffset, _nElemsInBlock);

    if(fDimN == 0)
    {
        s = processBlock(inputTensor, resultTensor, wArray, fDimN, 0, inputLayout, wSize, wOffset, wStart, wLen, inDims, wOffsets);
    }
    else
    {
        size_t nBlocks = inputTensor.getSize(0, fDimN);

        __DAAL_MAKE_TENSOR_THREADSAFE(const_cast<Tensor *>(&inputTensor))
        __DAAL_MAKE_TENSOR_THREADSAFE(&resultTensor)

        SafeStatus safeStat;

        daal::threader_for(nBlocks, nBlocks, [ =, &safeStat, &inputTensor, &resultTensor ](size_t i)
        {
            TArray<size_t, cpu> fdimsBlock(fDimN);
            size_t *fDims = fdimsBlock.get();

            if(!fDims) { safeStat |= Status(ErrorMemoryAllocationFailed); return; }

            getFixedDimsIndexes(fDimN, fDims, inDims, i);
            safeStat |= processBlock(inputTensor, resultTensor, wArray, fDimN, fDims, inputLayout, wSize, wOffset, wStart, wLen, inDims, wOffsets);
        } );

        s = safeStat.detach();
    }
    return s;
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status PReLUKernel<algorithmFPType, method, cpu>::processBlock(const Tensor &inputTensor, Tensor &resultTensor, const algorithmFPType *wArray, size_t fDimN,
                                                             size_t *fDims, const TensorOffsetLayout &layout, size_t wSize,
                                                             size_t wOffset, size_t wStart, size_t wLen, const Collection<size_t> &inDims,
                                                             const Collection<size_t> &wOffsets)
{
    ReadSubtensor<algorithmFPType, cpu, Tensor> inputBlock(const_cast<Tensor &>(inputTensor), fDimN, fDims, 0, inDims[fDimN], layout);
    DAAL_CHECK_BLOCK_STATUS(inputBlock);
    const algorithmFPType *inputArray = inputBlock.get();

    WriteSubtensor<algorithmFPType, cpu, Tensor> resultBlock(resultTensor, fDimN, fDims, 0, inDims[fDimN], layout);
    DAAL_CHECK_BLOCK_STATUS(resultBlock);
    algorithmFPType *resultArray = resultBlock.get();

    size_t nDataElements = inputBlock.getSize();
    size_t start = wStart;
    size_t end   = wStart;

    if(wStart + wLen <= fDimN) // weights are at the left of split dim
    {
        end += wLen;
    }
    if(wStart < fDimN && wStart + wLen > fDimN) // split dim is in the midddle of weights dims
    {
        end = fDimN;
    }

    size_t wJ = 0;
    for(size_t j = start; j < end; j++)
    {
        wJ += fDims[j] * wOffsets[j - wStart];
    }

    for(size_t i = 0; i < nDataElements; i++)
    {
        if(wStart + wLen > fDimN)
        {
            wJ += (i != 0 && i % wOffset == 0);
            if(wJ == wSize)
            {
                wJ = 0;
            }
        }

        if (inputArray[i] >= (algorithmFPType)0)
        {
            resultArray[i] = inputArray[i];
        }
        else
        {
            resultArray[i] = inputArray[i] * wArray[wJ];
        }
    }
    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void PReLUKernel<algorithmFPType, method, cpu>::getNumberOfFixedDimensions(TensorOffsetLayout &inputLayout, const Collection<size_t> &dims, size_t wEndDim,
                                                                           size_t &fDimN, size_t &wOffset, size_t minElementsNumInBlock)
{
    size_t nDims = dims.size();
    Collection<size_t> inputOffsets(nDims);

    inputOffsets[nDims - 1] = 1;
    for(size_t i = 1; i < nDims; i++)
    {
        inputOffsets[nDims - i - 1] = inputOffsets[nDims - i] * dims[nDims - i];
    }

    wOffset = inputOffsets[wEndDim - 1];

    for(int idx = nDims - 1; idx >= 0; idx--)
    {
        if (inputOffsets[idx] > minElementsNumInBlock)
        {
            fDimN = idx + 1;
            break;
        }
    }
}

} // namespace internal
} // namespace forward
} // namespace prelu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
