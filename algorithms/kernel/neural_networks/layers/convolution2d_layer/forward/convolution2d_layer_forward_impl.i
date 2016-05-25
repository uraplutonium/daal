/* file: convolution2d_layer_forward_impl.i */
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
//  Implementation of convolution2d algorithm
//--
*/

//#include "mkl_daal.h"
//#include "mkl_dnn_types.h"
#include "service_dnn.h"

#define ON_ERR(err) { \
    if ((err) != E_SUCCESS) { \
        if((err) == E_MEMORY_ERROR) {this->_errors->add(services::ErrorMemoryAllocationFailed);return;} \
        this->_errors->add(services::ErrorConvolutionInternal);\
        return; \
    } \
}

using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace convolution2d
{
namespace forward
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu>
static dnnError_t init_conversion(dnnPrimitive_t *cv, algorithmFPType **ptr_out,
                                 dnnLayout_t lt_pr, dnnLayout_t lt_us, algorithmFPType *ptr_us)
{
    dnnError_t err = E_SUCCESS;
    *ptr_out = NULL;
    if (!Dnn<algorithmFPType, cpu>::xLayoutCompare(lt_pr, lt_us))
    {
        err = Dnn<algorithmFPType, cpu>::xConversionCreate(cv, lt_us, lt_pr);
        if(err != E_SUCCESS) return err;
        err = Dnn<algorithmFPType, cpu>::xAllocateBuffer((void**)ptr_out, lt_pr);
    }
    else
    {
        *ptr_out = ptr_us;
    }
    return err;
}

template<typename algorithmFPType, Method method, CpuType cpu>
void Convolution2dKernel<algorithmFPType, method, cpu>::compute(const convolution2d::forward::Input *input,
    const convolution2d::Parameter *parameter, convolution2d::forward::Result *result)
{
    dnnError_t err;
    typedef Dnn<algorithmFPType, cpu> dnn;

    const size_t dimension = 4;

    SharedPtr<Tensor> inputTable   = input->get(layers::forward::data);
    SharedPtr<Tensor> wTable       = input->get(layers::forward::weights);
    SharedPtr<Tensor> bTable       = input->get(layers::forward::biases);
    SharedPtr<Tensor> resultTable  = result->get(layers::forward::value);

    const services::Collection<size_t>& inDimsFull  = inputTable->getDimensions();
    const services::Collection<size_t>& wDims   = wTable->getDimensions();
    const services::Collection<size_t>& bDims   = bTable->getDimensions();
    const services::Collection<size_t>& outDimsFull = resultTable->getDimensions();
    services::Collection<size_t> inDims(dimension);
    services::Collection<size_t> outDims(dimension);

    size_t fullSize = inDimsFull.size();
    size_t inDims0 = 1;
    size_t outDims0 = 1;
    for (size_t i = 0; i < fullSize - dimension + 1; i++)
    {
        inDims0 *= inDimsFull[i];
        outDims0 *= outDimsFull[i];
    }
    inDims[0] = inDims0;
    outDims[0] = outDims0;
    for (size_t i = 1; i < dimension; i++)
    {
        inDims[i] = inDimsFull[fullSize - dimension + i];
        outDims[i] = outDimsFull[fullSize - dimension + i];
    }

    SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTable->getSubtensor(0, 0, 0, inDimsFull[0], readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> wBlock;
    wTable->getSubtensor(0, 0, 0, wDims[0], readOnly, wBlock);
    algorithmFPType *wArray = wBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> bBlock;
    bTable->getSubtensor(0, 0, 0, bDims[0], readOnly, bBlock);
    algorithmFPType *bArray = bBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTable->getSubtensor(0, 0, 0, outDimsFull[0], writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    const size_t bufferSize = 6 * dimension * sizeof(size_t);
    size_t *buffer = (size_t*)services::daal_malloc(bufferSize);
    if(!buffer) {this->_errors->add(services::ErrorMemoryAllocationFailed); return;}
    size_t *inputSize     = buffer;
    size_t *inputStrides  = buffer +     dimension;
    size_t *outputSize    = buffer + 2 * dimension;
    size_t *outputStrides = buffer + 3 * dimension;
    size_t *filterSize    = buffer + 4 * dimension;
    size_t *filterStrides = buffer + 5 * dimension;

    size_t  biasSize[1] = {parameter->nKernels};
    size_t  biasStrides[1] = {1};

    inputSize    [0] = inDims [dimension-1];
    inputStrides [0] = 1;
    outputSize   [0] = outDims[dimension-1];
    outputStrides[0] = 1;
    filterSize   [0] = wDims  [dimension-1];
    filterStrides[0] = 1;

    for(size_t i=1; i<dimension; i++)
    {
        inputSize    [i] = inDims[dimension-1-i];
        inputStrides [i] = inputStrides [i-1]*inputSize[i-1];
        outputSize   [i] = outDims[dimension-1-i];
        outputStrides[i] = outputStrides[i-1]*outputSize[i-1];
        filterSize   [i] = wDims[dimension-1-i];
        filterStrides[i] = filterStrides[i-1]*filterSize[i-1];
    }

    size_t convolutionStride[2] = {parameter->stride.size[1], parameter->stride.size[0]};
    int    inputOffset      [2] = {-(int)(parameter->padding.size[1]), -(int)(parameter->padding.size[0])};

    dnnLayout_t ltUserInput=NULL, ltUserFilt=NULL, ltUserBias=NULL, ltUserOutput=NULL;

    err = dnn::xLayoutCreate(&ltUserInput,  dimension, inputSize,  inputStrides ); ON_ERR(err);
    err = dnn::xLayoutCreate(&ltUserFilt,   dimension, filterSize, filterStrides); ON_ERR(err);
    err = dnn::xLayoutCreate(&ltUserBias,   1,         biasSize,   biasStrides  ); ON_ERR(err);
    err = dnn::xLayoutCreate(&ltUserOutput, dimension, outputSize, outputStrides); ON_ERR(err);

    dnnPrimitive_t convPrim = NULL;

    err = dnn::xConvolutionCreateForwardBias( &convPrim, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize,
        filterSize, convolutionStride, inputOffset, dnnBorderZeros);  ON_ERR(err);

    dnnLayout_t ltInnerInput = NULL, ltInnerFilt = NULL, ltInnerBias = NULL, ltInnerOutput = NULL;

    err = dnn::xLayoutCreateFromPrimitive(&ltInnerInput,  convPrim, dnnResourceSrc   ); ON_ERR(err);
    err = dnn::xLayoutCreateFromPrimitive(&ltInnerFilt,   convPrim, dnnResourceFilter); ON_ERR(err);
    err = dnn::xLayoutCreateFromPrimitive(&ltInnerBias,   convPrim, dnnResourceBias  ); ON_ERR(err);
    err = dnn::xLayoutCreateFromPrimitive(&ltInnerOutput, convPrim, dnnResourceDst   ); ON_ERR(err);

    algorithmFPType* uOut;
    algorithmFPType* convRes[dnnResourceNumber] = {0};
    dnnPrimitive_t cvToInnerInput = NULL, cvToInnerFilt = NULL, cvToInnerBias = NULL, cvFromInnerOutput = NULL;
    err = init_conversion<algorithmFPType, cpu>(&cvToInnerInput, &convRes[dnnResourceSrc],    ltInnerInput, ltUserInput,
        (algorithmFPType*)inputArray); ON_ERR(err);
    err = init_conversion<algorithmFPType, cpu>(&cvToInnerFilt,  &convRes[dnnResourceFilter], ltInnerFilt,  ltUserFilt,
        (algorithmFPType*)wArray    ); ON_ERR(err);
    err = init_conversion<algorithmFPType, cpu>(&cvToInnerBias,  &convRes[dnnResourceBias],   ltInnerBias,  ltUserBias,
        (algorithmFPType*)bArray    ); ON_ERR(err);

    err = dnn::xAllocateBuffer((void**) &convRes[dnnResourceDst],    ltInnerOutput); ON_ERR(err);

    err = init_conversion<algorithmFPType, cpu>(&cvFromInnerOutput, (algorithmFPType**)&uOut, ltUserOutput, ltInnerOutput,
        convRes[dnnResourceDst]); ON_ERR(err);

    err = dnn::xConversionExecute(cvToInnerInput, inputArray, convRes[dnnResourceSrc]   ); ON_ERR(err);
    err = dnn::xConversionExecute(cvToInnerFilt,  wArray,     convRes[dnnResourceFilter]); ON_ERR(err);
    err = dnn::xConversionExecute(cvToInnerBias,  bArray,     convRes[dnnResourceBias]  ); ON_ERR(err);

    err = dnn::xExecute(convPrim, (void**)convRes); ON_ERR(err);

    err = dnn::xConversionExecute(cvFromInnerOutput, convRes[dnnResourceDst], uOut); ON_ERR(err);

    size_t size = resultBlock.getSize();
    for(size_t i=0; i<size; i++)
    {
        resultArray[i] = uOut[i];
    }

    dnn::xReleaseBuffer(convRes[dnnResourceDst]);

    dnn::xDelete(convPrim);
    dnn::xDelete(cvToInnerInput);
    dnn::xDelete(cvToInnerFilt);
    dnn::xDelete(cvToInnerBias);
    dnn::xDelete(cvFromInnerOutput);

    dnn::xLayoutDelete(ltUserInput);
    dnn::xLayoutDelete(ltUserFilt);
    dnn::xLayoutDelete(ltUserBias);
    dnn::xLayoutDelete(ltUserOutput);
    dnn::xLayoutDelete(ltInnerInput);
    dnn::xLayoutDelete(ltInnerFilt);
    dnn::xLayoutDelete(ltInnerBias);
    dnn::xLayoutDelete(ltInnerOutput);

    services::daal_free(buffer);

    inputTable->releaseSubtensor(inputBlock);
    wTable->releaseSubtensor(wBlock);
    bTable->releaseSubtensor(bBlock);
    resultTable->releaseSubtensor(resultBlock);
}

} // internal
} // forward
} // namespace convolution2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
