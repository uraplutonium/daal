/* file: convolution2d_layer_backward_impl.i */
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
using namespace daal::data_management;

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
namespace backward
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
void Convolution2dKernel<algorithmFPType, method, cpu>::compute(const convolution2d::backward::Input *input,
    const convolution2d::Parameter *parameter, convolution2d::backward::Result *result)
{
    dnnError_t err;
    typedef Dnn<algorithmFPType, cpu> dnn;

    const size_t dimension = 4;

    SharedPtr<Tensor> inGradTable  = input->get(layers::backward::inputGradient);
    SharedPtr<LayerData> layerData = input->get(layers::backward::inputFromForward);
    SharedPtr<Tensor> xTable       = staticPointerCast<Tensor, SerializationIface>((*layerData)[convolution2d::auxData]);
    SharedPtr<Tensor> wTable       = staticPointerCast<Tensor, SerializationIface>((*layerData)[convolution2d::auxWeights]);
    SharedPtr<Tensor> wDerTable    = result->get(layers::backward::weightDerivatives);
    SharedPtr<Tensor> bDerTable    = result->get(layers::backward::biasDerivatives);
    SharedPtr<Tensor> resultTable  = result->get(layers::backward::gradient);

    const services::Collection<size_t>& gDimsFull = inGradTable->getDimensions();
    const services::Collection<size_t>& xDimsFull = xTable->getDimensions();
    const services::Collection<size_t>& wDims = wDerTable->getDimensions();
    const services::Collection<size_t>& bDims = bDerTable->getDimensions();
    services::Collection<size_t> gDims(dimension);
    services::Collection<size_t> xDims(dimension);

    size_t fullSize = xDimsFull.size();
    size_t gDims0 = 1;
    size_t xDims0 = 1;
    for (size_t i = 0; i < fullSize - dimension + 1; i++)
    {
        gDims0 *= gDimsFull[i];
        xDims0 *= xDimsFull[i];
    }
    gDims[0] = gDims0;
    xDims[0] = xDims0;
    for (size_t i = 1; i < dimension; i++)
    {
        gDims[i] = gDimsFull[fullSize - dimension + i];
        xDims[i] = xDimsFull[fullSize - dimension + i];
    }

    SubtensorDescriptor<algorithmFPType> inGradBlock;
    inGradTable->getSubtensor(0, 0, 0, gDimsFull[0], readOnly, inGradBlock);
    algorithmFPType *inGradArray = inGradBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> xBlock;
    xTable->getSubtensor(0, 0, 0, xDimsFull[0], readOnly, xBlock);
    algorithmFPType *xArray = xBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> wBlock;
    wTable->getSubtensor(0, 0, 0, wDims[0], readOnly, wBlock);
    algorithmFPType *wArray = wBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> wDerBlock;
    wDerTable->getSubtensor(0, 0, 0, wDims[0], writeOnly, wDerBlock);
    algorithmFPType *wDerArray = wDerBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> bDerBlock;
    bDerTable->getSubtensor(0, 0, 0, bDims[0], writeOnly, bDerBlock);
    algorithmFPType *bDerArray = bDerBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTable->getSubtensor(0, 0, 0, xDimsFull[0], writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    const size_t bufferSize = 6 * dimension * sizeof(size_t);
    size_t *buffer = (size_t*)services::daal_malloc(bufferSize);
    if(!buffer) {this->_errors->add(services::ErrorMemoryAllocationFailed); return;}
    size_t *xSize         = buffer;
    size_t *xStrides      = buffer +     dimension;
    size_t *gradSize      = buffer + 2 * dimension;
    size_t *gradStrides   = buffer + 3 * dimension;
    size_t *filterSize    = buffer + 4 * dimension;
    size_t *filterStrides = buffer + 5 * dimension;

    size_t  biasSize[1] = {parameter->nKernels};
    size_t  biasStrides[1] = {1};

    xSize        [0] = xDims[dimension-1];
    xStrides     [0] = 1;
    gradSize     [0] = gDims[dimension-1];
    gradStrides  [0] = 1;
    filterSize   [0] = wDims[dimension-1];
    filterStrides[0] = 1;

    for(size_t i=1; i<dimension; i++)
    {
        xSize        [i] = xDims[dimension-1-i];
        xStrides     [i] = xStrides[i-1]*xSize[i-1];
        gradSize     [i] = gDims[dimension-1-i];
        gradStrides  [i] = gradStrides[i-1]*gradSize[i-1];
        filterSize   [i] = wDims[dimension-1-i];
        filterStrides[i] = filterStrides[i-1]*filterSize[i-1];
    }

    size_t convolutionStride[2] = {parameter->stride.size[1],  parameter->stride.size[0]};
    int    xOffset          [2] = {-(int)(parameter->padding.size[1]), -(int)(parameter->padding.size[0])};

    dnnLayout_t ltUserX=NULL, ltUserFilt=NULL, ltUserBias=NULL, ltUserGrad=NULL;

    err = dnn::xLayoutCreate(&ltUserX,      dimension, xSize,      xStrides     ); ON_ERR(err);
    err = dnn::xLayoutCreate(&ltUserFilt,   dimension, filterSize, filterStrides); ON_ERR(err);
    err = dnn::xLayoutCreate(&ltUserBias,   1,         biasSize,   biasStrides  ); ON_ERR(err);
    err = dnn::xLayoutCreate(&ltUserGrad,   dimension, gradSize,   gradStrides  ); ON_ERR(err);

    dnnPrimitive_t convFwd = NULL, convGrad = NULL, convFilt = NULL, convBias = NULL;

    err = dnn::xConvolutionCreateForwardBias   ( &convFwd,  dnnAlgorithmConvolutionDirect, dimension, xSize, gradSize,
                                                 filterSize, convolutionStride, xOffset, dnnBorderZeros);  ON_ERR(err);
    err = dnn::xConvolutionCreateBackwardData  ( &convGrad, dnnAlgorithmConvolutionDirect, dimension, xSize, gradSize,
                                                 filterSize, convolutionStride, xOffset, dnnBorderZeros);  ON_ERR(err);
    err = dnn::xConvolutionCreateBackwardFilter( &convFilt, dnnAlgorithmConvolutionDirect, dimension, xSize, gradSize,
                                                 filterSize, convolutionStride, xOffset, dnnBorderZeros);  ON_ERR(err);
    err = dnn::xConvolutionCreateBackwardBias  ( &convBias, dnnAlgorithmConvolutionDirect, dimension, gradSize);  ON_ERR(err);

    dnnLayout_t ltInnerInput = NULL, ltInnerFilt = NULL, ltInnerGrad = NULL;
    dnnLayout_t ltInnerBack  = NULL, ltInnerDerFilt = NULL, ltInnerDerBias = NULL;

    err = dnn::xLayoutCreateFromPrimitive(&ltInnerInput, convFwd, dnnResourceSrc   ); ON_ERR(err);
    err = dnn::xLayoutCreateFromPrimitive(&ltInnerFilt,  convFwd, dnnResourceFilter); ON_ERR(err);
    err = dnn::xLayoutCreateFromPrimitive(&ltInnerGrad,  convFwd, dnnResourceDst   ); ON_ERR(err);

    err = dnn::xLayoutCreateFromPrimitive(&ltInnerBack,    convGrad, dnnResourceDiffSrc   ); ON_ERR(err);
    err = dnn::xLayoutCreateFromPrimitive(&ltInnerDerFilt, convFilt, dnnResourceDiffFilter); ON_ERR(err);
    err = dnn::xLayoutCreateFromPrimitive(&ltInnerDerBias, convBias, dnnResourceDiffBias  ); ON_ERR(err);

    algorithmFPType* convRes[dnnResourceNumber] = {0};
    dnnPrimitive_t cvToInnerInput = NULL, cvToInnerFilt = NULL, cvToInnerGrad = NULL;
    dnnPrimitive_t cvFromInnerBack = NULL, cvFromInnerDerFilt = NULL, cvFromInnerDerBias = NULL;
    err = init_conversion<algorithmFPType, cpu>(&cvToInnerInput, &convRes[dnnResourceSrc    ], ltInnerInput, ltUserX,
        (algorithmFPType*)xArray); ON_ERR(err);
    err = init_conversion<algorithmFPType, cpu>(&cvToInnerFilt,  &convRes[dnnResourceFilter ], ltInnerFilt,  ltUserFilt,
        (algorithmFPType*)wArray); ON_ERR(err);
    err = init_conversion<algorithmFPType, cpu>(&cvToInnerGrad,  &convRes[dnnResourceDiffDst], ltInnerGrad,  ltUserGrad,
        (algorithmFPType*)inGradArray); ON_ERR(err);

    err = dnn::xAllocateBuffer((void**) &convRes[dnnResourceDiffSrc   ], ltInnerBack   ); ON_ERR(err);
    err = dnn::xAllocateBuffer((void**) &convRes[dnnResourceDiffFilter], ltInnerDerFilt); ON_ERR(err);
    err = dnn::xAllocateBuffer((void**) &convRes[dnnResourceDiffBias  ], ltInnerDerBias); ON_ERR(err);

    algorithmFPType *backData=0, *derFilt=0, *derBias=0;
    err = init_conversion<algorithmFPType, cpu>(&cvFromInnerBack,    (algorithmFPType**)&backData, ltUserX   , ltInnerBack   ,
        convRes[dnnResourceDiffSrc   ]); ON_ERR(err);
    err = init_conversion<algorithmFPType, cpu>(&cvFromInnerDerFilt, (algorithmFPType**)&derFilt,  ltUserFilt, ltInnerDerFilt,
        convRes[dnnResourceDiffFilter]); ON_ERR(err);
    err = init_conversion<algorithmFPType, cpu>(&cvFromInnerDerBias, (algorithmFPType**)&derBias,  ltUserBias, ltInnerDerBias,
        convRes[dnnResourceDiffBias  ]); ON_ERR(err);

    err = dnn::xConversionExecute(cvToInnerInput, xArray,      convRes[dnnResourceSrc    ]); ON_ERR(err);
    err = dnn::xConversionExecute(cvToInnerFilt,  wArray,      convRes[dnnResourceFilter ]); ON_ERR(err);
    err = dnn::xConversionExecute(cvToInnerGrad,  inGradArray, convRes[dnnResourceDiffDst]); ON_ERR(err);

    err = dnn::xExecute(convGrad, (void**)convRes); ON_ERR(err);
    err = dnn::xExecute(convFilt, (void**)convRes); ON_ERR(err);
    err = dnn::xExecute(convBias, (void**)convRes); ON_ERR(err);

    err = dnn::xConversionExecute(cvFromInnerBack,    convRes[dnnResourceDiffDst   ], backData); ON_ERR(err);
    err = dnn::xConversionExecute(cvFromInnerDerFilt, convRes[dnnResourceDiffFilter], derFilt ); ON_ERR(err);
    err = dnn::xConversionExecute(cvFromInnerDerBias, convRes[dnnResourceDiffBias  ], derBias ); ON_ERR(err);

    size_t size = resultBlock.getSize();
    for(size_t i=0; i<size; i++)
    {
        resultArray[i] = backData[i];
    }

    size = wDerBlock.getSize();
    for(size_t i=0; i<size; i++)
    {
        wDerArray[i] = derFilt[i];
    }

    size = bDerBlock.getSize();
    for(size_t i=0; i<size; i++)
    {
        bDerArray[i] = derBias[i];
    }

    dnn::xReleaseBuffer(convRes[dnnResourceDiffSrc   ]);
    dnn::xReleaseBuffer(convRes[dnnResourceDiffFilter]);
    dnn::xReleaseBuffer(convRes[dnnResourceDiffBias  ]);

    dnn::xDelete(convFwd );
    dnn::xDelete(convGrad);
    dnn::xDelete(convFilt);
    dnn::xDelete(convBias);

    dnn::xDelete(cvToInnerInput);
    dnn::xDelete(cvToInnerFilt);
    dnn::xDelete(cvToInnerGrad);
    dnn::xDelete(cvFromInnerBack);
    dnn::xDelete(cvFromInnerDerFilt);
    dnn::xDelete(cvFromInnerDerBias);

    dnn::xLayoutDelete(ltUserX);
    dnn::xLayoutDelete(ltUserFilt);
    dnn::xLayoutDelete(ltUserBias);
    dnn::xLayoutDelete(ltUserGrad);

    dnn::xLayoutDelete(ltInnerInput);
    dnn::xLayoutDelete(ltInnerFilt);
    dnn::xLayoutDelete(ltInnerGrad);

    dnn::xLayoutDelete(ltInnerBack);
    dnn::xLayoutDelete(ltInnerDerFilt);
    dnn::xLayoutDelete(ltInnerDerBias);

    services::daal_free(buffer);

    inGradTable->releaseSubtensor(inGradBlock);
    xTable->releaseSubtensor(xBlock);
    wTable->releaseSubtensor(wBlock);
    wDerTable->releaseSubtensor(wDerBlock);
    bDerTable->releaseSubtensor(bDerBlock);
    resultTable->releaseSubtensor(resultBlock);
}

} // internal
} // backward
} // namespace convolution2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
