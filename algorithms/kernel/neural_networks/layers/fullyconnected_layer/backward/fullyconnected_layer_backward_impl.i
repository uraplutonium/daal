/* file: fullyconnected_layer_backward_impl.i */
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
//  Implementation of fullyconnected algorithm
//--
*/

#ifndef __FULLYCONNECTED_LAYER_BACKWARD_IMPL_I__
#define __FULLYCONNECTED_LAYER_BACKWARD_IMPL_I__

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
namespace fullyconnected
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void FullyconnectedKernel<algorithmFPType, method, cpu>::compute(const fullyconnected::backward::Input *input,
    const fullyconnected::Parameter *parameter, fullyconnected::backward::Result *result)
{
    SharedPtr<Tensor> inGradTable  = input->get(layers::backward::inputGradient);
    SharedPtr<LayerData> layerData = input->get(layers::backward::inputFromForward);
    SharedPtr<Tensor> xTable       = staticPointerCast<Tensor, SerializationIface>((*layerData)[fullyconnected::auxData]);
    SharedPtr<Tensor> wTable       = staticPointerCast<Tensor, SerializationIface>((*layerData)[fullyconnected::auxWeights]);
    SharedPtr<Tensor> wDerTable    = result->get(layers::backward::weightDerivatives);
    SharedPtr<Tensor> bDerTable    = result->get(layers::backward::biasDerivatives);
    SharedPtr<Tensor> resultTable  = result->get(layers::backward::gradient);

    size_t k = parameter->dim;
    size_t m = parameter->nOutputs;

    const services::Collection<size_t>& xDims = xTable->getDimensions();
    const services::Collection<size_t>& wDims = wDerTable->getDimensions();

    size_t nDims = xDims.size();

    size_t* dimsCounter = (size_t*)services::daal_malloc(sizeof(size_t) * nDims);
    if(!dimsCounter) {this->_errors->add(services::ErrorMemoryAllocationFailed); return;}

    SubtensorDescriptor<algorithmFPType> inGradBlock;
    inGradTable->getSubtensor(0, 0, 0, xDims[k], readOnly, inGradBlock);
    algorithmFPType *inGradArray = inGradBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> xBlock;
    xTable->getSubtensor(0, 0, 0, xDims[0], readOnly, xBlock);
    algorithmFPType *xArray = xBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> wBlock;
    wTable->getSubtensor(0, 0, 0, wDims[0], readOnly, wBlock);
    algorithmFPType *wArray = wBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> wDerBlock;
    wDerTable->getSubtensor(0, 0, 0, wDims[0], writeOnly, wDerBlock);
    algorithmFPType *wDerArray = wDerBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> bDerBlock;
    bDerTable->getSubtensor(0, 0, 0, m, writeOnly, bDerBlock);
    algorithmFPType *bDerArray = bDerBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTable->getSubtensor(0, 0, 0, xDims[0], writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    size_t size = 1;
    for(size_t i=0; i<nDims; i++)
    {
        dimsCounter[i] = 0;
        size *= xDims[i];
    }
    size_t wSize = size / xDims[k] * m;

    for(size_t i=0; i<m; i++)
    {
        bDerArray[i] = (algorithmFPType)0.0;
    }

    for(size_t j=0; j<xDims[k]; j++)
    {
        for(size_t i=0; i<m; i++)
        {
            bDerArray[i] += inGradArray[j*m + i];
        }
    }

    for(size_t j=0; j<size; j++)
    {
        resultArray[j] = 0;
    }

    for(size_t j=0; j<wSize; j++)
    {
        wDerArray[j] = 0;
    }

    for(size_t i=0; i<m; i++)
    {
        for(size_t j=0; j<size; j++)
        {
            size_t wJ = 0;

            size_t mul=1;
            for(size_t d=1; d<nDims+1; d++)
            {
                if(nDims-d != k)
                {
                    wJ += dimsCounter[nDims-d]*mul;
                    mul *= xDims[nDims-d];
                }
                else
                {
                    wJ += i*mul;
                    mul *= m;
                }
            }

            resultArray[j] += inGradArray[dimsCounter[k]*m + i] * wArray[wJ];

            wDerArray[wJ]  += inGradArray[dimsCounter[k]*m + i] * xArray[j];

            for(size_t d=1; d<nDims+1; d++)
            {
                dimsCounter[nDims-d]++;
                if(dimsCounter[nDims-d] < xDims[nDims-d]) break;
                dimsCounter[nDims-d]=0;
            }
        }
    }

    inGradTable->releaseSubtensor(inGradBlock);
    xTable->releaseSubtensor(xBlock);
    wTable->releaseSubtensor(wBlock);
    wDerTable->releaseSubtensor(wDerBlock);
    bDerTable->releaseSubtensor(bDerBlock);
    resultTable->releaseSubtensor(resultBlock);

    services::daal_free( dimsCounter );
}

} // internal
} // backward
} // namespace fullyconnected
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
