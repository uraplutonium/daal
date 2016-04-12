/* file: neural_networks_prediction_feedforward_impl.i */
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
//  Implementation of feedforward algorithm
//--
*/

#ifndef __NEURAL_NETWORKS_PREDICTION_FEEDFORWARD_IMPL_I__
#define __NEURAL_NETWORKS_PREDICTION_FEEDFORWARD_IMPL_I__

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
using namespace layers;
namespace prediction
{
namespace internal
{

namespace
{
template<typename algorithmFPType>
SharedPtr<HomogenTensor<algorithmFPType> > getSample(const size_t firstElement, const size_t nElements, const SharedPtr<Tensor> tensor)
{
    SubtensorDescriptor<algorithmFPType> tensorBlock;
    tensor->getSubtensor(0, 0, firstElement, nElements, readOnly, tensorBlock);
    algorithmFPType *sampleArray = tensorBlock.getPtr();

    Collection<size_t> tensorSize = tensor->getDimensions();
    Collection<size_t> sampleSize;
    sampleSize.push_back(nElements);
    for(size_t i = 1; i < tensorSize.size(); i++) { sampleSize.push_back(tensorSize[i]); }

    SharedPtr<HomogenTensor<algorithmFPType> > sample(new HomogenTensor<algorithmFPType>(sampleSize, sampleArray));
    tensor->releaseSubtensor(tensorBlock);
    return sample;
}

template<typename algorithmFPType>
SharedPtr<HomogenTensor<algorithmFPType> > getSampleValue(const size_t firstElement, const size_t nElements, const SharedPtr<Tensor> tensor)
{
    SubtensorDescriptor<algorithmFPType> tensorBlock;
    tensor->getSubtensor(0, 0, firstElement, nElements, readOnly, tensorBlock);
    algorithmFPType *sampleArray = tensorBlock.getPtr();

    Collection<size_t> tensorSize = tensor->getDimensions();

    tensorSize[0] = nElements;

    SharedPtr<HomogenTensor<algorithmFPType> > sample(new HomogenTensor<algorithmFPType>(tensorSize, Tensor::doAllocate));

    SubtensorDescriptor<algorithmFPType> sampleBlock;
    sample->getSubtensor(0, 0, 0, nElements, writeOnly, sampleBlock);
    algorithmFPType *sampleData = sampleBlock.getPtr();

    size_t size = tensorBlock.getSize();
    for (int i = 0; i < size; i++) {
        sampleData[i] = sampleArray[i];
    }

    tensor->releaseSubtensor(tensorBlock);
    sample->releaseSubtensor(sampleBlock);
    return sample;
}
}

/**
 *  \brief Kernel for Neural Network prediction
 */
template<typename algorithmFPType, Method method, CpuType cpu>
void NeuralNetworksFeedforwardPredictionKernel<algorithmFPType, method, cpu>::compute(
    const Input *input, const neural_networks::prediction::Parameter *parameter, Result *result)
{
    SharedPtr<Model> model = input->get(prediction::model);
    SharedPtr<ForwardLayers> forwardLayers = model->getLayers();
    size_t nLayers = forwardLayers->size();

    SharedPtr<Tensor> data = input->get(prediction::data);
    SharedPtr<Tensor> predictionResults = result->get(prediction::prediction);
    size_t nSamples = data->getDimensions().get(0);

    for(size_t i = 0; i < nSamples; i++)
    {
        SharedPtr<HomogenTensor<algorithmFPType> > sample = getSampleValue<algorithmFPType>(i, 1, data);
        SharedPtr<HomogenTensor<algorithmFPType> > predictionSample = getSample<algorithmFPType>(i, 1, predictionResults);

        forwardLayers->get(0)->getLayerInput()->set(forward::data, sample);
        forwardLayers->get(nLayers - 1)->getLayerResult()->set(forward::value, predictionSample);

        for(size_t layerId = 0; layerId < nLayers; layerId++)
        {
            forwardLayers->get(layerId)->compute();
        }
    }
}

} // namespace daal::internal
} // namespace feedforward
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
