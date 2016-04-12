/* file: neural_networks_training_feedforward_impl.i */
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

#ifndef __NEURAL_NETWORKS_TRAINING_FEEDFORWARD_IMPL_I__
#define __NEURAL_NETWORKS_TRAINING_FEEDFORWARD_IMPL_I__

#include "service_numeric_table.h"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
using namespace layers;
namespace training
{
namespace internal
{
namespace
{

template<typename algorithmFPType, CpuType cpu>
SharedPtr<HomogenNumericTable<algorithmFPType> > tensorToColumnTable(SharedPtr<Tensor> tensor)
{
    using namespace data_management;

    SubtensorDescriptor<algorithmFPType> subtensor;
    const Collection<size_t> dims = tensor->getDimensions();
    size_t firstDimension = dims[0];
    tensor->getSubtensor(0, 0, 0, firstDimension, readOnly, subtensor);
    SharedPtr<HomogenNumericTable<algorithmFPType> > table(
        new HomogenNumericTableCPU<algorithmFPType, cpu>(subtensor.getPtr(), 1, subtensor.getSize()));
    return table;
}

template<typename algorithmFPType, CpuType cpu>
SharedPtr<HomogenNumericTable<algorithmFPType> > tensorToRowTable(SharedPtr<Tensor> tensor)
{
    using namespace data_management;

    SubtensorDescriptor<algorithmFPType> subtensor;
    const Collection<size_t> dims = tensor->getDimensions();
    size_t firstDimension = dims[0];
    tensor->getSubtensor(0, 0, 0, firstDimension, readOnly, subtensor);
    SharedPtr<HomogenNumericTable<algorithmFPType> > table(
        new HomogenNumericTableCPU<algorithmFPType, cpu>(subtensor.getPtr(), subtensor.getSize(), 1));
    return table;
}

template<typename algorithmFPType>
SharedPtr<HomogenTensor<algorithmFPType> > tableToTensor(
    const SharedPtr<HomogenNumericTable<algorithmFPType> > &table, const Collection<size_t> &dimensions)
{
    using namespace data_management;
    SharedPtr<HomogenTensor<algorithmFPType> > tensor(new HomogenTensor<algorithmFPType>(dimensions, Tensor::doAllocate));
    algorithmFPType *tensorData = tensor->getArray();
    algorithmFPType *tableData = table->getArray();

    size_t size = 1;
    for(size_t i = 0; i < dimensions.size(); i++)
    {
        size *= dimensions[i];
    }

    for(size_t i = 0; i < size; i++)
    {
        tensorData[i] = tableData[i];
    }

    return tensor;
}

template<typename algorithmFPType>
SharedPtr<HomogenTensor<algorithmFPType> > getSample(const size_t firstElement, const size_t nElements, const SharedPtr<Tensor> tensor)
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
 *  \brief Kernel for Neural Network training
 */
template<typename algorithmFPType, Method method, CpuType cpu>
void NeuralNetworksFeedforwardTrainingKernel<algorithmFPType, method, cpu>::compute(
    const Input *input, const neural_networks::training::Parameter<algorithmFPType> *parameter, Result *result)
{
    using namespace optimization_solver;
    using namespace optimization_solver::internal;

    SharedPtr<Model> nnModel = result->get(model);
    SharedPtr<ForwardLayers> forwardLayers = nnModel->getForwardLayers();
    SharedPtr<BackwardLayers> backwardLayers = nnModel->getBackwardLayers();
    size_t nLayers = forwardLayers->size();

    SharedPtr<Tensor> data = input->get(training::data);
    size_t nSamples = data->getDimensions().get(0);

    SharedPtr<Tensor> groundTruth = input->get(training::groundTruth);

    SharedPtr<Tensor> probabilities = forwardLayers->get(nLayers - 1)->getLayerResult()->get(forward::value);
    SharedPtr<NumericTable> probabilitiesTable = tensorToRowTable<algorithmFPType, cpu>(probabilities);

    SharedPtr<Tensor> objectiveFunctionGradient = backwardLayers->get(nLayers - 1)->getLayerInput()->get(backward::inputGradient);
    SharedPtr<NumericTable> objectiveFunctionGradientTable = tensorToRowTable<algorithmFPType, cpu>(objectiveFunctionGradient);

    optimization_solver::internal::cross_entropy::Batch<algorithmFPType> crossEntropy;
    crossEntropy.input.set(sum_of_loss::probabilities, probabilitiesTable);

    SharedPtr<DataCollection> crossEntropyCollection = SharedPtr<DataCollection>(new DataCollection(3));
    crossEntropyCollection->get(objective_function::gradientIdx) = objectiveFunctionGradientTable;

    SharedPtr<sum_of_loss::Result> crossEntropyResult(new sum_of_loss::Result());
    crossEntropyResult->set(objective_function::resultCollection, crossEntropyCollection);
    crossEntropy.setResult(crossEntropyResult);

    crossEntropy.parameter.resultsToCompute = objective_function::gradient;

    SharedPtr<NumericTable> weightsAndBiases = nnModel->getWeightsAndBiases();
    SharedPtr<NumericTable> weightsAndBiasesDerivatives = nnModel->getWeightsAndBiasesDerivatives();

    SharedPtr<optimization_solver::internal::precomputed::Batch<algorithmFPType> > precomputed(
        new optimization_solver::internal::precomputed::Batch<algorithmFPType>());
    SharedPtr<DataCollection> precomputedCollection = SharedPtr<DataCollection>(new DataCollection(3));
    precomputedCollection->get(objective_function::gradientIdx) = weightsAndBiasesDerivatives;
    SharedPtr<optimization_solver::internal::precomputed::Result> precomputedResult =
        SharedPtr<optimization_solver::internal::precomputed::Result> (new optimization_solver::internal::precomputed::Result());
    precomputedResult->set(objective_function::resultCollection, precomputedCollection);
    precomputed->setResult(precomputedResult);

    SharedPtr<HomogenNumericTable<algorithmFPType> > nIterations(new HomogenNumericTableCPU<algorithmFPType, cpu>(1, 1));
    algorithmFPType *nIterationsArray = nIterations->getArray();
    nIterationsArray[0] = 0;

    SharedPtr<sgd::Batch<algorithmFPType> > sgdAlgorithm = parameter->optimizationSolver;
    sgdAlgorithm->parameter.function = precomputed;
    sgdAlgorithm->input.set(sgd::inputArgument, weightsAndBiases);
    sgdAlgorithm->parameter.nIterations = 1;

    SharedPtr<sgd::Result> sgdResult = SharedPtr<sgd::Result>(new sgd::Result());
    sgdResult->set(sgd::minimum, weightsAndBiases);
    sgdResult->set(sgd::nIterations, nIterations);
    sgdAlgorithm->setResult(sgdResult);

    for(size_t i = 0; i < parameter->nIterations; i++)
    {
        size_t sampleId = i % nSamples;
        SharedPtr<HomogenTensor<algorithmFPType> > sample = getSample<algorithmFPType>(sampleId, 1, data);
        SharedPtr<HomogenTensor<algorithmFPType> > sampleGroundTruth = getSample<algorithmFPType>(sampleId, 1, groundTruth);

        forwardLayers->get(0)->getLayerInput()->set(forward::data, sample);
        forwardLayers->get(0)->allocateLayerData();
        backwardLayers->get(0)->getLayerInput()->set(backward::inputFromForward,
                                                     forwardLayers->get(0)->getLayerResult()->get(forward::resultForBackward));

        for(size_t layerId = 0; layerId < nLayers; layerId++)
        {
            forwardLayers->get(layerId)->compute();
        }

        SharedPtr<NumericTable> groundTruthTable = tensorToColumnTable<algorithmFPType, cpu>(sampleGroundTruth);
        crossEntropy.input.set(sum_of_loss::groundTruth, groundTruthTable);

        crossEntropy.compute();

        for(int layerId = nLayers - 1; layerId >= 0; layerId--)
        {
            backwardLayers->get(layerId)->compute();
        }

        sgdAlgorithm->compute();
    }
}

} // namespace daal::internal
} // namespace feedforward
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
