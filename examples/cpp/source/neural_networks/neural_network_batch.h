/* file: neural_network_batch.h */
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

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks;
using namespace daal::algorithms::neural_networks::layers;
using namespace daal::services;

Collection<LayerDescriptor> configureNet()
{
    /* Create layers of the neural network */
    /* Create fully-connected layer and initialize layer parameters */
    SharedPtr<layers::fullyconnected::Batch<> > fullyconnectedLayer1(new layers::fullyconnected::Batch<>(20));

    fullyconnectedLayer1->parameter.weightsInitializer = services::SharedPtr<initializers::uniform::Batch<> >(
                                                             new initializers::uniform::Batch<>(-0.001, 0.001));

    fullyconnectedLayer1->parameter.biasesInitializer = services::SharedPtr<initializers::uniform::Batch<> >(
                                                            new initializers::uniform::Batch<>(0, 0.5));

    /* Create fully-connected layer and initialize layer parameters */
    SharedPtr<layers::fullyconnected::Batch<> > fullyconnectedLayer2(new layers::fullyconnected::Batch<>(2));

    fullyconnectedLayer2->parameter.weightsInitializer = services::SharedPtr<initializers::uniform::Batch<> >(
                                                             new initializers::uniform::Batch<>(0.5, 1));

    fullyconnectedLayer2->parameter.biasesInitializer = services::SharedPtr<initializers::uniform::Batch<> >(
                                                            new initializers::uniform::Batch<>(0.5, 1));

    /* Create softmax layer and initialize layer parameters */
    SharedPtr<layers::softmax::Batch<> > softmaxLayer(new layers::softmax::Batch<>());
    softmaxLayer->parameter.dimension = 1;

    /* Create configuration of the neural network */
    Collection<LayerDescriptor> configuration;

    /* Add layers to the configuration of the neural network */
    configuration.push_back(LayerDescriptor(0, fullyconnectedLayer1, NextLayers(1)));
    configuration.push_back(LayerDescriptor(1, fullyconnectedLayer2, NextLayers(2)));
    configuration.push_back(LayerDescriptor(2, softmaxLayer, NextLayers()));

    return configuration;
}
