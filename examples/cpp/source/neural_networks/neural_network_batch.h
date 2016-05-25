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

enum LayerIndex
{
    fc1 = 0,
    fc2 = 1,
    sm1 = 2
};

Collection<LayerDescriptor> configureNet()
{
    /* Create layers of the neural network */
    /* Create fully-connected layer and initialize layer parameters */
    SharedPtr<fullyconnected::Batch<> > fullyConnectedLayer1(new fullyconnected::Batch<>(20));

    fullyConnectedLayer1->parameter.weightsInitializer = services::SharedPtr<initializers::uniform::Batch<> >(
                                                             new initializers::uniform::Batch<>(-0.001, 0.001));

    fullyConnectedLayer1->parameter.biasesInitializer = services::SharedPtr<initializers::uniform::Batch<> >(
                                                            new initializers::uniform::Batch<>(0, 0.5));

    /* Create fully-connected layer and initialize layer parameters */
    SharedPtr<fullyconnected::Batch<> > fullyConnectedLayer2(new fullyconnected::Batch<>(2));

    fullyConnectedLayer2->parameter.weightsInitializer = services::SharedPtr<initializers::uniform::Batch<> >(
                                                             new initializers::uniform::Batch<>(0.5, 1));

    fullyConnectedLayer2->parameter.biasesInitializer = services::SharedPtr<initializers::uniform::Batch<> >(
                                                            new initializers::uniform::Batch<>(0.5, 1));

    /* Create softmax layer and initialize layer parameters */
    SharedPtr<loss::softmax_cross::Batch<> > softmaxCrossEntropyLayer(new loss::softmax_cross::Batch<>());

    /* Create configuration of the neural network */
    Collection<LayerDescriptor> configuration;

    /* Add layers to the configuration of the neural network */
    configuration.push_back(LayerDescriptor(fc1, fullyConnectedLayer1, NextLayers(fc2)));
    configuration.push_back(LayerDescriptor(fc2, fullyConnectedLayer2, NextLayers(sm1)));
    configuration.push_back(LayerDescriptor(sm1, softmaxCrossEntropyLayer, NextLayers()));

    return configuration;
}
