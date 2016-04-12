/* file: NeuralNetworkConfigurator.java */
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
 //  Content:
 //     Java example of neural network configurator
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.*;
import com.intel.daal.algorithms.neural_networks.layers.LayerDescriptor;
import com.intel.daal.algorithms.neural_networks.layers.NextLayers;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-EXAMPLE-JAVA-NEURALNETWORKCONFIGURATOR">
 * @example NeuralNetworkConfigurator.java
 */
class NeuralNetworkConfigurator {
    public static LayerDescriptors configureNet(DaalContext context) {
        /* Create layers of the neural network */
        /* Create fully-connected layer and initialize layer parameters */
        com.intel.daal.algorithms.neural_networks.layers.fullyconnected.Batch fullyconnectedLayer1 =
            new com.intel.daal.algorithms.neural_networks.layers.fullyconnected.Batch
        (context, Float.class, com.intel.daal.algorithms.neural_networks.layers.fullyconnected.Method.defaultDense, 20, 0);

        fullyconnectedLayer1.parameter.setWeightsInitializer(
            new com.intel.daal.algorithms.neural_networks.initializers.uniform.Batch
            (context, Float.class, com.intel.daal.algorithms.neural_networks.initializers.uniform.Method.defaultDense, -0.001, 0.001));

        fullyconnectedLayer1.parameter.setBiasesInitializer(
            new com.intel.daal.algorithms.neural_networks.initializers.uniform.Batch
            (context, Float.class, com.intel.daal.algorithms.neural_networks.initializers.uniform.Method.defaultDense, 0, 0.5));

        /* Create fully-connected layer and initialize layer parameters */
        com.intel.daal.algorithms.neural_networks.layers.fullyconnected.Batch fullyconnectedLayer2 =
            new com.intel.daal.algorithms.neural_networks.layers.fullyconnected.Batch
        (context, Float.class, com.intel.daal.algorithms.neural_networks.layers.fullyconnected.Method.defaultDense, 2, 0);

        fullyconnectedLayer2.parameter.setWeightsInitializer(
            new com.intel.daal.algorithms.neural_networks.initializers.uniform.Batch
            (context, Float.class, com.intel.daal.algorithms.neural_networks.initializers.uniform.Method.defaultDense, 0.5, 1));

        fullyconnectedLayer2.parameter.setBiasesInitializer(
            new com.intel.daal.algorithms.neural_networks.initializers.uniform.Batch
            (context, Float.class, com.intel.daal.algorithms.neural_networks.initializers.uniform.Method.defaultDense, 0.5, 1));

        /* Create softmax layer and initialize layer parameters */
        com.intel.daal.algorithms.neural_networks.layers.softmax.Batch softmaxLayer =
            new com.intel.daal.algorithms.neural_networks.layers.softmax.Batch
        (context, Float.class, com.intel.daal.algorithms.neural_networks.layers.softmax.Method.defaultDense);
        softmaxLayer.parameter.setDimension(1);

        /* Create configuration of the neural network */
        LayerDescriptors configuration = new LayerDescriptors(context);

        /* Add layers to the configuration of the neural network */
        configuration.pushBack(new LayerDescriptor(context, 0, fullyconnectedLayer1, new NextLayers(context, 1)));
        configuration.pushBack(new LayerDescriptor(context, 1, fullyconnectedLayer2, new NextLayers(context, 2)));
        configuration.pushBack(new LayerDescriptor(context, 2, softmaxLayer, new NextLayers(context)));

        return configuration;
    }
}
