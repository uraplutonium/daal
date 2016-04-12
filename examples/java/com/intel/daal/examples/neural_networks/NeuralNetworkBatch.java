/* file: NeuralNetworkBatch.java */
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
 //     Java example of neural network in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.*;
import com.intel.daal.algorithms.neural_networks.prediction.*;
import com.intel.daal.algorithms.neural_networks.training.*;
import com.intel.daal.algorithms.neural_networks.layers.LayerDescriptor;
import com.intel.daal.algorithms.neural_networks.layers.NextLayers;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.ForwardInputId;
import com.intel.daal.algorithms.neural_networks.layers.ForwardResultId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardInputId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardResultId;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.data_management.data.HomogenTensor;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-EXAMPLE-JAVA-NEURALNETWORKBATCH">
 * @example NeuralNetworkBatch.java
 */
class NeuralNetworkBatch {
    private static final String trainDatasetFile     = "../data/batch/neural_network_train.csv";
    private static final String trainGroundTruthFile = "../data/batch/neural_network_train_ground_truth.csv";
    private static final String testDatasetFile      = "../data/batch/neural_network_test.csv";
    private static final String testGroundTruthFile  = "../data/batch/neural_network_test_ground_truth.csv";
    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Read datasetFileName from a file and create a tensor to store input data */
        Tensor trainingData = Service.readTensorFromCSV(context, trainDatasetFile);
        Tensor trainingGroundTruth = Service.readTensorFromCSV(context, trainGroundTruthFile);
        Tensor predictionData = Service.readTensorFromCSV(context, testDatasetFile);
        Tensor predictionGroundTruth = Service.readTensorFromCSV(context, testGroundTruthFile);

        /* Create an algorithm to compute neural network results using default method */
        TrainingBatch trainingNet = new TrainingBatch(context, Float.class, TrainingMethod.defaultDense);

        /* Configure the neural network */
        LayerDescriptors layersConfiguration = NeuralNetworkConfigurator.configureNet(context);
        trainingNet.initialize(trainingData.getDimensions(), layersConfiguration);

        /* Set input objects for the neural network */
        trainingNet.input.set(TrainingInputId.data, trainingData);
        trainingNet.input.set(TrainingInputId.groundTruth, trainingGroundTruth);

        /* Set learning rate for the optimization solver used in the neural network */
        double[] data = new double[1];
        trainingNet.parameter.getOptimizationSolver().parameter.setLearningRateSequence(
            new HomogenNumericTable(context, data, 1, 1, 0.001));

        /* Set the number of iterations to be done by the neural network */
        trainingNet.parameter.setNIterations(6000);

        /* Run the neural network training */
        TrainingResult result = trainingNet.compute();

        /* Get training and prediction models of the neural network */
        TrainingModel trainingModel = result.get(TrainingResultId.model);
        PredictionModel predictionModel = trainingModel.getPredictionModel();

        /* Create an algorithm to compute the neural network predictions */
        PredictionBatch predictionNet = new PredictionBatch(context, Float.class, PredictionMethod.defaultDense);

        /* Set input objects for the prediction neural network */
        predictionNet.input.set(PredictionTensorInputId.data, predictionData);
        predictionNet.input.set(PredictionModelInputId.model, predictionModel);

        /* Run the neural network prediction */
        PredictionResult predictionResult = predictionNet.compute();

        /* Print results of the neural network prediction */
        Service.printTensors("Ground truth", "Neural network predictions: each class probability",
                             "Neural network classification results (first 20 observations):",
                             predictionGroundTruth, predictionResult.get(PredictionResultId.prediction), 20);

        context.dispose();

    }
}
