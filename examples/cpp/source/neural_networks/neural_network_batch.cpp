/* file: neural_network_batch.cpp */
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

/**
 * <a name="DAAL-EXAMPLE-CPP-NEURAL_NETWORK_BATCH"></a>
 * \example neural_network_batch.cpp
 */

#include "daal.h"
#include "service.h"
#include "neural_network_batch.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks::training;
using namespace daal::services;

/* Input data set parameters */
string trainDatasetFile     = "../data/batch/neural_network_train.csv";
string trainGroundTruthFile = "../data/batch/neural_network_train_ground_truth.csv";
string testDatasetFile      = "../data/batch/neural_network_test.csv";
string testGroundTruthFile  = "../data/batch/neural_network_test_ground_truth.csv";

int main()
{
    /* Read datasetFile from a file and create a tensor to store input data */
    SharedPtr<Tensor> trainingData = readTensorFromCSV(trainDatasetFile);
    SharedPtr<Tensor> trainingGroundTruth = readTensorFromCSV(trainGroundTruthFile);
    SharedPtr<Tensor> predictionData = readTensorFromCSV(testDatasetFile);
    SharedPtr<Tensor> predictionGroundTruth = readTensorFromCSV(testGroundTruthFile);

    /* Create an algorithm to train neural network */
    training::Batch<> trainingNet;

    /* Configure the neural network */
    Collection<LayerDescriptor> layersConfiguration = configureNet();
    trainingNet.initialize(trainingData->getDimensions(), layersConfiguration);

    /* Set input objects for the training neural network */
    trainingNet.input.set(training::data, trainingData);
    trainingNet.input.set(groundTruth, trainingGroundTruth);

    /* Set learning rate for the optimization solver used in the neural network */
    trainingNet.parameter.optimizationSolver->parameter.learningRateSequence =
        SharedPtr<NumericTable>(new HomogenNumericTable<>(1, 1, NumericTable::doAllocate, 0.001));

    /* Set the number of iterations to be done by the neural network */
    trainingNet.parameter.nIterations = 6000;

    /* Run the neural network training */
    trainingNet.compute();

    /* Get training and prediction models of the neural network */
    SharedPtr<training::Model> trainingModel = trainingNet.getResult()->get(model);
    services::SharedPtr<prediction::Model> predictionModel = trainingModel->getPredictionModel();

    /* Create an algorithm to compute the neural network predictions */
    prediction::Batch<> predictionNet;

    /* Set input objects for the prediction neural network */
    predictionNet.input.set(prediction::model, predictionModel);
    predictionNet.input.set(prediction::data, predictionData);

    /* Run the neural network prediction */
    predictionNet.compute();

    /* Print results of the neural network prediction */
    SharedPtr<Tensor> predictionResults = predictionNet.getResult()->get(prediction::prediction);

    printTensors<int, float>(predictionGroundTruth, predictionResults,
                             "Ground truth", "Neural network predictions: each class probability",
                             "Neural network classification results (first 20 observations):", 20);

    return 0;
}
