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
using namespace daal::algorithms::neural_networks;
using namespace daal::services;

/* Input data set parameters */
string trainDatasetFile     = "../data/batch/neural_network_train.csv";
string trainGroundTruthFile = "../data/batch/neural_network_train_ground_truth.csv";
string testDatasetFile      = "../data/batch/neural_network_test.csv";
string testGroundTruthFile  = "../data/batch/neural_network_test_ground_truth.csv";

services::SharedPtr<prediction::Model> predictionModel;
services::SharedPtr<prediction::Result> predictionResult;

void trainModel();
void testModel();
void printResults();

int main()
{
    trainModel();

    testModel();

    printResults();

    return 0;
}

void trainModel()
{
    /* Read training data set from a .csv file and create a tensor to store input data */
    SharedPtr<Tensor> trainingData = readTensorFromCSV(trainDatasetFile);
    SharedPtr<Tensor> trainingGroundTruth = readTensorFromCSV(trainGroundTruthFile);

    /* Create an algorithm to train neural network */
    training::Batch<> net;

    /* Set the batch size for the neural network training */
    net.parameter.batchSize = 10;

    /* Configure the neural network */
    Collection<LayerDescriptor> layersConfiguration = configureNet();
    net.initialize(trainingData->getDimensions(), layersConfiguration);

    /* Pass a training data set and dependent values to the algorithm */
    net.input.set(training::data, trainingData);
    net.input.set(training::groundTruth, trainingGroundTruth);


    SharedPtr<optimization_solver::sgd::Batch<float> > sgdAlgorithm(new optimization_solver::sgd::Batch<float>());

    float learningRate = 0.001f;
    sgdAlgorithm->parameter.learningRateSequence = services::SharedPtr<NumericTable>(new HomogenNumericTable<double>(1, 1, NumericTable::doAllocate, learningRate));

    /* Set learning rate for the optimization solver used in the neural network */
    net.parameter.optimizationSolver = sgdAlgorithm;


    /* Run the neural network training */
    net.compute();

    /* Retrieve training and prediction models of the neural network */
    SharedPtr<training::Model> trainingModel = net.getResult()->get(training::model);
    predictionModel = trainingModel->getPredictionModel<float>();
}

void testModel()
{
    /* Read testing data set from a .csv file and create a tensor to store input data */
    SharedPtr<Tensor> predictionData = readTensorFromCSV(testDatasetFile);

    /* Create an algorithm to compute the neural network predictions */
    prediction::Batch<> net;

    /* Set input objects for the prediction neural network */
    net.input.set(prediction::model, predictionModel);
    net.input.set(prediction::data, predictionData);

    /* Run the neural network prediction */
    net.compute();

    /* Print results of the neural network prediction */
    predictionResult = net.getResult();
}

void printResults()
{
    /* Read testing ground truth from a .csv file and create a tensor to store the data */
    SharedPtr<Tensor> predictionGroundTruth = readTensorFromCSV(testGroundTruthFile);

    printTensors<int, float>(predictionGroundTruth, predictionResult->get(prediction::prediction),
                             "Ground truth", "Neural network predictions: each class probability",
                             "Neural network classification results (first 20 observations):", 20);
}
