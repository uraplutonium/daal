/* file: neural_net_predict_dense_batch.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
!  Content:
!    C++ example of neural network scoring
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-NEURAL_NETWORK_PREDICTION_BATCH"></a>
 * \example neural_net_predict_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"
#include "neural_net_predict_dense_batch.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks;
using namespace daal::services;

/* Input data set parameters */
string testDatasetFile      = "../data/batch/neural_network_test.csv";
string testGroundTruthFile  = "../data/batch/neural_network_test_ground_truth.csv";

/* Weights and biases obtained on the training stage */
string fc1WeightsFile = "../data/batch/fc1_weights.csv";
string fc1BiasesFile  = "../data/batch/fc1_biases.csv";
string fc2WeightsFile = "../data/batch/fc2_weights.csv";
string fc2BiasesFile  = "../data/batch/fc2_biases.csv";

TensorPtr predictionData;
prediction::ModelPtr predictionModel;
prediction::ResultPtr predictionResult;

void createModel();
void testModel();
void printResults();

int main()
{
    createModel();

    testModel();

    printResults();

    return 0;
}

void createModel()
{
    /* Read testing data set from a .csv file and create a tensor to store input data */
    predictionData = readTensorFromCSV(testDatasetFile);

    /* Configure the neural network */
    LayerIds ids;
    prediction::TopologyPtr topology = configureNet(&ids);

    /* Create prediction model of the neural network */
    predictionModel = prediction::Model::create(*topology);
    checkPtr(predictionModel.get());

    /* Read 1st fully-connected layer weights and biases from CSV file */
    /* 1st fully-connected layer weights are a 2D tensor of size 5 x 20 */
    TensorPtr fc1Weights = readTensorFromCSV(fc1WeightsFile);
    /* 1st fully-connected layer biases are a 1D tensor of size 5 */
    TensorPtr fc1Biases = readTensorFromCSV(fc1BiasesFile);

    /* Set weights and biases of the 1st fully-connected layer */
    forward::Input *fc1Input = predictionModel->getLayer(ids.fc1)->getLayerInput();
    fc1Input->set(forward::weights, fc1Weights);
    fc1Input->set(forward::biases, fc1Biases);

    /* Set flag that specifies that weights and biases of the 1st fully-connected layer are initialized */
    predictionModel->getLayer(ids.fc1)->getLayerParameter()->weightsAndBiasesInitialized = true;

    /* Read 2nd fully-connected layer weights and biases from CSV file */
    /* 2nd fully-connected layer weights are a 2D tensor of size 2 x 5 */
    TensorPtr fc2Weights = readTensorFromCSV(fc2WeightsFile);
    /* 2nd fully-connected layer biases are a 1D tensor of size 2 */
    TensorPtr fc2Biases = readTensorFromCSV(fc2BiasesFile);

    /* Set weights and biases of the 2nd fully-connected layer */
    forward::Input *fc2Input = predictionModel->getLayer(ids.fc2)->getLayerInput();
    fc2Input->set(forward::weights, fc2Weights);
    fc2Input->set(forward::biases, fc2Biases);

    /* Set flag that specifies that weights and biases of the 2nd fully-connected layer are initialized */
    predictionModel->getLayer(ids.fc2)->getLayerParameter()->weightsAndBiasesInitialized = true;
}

void testModel()
{
    /* Create an algorithm to compute the neural network predictions */
    prediction::Batch<> net;

    /* Set parameters for the prediction neural network */
    net.parameter.batchSize = predictionData->getDimensionSize(0);

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
    TensorPtr predictionGroundTruth = readTensorFromCSV(testGroundTruthFile);

    printTensors<int, float>(predictionGroundTruth, predictionResult->get(prediction::prediction),
                             "Ground truth", "Neural network predictions: each class probability",
                             "Neural network classification results (first 20 observations):", 20);
}
