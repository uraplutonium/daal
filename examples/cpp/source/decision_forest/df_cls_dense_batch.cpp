/* file: df_cls_dense_batch.cpp */
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
!    C++ example of decision forest classification in the batch processing mode.
!
!    The program trains the decision forest classification model on a training
!    datasetFileName and computes classification for the test data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DF_CLS_DENSE_BATCH"></a>
 * \example df_cls_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::decision_forest::classification;

/* Input data set parameters */
const string trainDatasetFileName = "../data/batch/df_classification_train.csv";
const string testDatasetFileName  = "../data/batch/df_classification_test.csv";
const size_t categoricalFeaturesIndices[] = { 2 };
const size_t nFeatures  = 3;  /* Number of features in training and testing data sets */

/* Decision forest parameters */
const size_t nTrees = 10;
const size_t minObservationsInLeafNode = 8;

const size_t nClasses = 5;  /* Number of classes */

training::ResultPtr trainModel();
void testModel(const training::ResultPtr& res);
void loadData(const std::string& fileName, NumericTablePtr& pData, NumericTablePtr& pDependentVar);

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    training::ResultPtr trainingResult = trainModel();
    testModel(trainingResult);

    return 0;
}

training::ResultPtr trainModel()
{
    /* Create Numeric Tables for training data and dependent variables */
    NumericTablePtr trainData;
    NumericTablePtr trainDependentVariable;

    loadData(trainDatasetFileName, trainData, trainDependentVariable);

    /* Create an algorithm object to train the decision forest classification model */
    training::Batch<> algorithm(nClasses);

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainDependentVariable);

    algorithm.parameter.nTrees = nTrees;
    algorithm.parameter.featuresPerNode = nFeatures;
    algorithm.parameter.minObservationsInLeafNode = minObservationsInLeafNode;
    algorithm.parameter.varImportance = algorithms::decision_forest::training::MDI;
    algorithm.parameter.resultsToCompute = algorithms::decision_forest::training::computeOutOfBagError;

    /* Build the decision forest classification model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    training::ResultPtr trainingResult = algorithm.getResult();
    printNumericTable(trainingResult->get(training::variableImportance), "Variable importance results: ");
    printNumericTable(trainingResult->get(training::outOfBagError), "OOB error: ");
    return trainingResult;
}

void testModel(const training::ResultPtr& trainingResult)
{
    /* Create Numeric Tables for testing data and ground truth values */
    NumericTablePtr testData;
    NumericTablePtr testGroundTruth;

    loadData(testDatasetFileName, testData, testGroundTruth);

    /* Create an algorithm object to predict values of decision forest classification */
    prediction::Batch<> algorithm(nClasses);

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testData);
    algorithm.input.set(classifier::prediction::model, trainingResult->get(classifier::training::model));

    /* Predict values of decision forest classification */
    algorithm.compute();

    /* Retrieve the algorithm results */
    classifier::prediction::ResultPtr predictionResult = algorithm.getResult();
    printNumericTable(predictionResult->get(classifier::prediction::prediction),
        "Decision forest prediction results (first 10 rows):", 10);
    printNumericTable(testGroundTruth, "Ground truth (first 10 rows):", 10);
}

void loadData(const std::string& fileName, NumericTablePtr& pData, NumericTablePtr& pDependentVar)
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(fileName,
        DataSource::notAllocateNumericTable,
        DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and dependent variables */
    pData.reset(new HomogenNumericTable<>(nFeatures, 0, NumericTable::notAllocate));
    pDependentVar.reset(new HomogenNumericTable<>(1, 0, NumericTable::notAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(pData, pDependentVar));

    /* Retrieve the data from input file */
    trainDataSource.loadDataBlock(mergedData.get());

    NumericTableDictionaryPtr pDictionary = pData->getDictionarySharedPtr();
    for(size_t i = 0, n = sizeof(categoricalFeaturesIndices) / sizeof(categoricalFeaturesIndices[0]); i < n; ++i)
        (*pDictionary)[categoricalFeaturesIndices[i]].featureType = data_feature_utils::DAAL_CATEGORICAL;
}
