/* file: kdtree_knn_classification_dense_distributed_mpi.cpp */
/*******************************************************************************
* Copyright 2017 Intel Corporation
* All Rights Reserved.
*
* If this  software was obtained  under the  Intel Simplified  Software License,
* the following terms apply:
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*
*
* If this  software  was obtained  under the  Apache License,  Version  2.0 (the
* "License"), the following terms apply:
*
* You may  not use this  file except  in compliance  with  the License.  You may
* obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
*
*
* Unless  required  by   applicable  law  or  agreed  to  in  writing,  software
* distributed under the License  is distributed  on an  "AS IS"  BASIS,  WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*
* See the   License  for the   specific  language   governing   permissions  and
* limitations under the License.
*******************************************************************************/

/*
!  Content:
!    C++ sample of multiple linear regression in the distributed processing
!    mode.
!
!    The program trains the multiple linear regression model on a training
!    data set with the normal equations method and computes regression for the
!    test data.
!******************************************************************************/

/**
 * <a name="DAAL-SAMPLE-CPP-KDTREE_KNN_CLASSIFICATION_DENSE_DISTRIBUTED"></a>
 * \example kdtree_knn_classification_dense_distributed_mpi.cpp
 */

#include <mpi.h>
#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms::kdtree_knn_classification;

#define TIME_MEASUREMENT
#define BINARY_DATASET

int rankId, comm_size;
const int mpi_root = 0;

#ifdef BINARY_DATASET
const size_t nClasses             = 5;
#else
const size_t nClasses             = 5;
const size_t nBlocks              = 4;
const size_t nFeatures            = 5;
#endif

#ifdef BINARY_DATASET
const char * const defaultTrainDatasetFileName            = "./data/distributed/knn_train_data.bin";
const char * const defaultTrainDatasetGroundTruthFileName = "./data/distributed/knn_train_labels.bin";
const char * const defaultTestDatasetFileName             = "./data/distributed/knn_test_data.bin";
const char * const defaultTestDatasetGroundTruthFileName  = "./data/distributed/knn_test_labels.bin";

const char * trainDatasetFileName                         = defaultTrainDatasetFileName;
const char * trainDatasetGroundTruthFileName              = defaultTrainDatasetGroundTruthFileName;
const char * testDatasetFileName                          = defaultTestDatasetFileName;
const char * testDatasetGroundTruthFileName               = defaultTestDatasetGroundTruthFileName;
#else
const string trainDatasetFileNames[] =
{
    "./data/distributed/k_nearest_neighbors_train_1.csv",
    "./data/distributed/k_nearest_neighbors_train_2.csv",
    "./data/distributed/k_nearest_neighbors_train_3.csv",
    "./data/distributed/k_nearest_neighbors_train_4.csv"
};

const string testDatasetFileNames[] =
{
    "./data/distributed/k_nearest_neighbors_test_1.csv",
    "./data/distributed/k_nearest_neighbors_test_2.csv",
    "./data/distributed/k_nearest_neighbors_test_3.csv",
    "./data/distributed/k_nearest_neighbors_test_4.csv"
};
#endif

NumericTablePtr trainData;
NumericTablePtr trainGroundTruth;
services::SharedPtr<training::DistributedPartialResultStep1> trainingPartialResults1;
services::SharedPtr<training::DistributedPartialResultStep2> trainingPartialResults2;
services::SharedPtr<training::DistributedPartialResultStep3> trainingPartialResults3;
services::SharedPtr<training::DistributedPartialResultStep4> trainingPartialResults4;
services::SharedPtr<training::DistributedPartialResultStep5> trainingPartialResults5;
services::SharedPtr<training::DistributedPartialResultStep6> trainingPartialResults6;
services::SharedPtr<training::DistributedPartialResultStep7> trainingPartialResults7;
services::SharedPtr<training::DistributedPartialResultStep8> trainingPartialResults8;
services::SharedPtr<prediction::DistributedPartialResultStep1> predictionPartialResults1;
services::SharedPtr<prediction::DistributedPartialResultStep2> predictionPartialResults2;
NumericTablePtr testData, arrangedTestData;
NumericTablePtr testGroundTruth, arrangedTestGroundTruth;

void trainModel();
void testModel();

template <typename T>
void gather(const ByteBuffer & nodeBuffer, T * result);

template <typename T>
void allgather(const ByteBuffer & nodeBuffer, T * result, MPI_Comm comm = MPI_COMM_WORLD, int commSize = 0);

template <typename T>
void broadcast(ByteBuffer & nodeBuffer, services::SharedPtr<T> & result);

template <typename T>
void sendrecv(const ByteBuffer & nodeBuffer, services::SharedPtr<T> & result, int partnerRankID);

void sendrecv(const data_management::NumericTableConstPtr & data, const data_management::NumericTableConstPtr & labels,
              const data_management::NumericTableConstPtr & keys, const data_management::NumericTablePtr & arrangedData,
              const data_management::NumericTablePtr & arrangedLabels);

template <typename T>
inline void recv(T * buffer, size_t length, int source, int tag, MPI_Request * & req, size_t & reqCount, size_t & reqCapacity);

template <typename T>
inline void send(const T * buffer, size_t length, int target, int tag, MPI_Request * & req, size_t & reqCount, size_t & reqCapacity);

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);

#ifdef TIME_MEASUREMENT
    std::cout << "Number of threads for DAAL: " << services::Environment::getInstance()->getNumberOfThreads() << std::endl;
#endif

#ifdef BINARY_DATASET
    if (argc == 5)
    {
        trainDatasetFileName = argv[1];
        trainDatasetGroundTruthFileName = argv[2];
        testDatasetFileName = argv[3];
        testDatasetGroundTruthFileName = argv[4];
    }
#endif
    trainModel();
    testModel();

    MPI_Finalize();

    return 0;
}

void trainModel()
{
#ifdef TIME_MEASUREMENT
    double funcStart = MPI_Wtime();
    double start =  MPI_Wtime();
#endif

#ifdef BINARY_DATASET
    DAAL_INT64 observationCount;
    DAAL_INT64 featureCount;

    FILE * f = fopen(trainDatasetFileName, "rb");
    if (!f)
    {
        std::cout << "Training file opening failed!" << std::endl;
        return;
    }
    const size_t readCount1 = fread(&observationCount, sizeof(observationCount), 1, f);
    if (readCount1 != 1)
    {
        std::cout << "Reading number of observations from training file opening failed!" << std::endl;
        return;
    }
    const size_t readCount2 = fread(&featureCount, sizeof(featureCount), 1, f);
    if (readCount2 != 1)
    {
        std::cout << "Reading number of features from training file opening failed!" << std::endl;
        return;
    }
    const size_t rowsPerRank = (observationCount + comm_size - 1) / comm_size;
    const size_t firstDataSetRow = rankId * rowsPerRank;
    const size_t lastDataSetRow = (firstDataSetRow + rowsPerRank > observationCount) ? observationCount : firstDataSetRow + rowsPerRank;

    services::SharedPtr<SOANumericTable> dataTable(new SOANumericTable(featureCount, lastDataSetRow - firstDataSetRow, DictionaryIface::equal));
    dataTable->getDictionarySharedPtr()->setAllFeatures<float>();
    dataTable->resize(dataTable->getNumberOfRows()); // Just to allocate memory.
    services::SharedPtr<SOANumericTable> labelsTable(new SOANumericTable(1, lastDataSetRow - firstDataSetRow));
    labelsTable->getDictionarySharedPtr()->setAllFeatures<int>();
    labelsTable->resize(labelsTable->getNumberOfRows()); // Just to allocate memory.

    for (size_t d = 0; d < featureCount; ++d)
    {
        if (firstDataSetRow > 0)
        {
            const int seekResult = fseek(f, firstDataSetRow * sizeof(float), SEEK_CUR);
            if (seekResult)
            {
                std::cout << "Seek in training file failed" << std::endl;
                return;
            }
        }
        float * const ptr = static_cast<float *>(dataTable->getArray(d));
        const size_t readCount = fread(ptr, sizeof(float), lastDataSetRow - firstDataSetRow, f);
        if (readCount != lastDataSetRow - firstDataSetRow)
        {
            std::cout << "ERROR: Only " << readCount << " of " << (lastDataSetRow - firstDataSetRow) << " read for feature " << d << std::endl;
            return;
        }
        if (lastDataSetRow != observationCount)
        {
            const int seekResult = fseek(f, (observationCount - lastDataSetRow) * sizeof(float), SEEK_CUR);
            if (seekResult)
            {
                std::cout << "Seek in training file failed" << std::endl;
                return;
            }
        }
    }

    fclose(f); f = 0;

#ifdef TIME_MEASUREMENT
    std::cout << "Reading the data completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif

    {
        DAAL_INT64 observationCount2;
        DAAL_INT64 featureCount2;
        f = fopen(trainDatasetGroundTruthFileName, "rb");
        if (!f)
        {
            std::cout << "Training file with labels opening failed!" << std::endl;
            return;
        }
        const size_t readCount1 = fread(&observationCount2, sizeof(observationCount2), 1, f);
        if (readCount1 != 1)
        {
            std::cout << "Reading number of observations from training file with labels opening failed!" << std::endl;
            return;
        }

        const size_t readCount2 = fread(&featureCount2, sizeof(featureCount2), 1, f);
        if (readCount2 != 1)
        {
            std::cout << "Reading number of features from training file with labels opening failed!" << std::endl;
            return;
        }

        if (observationCount != observationCount2)
        {
            std::cout << "Training data and labels must have equal number of rows!" << std::endl;
        }
        if (featureCount2 != 1)
        {
            std::cout << "Training labels file must contain exactly one column!" << std::endl;
        }

        if (firstDataSetRow > 0)
        {
            const int seekResult = fseek(f, firstDataSetRow * sizeof(float), SEEK_CUR);
            if (seekResult)
            {
                std::cout << "Seek in training file with labels failed" << std::endl;
            }
        }
        int * const ptr = static_cast<int *>(labelsTable->getArray(0));
        vector<float> tempLabels(lastDataSetRow - firstDataSetRow);
        float * const tempLabelsPtr = &tempLabels[0];
        const size_t readCount = fread(tempLabelsPtr, sizeof(float), lastDataSetRow - firstDataSetRow, f);
        if (readCount != lastDataSetRow - firstDataSetRow)
        {
            std::cout << "ERROR: Only " << readCount << " of " << (lastDataSetRow - firstDataSetRow) << " read of class labels" << std::endl;
            return;
        }
        for (size_t i = 0, cnt = lastDataSetRow - firstDataSetRow; i != cnt; ++i)
        {
            ptr[i] = tempLabelsPtr[i];
        }

        fclose(f); f = 0;
    }

#ifdef TIME_MEASUREMENT
    std::cout << "Reading the training labels completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif

    trainData = dataTable;
    trainGroundTruth = labelsTable;
#else // #ifdef BINARY_DATASET
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileNames[rankId],
                                                      DataSource::notAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and labels */
    trainData.reset(new HomogenNumericTable<double>(nFeatures, 0, NumericTable::notAllocate));
    trainGroundTruth.reset(new HomogenNumericTable<double>(1, 0, NumericTable::notAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(trainData, trainGroundTruth));

    /* Retrieve the data from the input file */
    trainDataSource.loadDataBlock(mergedData.get());

#ifdef TIME_MEASUREMENT
    std::cout << "Reading the data completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif

#endif // #ifdef BINARY_DATASET

#ifdef TIME_MEASUREMENT
    double trainStart = MPI_Wtime();
#endif

    /* Create an algorithm object to train the KD-tree based kNN model */
    training::Distributed<step1Local> algStep1;

    /* Pass the training data set and dependent values to the algorithm */
    algStep1.input.set(algorithms::classifier::training::data, trainData);
    algStep1.input.set(algorithms::classifier::training::labels, trainGroundTruth);

    /* Train the KD-tree based kNN model */
    algStep1.compute();

#ifdef TIME_MEASUREMENT
    std::cout << "1st step of training completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif

    /* Retrieve the results of the training algorithm  */
    trainingPartialResults1 = algStep1.getPartialResult();

    ByteBuffer buffer;

    /* Serialize partial results required by step 2 */
    serializeDAALObject(trainingPartialResults1.get(), buffer);

    /* Send data from all computation nodes to root computation node */
    services::SharedPtr<training::DistributedPartialResultStep1> * trainingPartialResults1OnMaster =
        new services::SharedPtr<training::DistributedPartialResultStep1>[comm_size];
    gather(buffer, trainingPartialResults1OnMaster);

#ifdef TIME_MEASUREMENT
    std::cout << "Communication after 1st step of training completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif

    if (rankId == mpi_root)
    {
        /* Create an algorithm object to train the KD-tree based kNN model */
        training::Distributed<step2Master> algStep2;

        /* Pass the training data set and dependent values to the algorithm */
        for (size_t i = 0; i < comm_size; ++i)
        {
            algStep2.input.add(training::inputOfStep2, i, trainingPartialResults1OnMaster[i]->get(training::boundingBoxes));
        }

        /* Train the KD-tree based kNN model */
        algStep2.compute();

        /* Retrieve the results of the training algorithm  */
        trainingPartialResults2 = algStep2.getPartialResult();

        /* Serialize partial results required by step 3 */
        serializeDAALObject(trainingPartialResults2.get(), buffer);
    }

#ifdef TIME_MEASUREMENT
    std::cout << "2nd step of training completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif

    delete[] trainingPartialResults1OnMaster;

    /* Send data from root computation node to all computation nodes */
    broadcast(buffer, trainingPartialResults2);

#ifdef TIME_MEASUREMENT
    std::cout << "Communication after 2nd step of training completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif

    NumericTablePtr boundingBoxes = trainingPartialResults2->get(training::globalBoundingBoxes);
    NumericTablePtr cx = trainData, cy = trainGroundTruth;
    NumericTablePtr * nt = new NumericTablePtr[comm_size];
    NumericTablePtr * nt2 = new NumericTablePtr[comm_size];
    const int loops = trainingPartialResults2->get(training::numberOfLoops)->getValue<int>(0, 0);
    for (int loop = 0; loop < loops; ++loop)
    {
        /* Create an algorithm object to train the KD-tree based kNN model */
        training::Distributed<step3Local> algStep3;

        /* Pass the training data set and dependent values to the algorithm */
        algStep3.input.set(training::dataForStep3, cx);
        algStep3.input.set(training::labelsForStep3, cy);
        algStep3.input.set(training::boundingBoxesForStep3, boundingBoxes);
        algStep3.input.set(training::numberOfLoopsForStep3, trainingPartialResults2->get(training::numberOfLoops));
        algStep3.input.set(training::loopNumberForStep3, loop);
        algStep3.input.set(training::nodeIndexForStep3, rankId);
        algStep3.input.set(training::nodeCountForStep3, comm_size);

        /* Train the KD-tree based kNN model */
        algStep3.compute();

#ifdef TIME_MEASUREMENT
    std::cout << "3rd step of training completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif

        /* Retrieve the results of the training algorithm  */
        trainingPartialResults3 = algStep3.getPartialResult();

        /* Split communication world by color */
        const int color = trainingPartialResults3->get(training::color)->getValue<int>(0, 0);
        MPI_Comm newComm;
        MPI_Comm_split(MPI_COMM_WORLD, color, rankId, &newComm);
        int newCommSize;
        MPI_Comm_size(newComm, &newCommSize);

        /* Serialize partial results required by step 4 */
        serializeDAALObject(trainingPartialResults3->get(training::localSamples).get(), buffer);

        /* Send data from all computation nodes to all computation nodes in newComm */
        allgather(buffer, nt, newComm, newCommSize);

#ifdef TIME_MEASUREMENT
    std::cout << "Communication after 3rd step of training completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif

        /* Create an algorithm object to train the KD-tree based kNN model */
        training::Distributed<step4Local> algStep4;

        /* Pass the training data set and dependent values to the algorithm */
        algStep4.input.set(training::dataForStep4, cx);
        algStep4.input.set(training::labelsForStep4, cy);
        algStep4.input.set(training::dimensionForStep4, trainingPartialResults3->get(training::dimension));
        algStep4.input.set(training::boundingBoxesForStep4, boundingBoxes);
        for (int j = 0; j < newCommSize; ++j)
        {
            algStep4.input.add(training::samplesForStep4, j, nt[j]);
        }

        /* Train the KD-tree based kNN model */
        algStep4.compute();

#ifdef TIME_MEASUREMENT
    std::cout << "4th step of training completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif

        /* Retrieve the results of the training algorithm  */
        trainingPartialResults4 = algStep4.getPartialResult();

        /* Serialize partial results required by step 5 */
        serializeDAALObject(trainingPartialResults4->get(training::localHistogram).get(), buffer);

        /* Send data from all computation nodes to all computation nodes in newComm */
        allgather(buffer, nt, newComm, newCommSize);

#ifdef TIME_MEASUREMENT
    std::cout << "Communication after 4th step of training completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif

        const int partnerRankID = rankId ^ (comm_size / (1 << (loop + 1)));

        /* Create an algorithm object to train the KD-tree based kNN model */
        training::Distributed<step5Local> algStep5;

        /* Pass the training data set and dependent values to the algorithm */
        algStep5.input.set(training::dataForStep5, cx);
        algStep5.input.set(training::labelsForStep5, cy);
        algStep5.input.set(training::dimensionForStep5, trainingPartialResults3->get(training::dimension));
        algStep5.input.set(training::isPartnerGreaterForStep5, rankId < partnerRankID);
        for (int j = 0; j < newCommSize; ++j)
        {
            algStep5.input.add(training::histogramForStep5, j, nt[j]);
        }

        /* Train the KD-tree based kNN model */
        algStep5.compute();

#ifdef TIME_MEASUREMENT
    std::cout << "5th step of training completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif

        /* Retrieve the results of the training algorithm  */
        trainingPartialResults5 = algStep5.getPartialResult();

        /* Serialize partial results required by step 6 */
        serializeDAALObject(trainingPartialResults5->get(training::dataForPartner).get(), buffer);

        /* Send data to partner computation node */
        NumericTablePtr dataFromPartner;
        sendrecv(buffer, dataFromPartner, partnerRankID);

        /* Serialize partial results required by step 6 */
        serializeDAALObject(trainingPartialResults5->get(training::labelsForPartner).get(), buffer);

        /* Send labels to partner computation node */
        NumericTablePtr labelsFromPartner;
        sendrecv(buffer, labelsFromPartner, partnerRankID);

#ifdef TIME_MEASUREMENT
    std::cout << "Communication after 5th step of training completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif

        /* Create an algorithm object to train the KD-tree based kNN model */
        training::Distributed<step6Local> algStep6;

        /* Pass the training data set and dependent values to the algorithm */
        algStep6.input.set(training::dataForStep6, cx);
        algStep6.input.set(training::labelsForStep6, cy);
        algStep6.input.set(training::dataFromPartnerForStep6, dataFromPartner);
        algStep6.input.set(training::labelsFromPartnerForStep6, labelsFromPartner);
        algStep6.input.set(training::markersForStep6, trainingPartialResults5->get(training::markers));

        /* Train the KD-tree based kNN model */
        algStep6.compute();

#ifdef TIME_MEASUREMENT
    std::cout << "6th step of training completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif

        /* Retrieve the results of the training algorithm  */
        trainingPartialResults6 = algStep6.getPartialResult();
        cx = trainingPartialResults6->get(training::concatenatedData);
        cy = trainingPartialResults6->get(training::concatenatedLabels);

        /* Serialize partial results required by step 7 */
        serializeDAALObject(trainingPartialResults3->get(training::dimension).get(), buffer);

        /* Send dimension from all computation nodes to root computation node */
        gather(buffer, nt);

        /* Serialize partial results required by step 7 */
        serializeDAALObject(trainingPartialResults5->get(training::median).get(), buffer);

        /* Send dimension from all computation nodes to root computation node */
        gather(buffer, nt2);

#ifdef TIME_MEASUREMENT
    std::cout << "Communication after 6th step of training completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif

        if (rankId == mpi_root)
        {
            /* Create an algorithm object to train the KD-tree based kNN model */
            training::Distributed<step7Master> algStep7;

            /* Pass the training data set and dependent values to the algorithm */
            algStep7.input.set(training::boundingBoxesForStep7, boundingBoxes);
            algStep7.input.set(training::numberOfLoopsForStep7, trainingPartialResults2->get(training::numberOfLoops));
            algStep7.input.set(training::partialModelForStep7,
                               trainingPartialResults7 ? trainingPartialResults7->get(training::partialModelOfStep7) :
                                   services::SharedPtr<PartialModel>());
            algStep7.input.set(training::loopNumberForStep7, loop);

            for (size_t i = 0; i < comm_size; ++i)
            {
                algStep7.input.add(training::dimensionForStep7, i, nt[i]);
                algStep7.input.add(training::medianForStep7, i, nt2[i]);
            }

            /* Train the KD-tree based kNN model */
            algStep7.compute();

#ifdef TIME_MEASUREMENT
    std::cout << "7th step of training completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif

            /* Retrieve the results of the training algorithm  */
            trainingPartialResults7 = algStep7.getPartialResult();

            /* Serialize partial results required by step 3 */
            serializeDAALObject(trainingPartialResults7.get(), buffer);
        }

        /* Send data from root computation node to all computation nodes */
        broadcast(buffer, trainingPartialResults7);
        boundingBoxes = trainingPartialResults7->get(training::boundingBoxesOfStep7ForStep3);

#ifdef TIME_MEASUREMENT
    std::cout << "Communication after 7th step of training completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif
    }

    delete[] nt;
    delete[] nt2;

    /* Create an algorithm object to train the KD-tree based kNN model */
    training::Distributed<step8Local> algStep8;

    /* Pass the training data set and dependent values to the algorithm */
    algStep8.input.set(training::dataForStep8, cx);
    algStep8.input.set(training::labelsForStep8, cy);
    algStep8.input.set(training::partialModelForStep8, trainingPartialResults7->get(training::partialModelOfStep7));

    /* Train the KD-tree based kNN model */
    algStep8.compute();

    /* Retrieve the results of the training algorithm  */
    trainingPartialResults8 = algStep8.getPartialResult();

#ifdef TIME_MEASUREMENT
    std::cout << "8th step of training completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif

#ifdef TIME_MEASUREMENT
    std::cout << "Training completed in " << MPI_Wtime() - trainStart << " seconds" << std::endl;
    std::cout << "Training function completed in " << MPI_Wtime() - funcStart << " seconds" << std::endl;
#endif
}

void testModel()
{
#ifdef TIME_MEASUREMENT
    double funcStart = MPI_Wtime();
    double start = MPI_Wtime();
#endif

    const size_t k = 10;

#ifdef BINARY_DATASET
    DAAL_INT64 observationCount;
    DAAL_INT64 featureCount;

    FILE * f = fopen(testDatasetFileName, "rb");
    if (!f)
    {
        std::cout << "Testing file opening failed!" << std::endl;
        return;
    }

    const size_t readCount1 = fread(&observationCount, sizeof(observationCount), 1, f);
    if (readCount1 != 1)
    {
        std::cout << "Reading number of observations from testing file opening failed!" << std::endl;
        return;
    }
    const size_t readCount2 = fread(&featureCount, sizeof(featureCount), 1, f);
    if (readCount2 != 1)
    {
        std::cout << "Reading number of features from testing file opening failed!" << std::endl;
        return;
    }
    const size_t rowsPerRank = (observationCount + comm_size - 1) / comm_size;
    const size_t firstDataSetRow = rankId * rowsPerRank;
    const size_t lastDataSetRow = (firstDataSetRow + rowsPerRank > observationCount) ? observationCount : firstDataSetRow + rowsPerRank;

    services::SharedPtr<SOANumericTable> dataTable(new SOANumericTable(featureCount, lastDataSetRow - firstDataSetRow, DictionaryIface::equal));
    dataTable->getDictionarySharedPtr()->setAllFeatures<float>();
    dataTable->resize(dataTable->getNumberOfRows()); // Just to allocate memory.
    services::SharedPtr<SOANumericTable> labelsTable(new SOANumericTable(1, lastDataSetRow - firstDataSetRow));
    labelsTable->getDictionarySharedPtr()->setAllFeatures<int>();
    labelsTable->resize(labelsTable->getNumberOfRows()); // Just to allocate memory.

    for (size_t d = 0; d < featureCount; ++d)
    {
        if (firstDataSetRow > 0)
        {
            const int seekResult = fseek(f, firstDataSetRow * sizeof(float), SEEK_CUR);
            if (seekResult)
            {
                std::cout << "Seek in testing file failed" << std::endl;
                return;
            }
        }
        float * const ptr = static_cast<float *>(dataTable->getArray(d));
        const size_t readCount = fread(ptr, sizeof(float), lastDataSetRow - firstDataSetRow, f);
        if (readCount != lastDataSetRow - firstDataSetRow)
        {
            std::cout << "ERROR: Only " << readCount << " of " << (lastDataSetRow - firstDataSetRow) << " read for feature " << d << std::endl;
            return;
        }
        if (lastDataSetRow != observationCount)
        {
            const int seekResult = fseek(f, (observationCount - lastDataSetRow) * sizeof(float), SEEK_CUR);
            if (seekResult)
            {
                std::cout << "Seek in testing file failed" << std::endl;
                return;
            }
        }
    }

    fclose(f); f = 0;
#ifdef TIME_MEASUREMENT
    std::cout << "Reading the testing data completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif

    {
        DAAL_INT64 observationCount2;
        DAAL_INT64 featureCount2;
        f = fopen(testDatasetGroundTruthFileName, "rb");
        if (!f)
        {
            std::cout << "Testing file with labels opening failed!" << std::endl;
            return;
        }
        fread(&observationCount2, sizeof(observationCount2), 1, f);
        fread(&featureCount2, sizeof(featureCount2), 1, f);

        if (observationCount != observationCount2)
        {
            std::cout << "Testing data and labels must have equal number of rows!" << std::endl;
            return;
        }
        if (featureCount2 != 1)
        {
            std::cout << "Testing labels file must contain exactly one column!" << std::endl;
        }

        if (firstDataSetRow > 0)
        {
            const int seekResult = fseek(f, firstDataSetRow * sizeof(float), SEEK_CUR);
        }
        int * const ptr = static_cast<int *>(labelsTable->getArray(0));
        vector<float> tempLabels(lastDataSetRow - firstDataSetRow);
        float * const tempLabelsPtr = &tempLabels[0];
        const size_t readCount = fread(tempLabelsPtr, sizeof(float), lastDataSetRow - firstDataSetRow, f);
        if (readCount != lastDataSetRow - firstDataSetRow)
        {
            std::cout << "ERROR: Only " << readCount << " of " << (lastDataSetRow - firstDataSetRow) << " read of class labels" << std::endl;
            return;
        }
        for (size_t i = 0, cnt = lastDataSetRow - firstDataSetRow; i != cnt; ++i)
        {
            ptr[i] = tempLabelsPtr[i];
        }
        fclose(f); f = 0;
    }

#ifdef TIME_MEASUREMENT
    std::cout << "Reading the testing labels completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif

    testData = dataTable;
    testGroundTruth = labelsTable;

    arrangedTestData = NumericTablePtr(new HomogenNumericTable<float>(featureCount, 0, NumericTable::notAllocate));
    arrangedTestGroundTruth = NumericTablePtr(new HomogenNumericTable<float>(1, 0, NumericTable::notAllocate));
#else // #ifdef BINARY_DATASET
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileNames[rankId],
                                                     DataSource::notAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and labels */
    testData = NumericTablePtr(new HomogenNumericTable<double>(nFeatures, 0, NumericTable::notAllocate));
    arrangedTestData = NumericTablePtr(new HomogenNumericTable<double>(nFeatures, 0, NumericTable::notAllocate));
    testGroundTruth = NumericTablePtr(new HomogenNumericTable<double>(1, 0, NumericTable::notAllocate));
    arrangedTestGroundTruth = NumericTablePtr(new HomogenNumericTable<double>(1, 0, NumericTable::notAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(testData, testGroundTruth));

    /* Retrieve the data from input file */
    testDataSource.loadDataBlock(mergedData.get());

#ifdef TIME_MEASUREMENT
    std::cout << "Reading the test data completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif
#endif // #ifdef BINARY_DATASET

#ifdef TIME_MEASUREMENT
    double testStart = MPI_Wtime();
#endif

    /* Create algorithm objects for KD-tree based kNN prediction with the default method */
    prediction::Distributed<step1Local> algStep1;
    algStep1.parameter.k = k;

    /* Pass the testing data set and trained model to the algorithm */
    algStep1.input.set(prediction::data, testData);
    algStep1.input.set(prediction::partialModel, trainingPartialResults8->get(training::partialModel));

    /* Compute prediction results */
    algStep1.compute();

    /* Retrieve algorithm results */
    predictionPartialResults1 = algStep1.getPartialResult();

#ifdef TIME_MEASUREMENT
    std::cout << "1th step of testing completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif

    sendrecv(testData, testGroundTruth, predictionPartialResults1->get(prediction::keys), arrangedTestData, arrangedTestGroundTruth);

#ifdef TIME_MEASUREMENT
    std::cout << "Communication after 1st step of testing completed in " << MPI_Wtime() - start << " seconds" << std::endl;
    start = MPI_Wtime();
#endif

    NumericTablePtr selfResp, interm;
    NumericTablePtr * const resp = new NumericTablePtr[comm_size];
    NumericTablePtr * const queries = new NumericTablePtr[comm_size];
    ByteBuffer buffer;
    for (size_t round = 0;; ++round)
    {
        /* Create algorithm objects for KD-tree based kNN prediction with the default method */
        prediction::Distributed<step2Local> algStep2;
        algStep2.parameter.k = k;

        /* Pass the testing data set and trained model to the algorithm */
        algStep2.input.set(prediction::arrangedData, arrangedTestData);
        algStep2.input.set(prediction::intermediatePrediction, interm);
        algStep2.input.set(prediction::key, rankId);
        algStep2.input.set(prediction::round, round);
        algStep2.input.set(prediction::partialModel, trainingPartialResults8->get(training::partialModel));
        for (size_t j = 0; j < comm_size; ++j)
        {
            algStep2.input.add(prediction::communicationResponses, j, resp[j]);
            if (j != rankId)
            {
                algStep2.input.add(prediction::communicationInputQueries, j, queries[j]);
            }
        }

        /* Compute prediction results */
        algStep2.compute();

#ifdef TIME_MEASUREMENT
        std::cout << "2nd step of testing completed in " << MPI_Wtime() - start << " seconds in round " << round
            << std::endl;
        start = MPI_Wtime();
#endif

        /* Retrieve algorithm results */
        predictionPartialResults2 = algStep2.getPartialResult();

        interm = predictionPartialResults2->get(prediction::prediction);

        for (size_t j = 0; j < comm_size; ++j)
        {
            resp[j] = predictionPartialResults2->get(prediction::communicationOutputResponses, j);
            queries[j] = predictionPartialResults2->get(prediction::communicationQueries, j);
        }

        swap(resp[rankId], selfResp);

        int breakConditionLocal = 1;
        if (selfResp && selfResp->getNumberOfRows() > 0)
        {
            breakConditionLocal = 0;
        }
        else
        {
            for (size_t j = 0; j < comm_size; ++j)
            {
                if ((resp[j] && resp[j]->getNumberOfRows() > 0) || (queries[j] && queries[j]->getNumberOfRows() > 0))
                {
                    breakConditionLocal = 0;
                    break;
                }
            }
        }

        int breakConditionGlobal;
        MPI_Allreduce(&breakConditionLocal, &breakConditionGlobal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (breakConditionGlobal == comm_size)
        {
#ifdef TIME_MEASUREMENT
            std::cout << "Communication after 2nd step of testing completed in " << MPI_Wtime() - start
                << " seconds in round " << round << std::endl;
            start = MPI_Wtime();
#endif
            break;
        }

        for (size_t j = 0; j < comm_size; ++j)
        {
            if (j != rankId)
            {
                /* Serialize partial results required by step 2 in next round */
                serializeDAALObject(resp[j].get(), buffer);

                /* Send data to appropriate computation node */
                sendrecv(buffer, resp[j], j);

                /* Serialize partial results required by step 2 in next round */
                serializeDAALObject(queries[j].get(), buffer);

                /* Send data to appropriate computation node */
                sendrecv(buffer, queries[j], j);
            }
        }

#ifdef TIME_MEASUREMENT
        std::cout << "Communication after 2nd step of testing completed in " << MPI_Wtime() - start << " seconds in round "
            << round << std::endl;
        start = MPI_Wtime();
#endif

    }

    delete[] queries;
    delete[] resp;

#ifdef TIME_MEASUREMENT
    std::cout << "Testing completed in " << MPI_Wtime() - testStart << " seconds" << std::endl;
    std::cout << "Testing function completed in " << MPI_Wtime() - funcStart << " seconds" << std::endl;
#endif
}

template <typename T>
void gather(const ByteBuffer & nodeBuffer, T * result)
{
    int nodeBufferSize = nodeBuffer.size();
    int * nodeBufferSizes = new int[comm_size];
    int * displs = new int[comm_size];

    MPI_Gather(&nodeBufferSize, sizeof(nodeBufferSize), MPI_CHAR, nodeBufferSizes, sizeof(nodeBufferSizes[0]), MPI_CHAR, mpi_root, MPI_COMM_WORLD);

    ByteBuffer::pointer recvPtr = 0;
    ByteBuffer recvBuffer;
    if (rankId == mpi_root)
    {
        size_t size = 0;
        for (size_t i = 0; i < comm_size; ++i)
        {
            displs[i] = size;
            size += nodeBufferSizes[i];
        }
        recvBuffer.resize(size);
        recvPtr = &(recvBuffer[0]);
    }

    MPI_Gatherv(&(nodeBuffer[0]), nodeBufferSize, MPI_CHAR, recvPtr, nodeBufferSizes, displs, MPI_CHAR, mpi_root, MPI_COMM_WORLD);

    if (rankId == mpi_root)
    {
        for (size_t i = 0; i < comm_size; ++i)
        {
            result[i] = result[i]->cast(deserializeDAALObject(recvPtr + displs[i], nodeBufferSizes[i]));
        }
    }

    delete[] displs;
    delete[] nodeBufferSizes;
}

template <typename T>
void allgather(const ByteBuffer & nodeBuffer, T * result, MPI_Comm comm, int commSize)
{
    if (commSize == 0)
        MPI_Comm_size(comm, &commSize);
    int nodeBufferSize = nodeBuffer.size();
    int * nodeBufferSizes = new int[commSize];
    int * displs = new int[commSize];

    MPI_Allgather(&nodeBufferSize, sizeof(nodeBufferSize), MPI_CHAR, nodeBufferSizes, sizeof(nodeBufferSizes[0]), MPI_CHAR, comm);

    size_t size = 0;
    for (int i = 0; i < commSize; ++i)
    {
        displs[i] = size;
        size += nodeBufferSizes[i];
    }
    ByteBuffer recvBuffer(size);

    MPI_Allgatherv(&(nodeBuffer[0]), nodeBufferSize, MPI_CHAR, &(recvBuffer[0]), nodeBufferSizes, displs, MPI_CHAR, comm);

    for (int i = 0; i < commSize; ++i)
    {
        result[i] = result[i]->cast(deserializeDAALObject(&(recvBuffer[0]) + displs[i], nodeBufferSizes[i]));
    }

    delete[] displs;
    delete[] nodeBufferSizes;
}

template <typename T>
void broadcast(ByteBuffer & nodeBuffer, services::SharedPtr<T> & result)
{
    int nodeBufferSize = 0;
    if (rankId == mpi_root)
    {
        nodeBufferSize = nodeBuffer.size();
    }
    MPI_Bcast(&nodeBufferSize, sizeof(nodeBufferSize), MPI_CHAR, mpi_root, MPI_COMM_WORLD);
    if (rankId != mpi_root)
    {
        nodeBuffer.resize(nodeBufferSize);
    }
    MPI_Bcast(&(nodeBuffer[0]), nodeBufferSize, MPI_CHAR, mpi_root, MPI_COMM_WORLD);
    result = T::cast(deserializeDAALObject(&(nodeBuffer[0]), nodeBufferSize));
}

template <typename T>
void sendrecv(const ByteBuffer & nodeBuffer, services::SharedPtr<T> & result, int partnerRankID)
{
    int nodeBufferSize = nodeBuffer.size();
    int partnerNodeBufferSize = 0;
    MPI_Sendrecv(&nodeBufferSize, sizeof(nodeBufferSize), MPI_CHAR, partnerRankID, 0,
                 &partnerNodeBufferSize, sizeof(partnerNodeBufferSize), MPI_CHAR, partnerRankID, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    ByteBuffer recvBuffer(partnerNodeBufferSize);
    MPI_Sendrecv(&(nodeBuffer[0]), nodeBufferSize, MPI_CHAR, partnerRankID, 1,
                 &(recvBuffer[0]), partnerNodeBufferSize, MPI_CHAR, partnerRankID, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    result = T::cast(deserializeDAALObject(&(recvBuffer[0]), partnerNodeBufferSize));
}

void sendrecv(const data_management::NumericTableConstPtr & data, const data_management::NumericTableConstPtr & labels,
              const data_management::NumericTableConstPtr & keys, const data_management::NumericTablePtr & arrangedData,
              const data_management::NumericTablePtr & arrangedLabels)
{
    typedef int KeyType;
    typedef double DataType;
    typedef double LabelsType;
    data_management::BlockDescriptor<KeyType> keyBlockDescriptor;

    // Estimates capacity for each outer numeric table
    size_t * const sendCount = new size_t[comm_size];
    for (int i = 0; i < comm_size; ++i) { sendCount[i] = 0; }
    const size_t numberOfRows = keys->getNumberOfRows();
    const_cast<NumericTable &>(*keys).getBlockOfRows(0, numberOfRows, data_management::readOnly, keyBlockDescriptor);
    const KeyType * const k = keyBlockDescriptor.getBlockPtr();
    for (size_t j = 0; j < numberOfRows; ++j)
    {
        ++sendCount[k[j]];
    }

    // Sends and receives capacities to/from appropriate hosts
    size_t * const recvCount = new size_t[comm_size];
    MPI_Alltoall(sendCount, sizeof(sendCount[0]), MPI_CHAR, recvCount, sizeof(recvCount[0]), MPI_CHAR, MPI_COMM_WORLD);

    // Calculate total number of rows to be received.
    size_t * const recvStartPos = new size_t[comm_size + 1];
    size_t recvTotalCount = 0;
    for (int i = 0; i < comm_size; ++i)
    {
        recvStartPos[i] = recvTotalCount;
        recvTotalCount += recvCount[i];
    }
    recvStartPos[comm_size] = recvTotalCount;

    // Allocates enough space in arrangedData and arrangedLabels
    arrangedData->resize(recvTotalCount);
    arrangedLabels->resize(recvTotalCount);

    // Sends and receives data and labels

    size_t recvReqCapacity = 200 * comm_size, recvReqCount = 0;
    MPI_Request * recvReqs = new MPI_Request[recvReqCapacity];
    data_management::BlockDescriptor<DataType> * const recvDataBD = new data_management::BlockDescriptor<DataType>[comm_size];
    data_management::BlockDescriptor<LabelsType> * const recvLabelsBD = new data_management::BlockDescriptor<LabelsType>[comm_size];

    const int tag = 2;
    for (int i = 0; i < comm_size; ++i)
    {
        if (recvCount[i] > 0)
        {
            arrangedData->getBlockOfRows(recvStartPos[i], recvCount[i], data_management::writeOnly, recvDataBD[i]);
            recv(recvDataBD[i].getBlockPtr(), recvCount[i] * recvDataBD[i].getNumberOfColumns(), i, tag, recvReqs, recvReqCount, recvReqCapacity);
            arrangedLabels->getBlockOfRows(recvStartPos[i], recvCount[i], data_management::writeOnly, recvLabelsBD[i]);
            recv(recvLabelsBD[i].getBlockPtr(), recvCount[i] * recvLabelsBD[i].getNumberOfColumns(), i, tag + 10000, recvReqs, recvReqCount,
                 recvReqCapacity);
        }
    }

    size_t sendReqCapacity = 200 * comm_size, sendReqCount = 0;
    MPI_Request * sendReqs = new MPI_Request[sendReqCapacity];
    data_management::HomogenNumericTable<DataType> * const sendData = new data_management::HomogenNumericTable<DataType>[comm_size];
    data_management::HomogenNumericTable<LabelsType> * const sendLabels = new data_management::HomogenNumericTable<LabelsType>[comm_size];
    size_t * const positions = new size_t[comm_size];

    for (int i = 0; i < comm_size; ++i)
    {
        sendData[i].getDictionary()->setNumberOfFeatures(data->getNumberOfColumns());
        sendData[i].resize(sendCount[i]);
        sendLabels[i].getDictionary()->setNumberOfFeatures(labels->getNumberOfColumns());
        sendLabels[i].resize(sendCount[i]);
        positions[i] = 0;
    }

    data_management::BlockDescriptor<DataType> dataBD;
    data_management::BlockDescriptor<LabelsType> labelsBD;
    const_cast<data_management::NumericTable &>(*data).getBlockOfRows(0, numberOfRows, data_management::readOnly, dataBD);
    const DataType * const ddata = dataBD.getBlockPtr();
    const_cast<data_management::NumericTable &>(*labels).getBlockOfRows(0, numberOfRows, data_management::readOnly, labelsBD);
    const LabelsType * const dlabels = labelsBD.getBlockPtr();
    for (size_t j = 0; j < numberOfRows; ++j)
    {
        copy(ddata + j * dataBD.getNumberOfColumns(), ddata + j * dataBD.getNumberOfColumns() + dataBD.getNumberOfColumns(),
             sendData[k[j]][positions[k[j]]]);
        copy(dlabels + j * labelsBD.getNumberOfColumns(), dlabels + j * labelsBD.getNumberOfColumns() + labelsBD.getNumberOfColumns(),
             sendLabels[k[j]][positions[k[j]]]);
        ++positions[k[j]];
    }
    const_cast<data_management::NumericTable &>(*data).releaseBlockOfRows(dataBD);
    const_cast<data_management::NumericTable &>(*labels).releaseBlockOfRows(labelsBD);

    const_cast<NumericTable &>(*keys).releaseBlockOfRows(keyBlockDescriptor);

    for (int i = 0; i < comm_size; ++i)
    {
        if (sendCount[i] > 0)
        {
            send(sendData[i].getArray(), sendData[i].getNumberOfRows() * sendData[i].getNumberOfColumns(), i, tag, sendReqs, sendReqCount,
                 sendReqCapacity);
            send(sendLabels[i].getArray(), sendLabels[i].getNumberOfRows() * sendLabels[i].getNumberOfColumns(), i, tag + 10000, sendReqs,
                 sendReqCount, sendReqCapacity);
        }
    }

    if (sendReqCount > 0)
    {
        MPI_Waitall(sendReqCount, sendReqs, MPI_STATUSES_IGNORE);
    }

    if (recvReqCount > 0)
    {
        MPI_Waitall(recvReqCount, recvReqs, MPI_STATUSES_IGNORE);
    }

    for (int i = 0; i < comm_size; ++i)
    {
        if (recvCount[i] > 0)
        {
            arrangedData->releaseBlockOfRows(recvDataBD[i]);
            arrangedLabels->releaseBlockOfRows(recvLabelsBD[i]);
        }
    }

    delete[] positions;
    delete[] sendReqs;
    delete[] sendLabels;
    delete[] sendData;
    delete[] recvLabelsBD;
    delete[] recvDataBD;
    delete[] recvReqs;
    delete[] recvStartPos;
    delete[] recvCount;
    delete[] sendCount;

    MPI_Barrier(MPI_COMM_WORLD);
}

template <typename T>
inline void recv(T * buffer, size_t length, int source, int tag, MPI_Request * & req, size_t & reqCount, size_t & reqCapacity)
{
    const size_t size = length * sizeof(T);
    const size_t maxChunkSize = 1048576;
    const size_t chunkCount = (size + maxChunkSize - 1) / maxChunkSize;
    if (reqCount + chunkCount > reqCapacity)
    {
        reqCapacity = chunkCount + reqCount;
        MPI_Request * const newReq = new MPI_Request[reqCapacity];
        copy(req, &(req[reqCount]), newReq);
        MPI_Request * const oldReq = req;
        req = newReq;
        delete[] oldReq;
    }

    size_t * const pos = new size_t[chunkCount];
    size_t * const count = new size_t[chunkCount];
    size_t p = 0;
    for (size_t i = 0; i < chunkCount; ++i)
    {
        if (i == chunkCount - 1)
        {
            pos[i] = p;
            count[i] = size - p;
        }
        else
        {
            pos[i] = p;
            count[i] = maxChunkSize;
            p += maxChunkSize;
        }
    }

    int t = tag;
    for (size_t i = 0; i < chunkCount; ++i)
    {
        MPI_Irecv(&(((char *)buffer)[pos[i]]), count[i], MPI_CHAR, source, t, MPI_COMM_WORLD, &(req[reqCount]));
        ++reqCount;
        ++t;
    }

    delete[] count;
    delete[] pos;
}

template <typename T>
inline void send(const T * buffer, size_t length, int target, int tag, MPI_Request * & req, size_t & reqCount, size_t & reqCapacity)
{
    const size_t size = length * sizeof(T);
    const size_t maxChunkSize = 1048576;
    const size_t chunkCount = (size + maxChunkSize - 1) / maxChunkSize;
    if (reqCount + chunkCount > reqCapacity)
    {
        reqCapacity = chunkCount + reqCount;
        MPI_Request * const newReq = new MPI_Request[reqCapacity];
        copy(req, &(req[reqCount]), newReq);
        MPI_Request * const oldReq = req;
        req = newReq;
        delete[] oldReq;
    }

    size_t * const pos = new size_t[chunkCount];
    size_t * const count = new size_t[chunkCount];
    size_t p = 0;
    for (size_t i = 0; i < chunkCount; ++i)
    {
        if (i == chunkCount - 1)
        {
            pos[i] = p;
            count[i] = size - p;
        }
        else
        {
            pos[i] = p;
            count[i] = maxChunkSize;
            p += maxChunkSize;
        }
    }

    int t = tag;
    for (size_t i = 0; i < chunkCount; ++i)
    {
        MPI_Isend(&(((const char *)buffer)[pos[i]]), count[i], MPI_CHAR, target, t, MPI_COMM_WORLD, &(req[reqCount]));
        ++reqCount;
        ++t;
    }

    delete[] count;
    delete[] pos;
}
