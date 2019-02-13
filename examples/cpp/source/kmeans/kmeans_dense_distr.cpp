/* file: kmeans_dense_distr.cpp */
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
!    C++ example of dense K-Means clustering in the distributed processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-KMEANS_DENSE_DISTRIBUTED"></a>
 * \example kmeans_dense_distr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

typedef float algorithmFPType; /* Algorithm floating-point type */

/* K-Means algorithm parameters */
const size_t nClusters   = 20;
const size_t nIterations = 5;
const size_t nBlocks     = 4;
const size_t nVectorsInBlock = 2500;

const string dataFileNames[] =
{
    "../data/distributed/kmeans_dense_1.csv", "../data/distributed/kmeans_dense_2.csv",
    "../data/distributed/kmeans_dense_3.csv", "../data/distributed/kmeans_dense_4.csv"
};

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 4, &dataFileNames[0], &dataFileNames[1], &dataFileNames[2], &dataFileNames[3]);

    kmeans::Distributed<step2Master> masterAlgorithm(nClusters);

    NumericTablePtr data[nBlocks];

    NumericTablePtr centroids;
    NumericTablePtr assignments[nBlocks];
    NumericTablePtr objectiveFunction;

    kmeans::init::Distributed<step2Master, algorithmFPType, kmeans::init::randomDense> masterInit(nClusters);
    for (size_t i = 0; i < nBlocks; i++)
    {
        /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
        FileDataSource<CSVFeatureManager> dataSource(dataFileNames[i], DataSource::doAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

        /* Retrieve the data from the input file */
        dataSource.loadDataBlock();
        data[i] = dataSource.getNumericTable();

        /* Create an algorithm object for the K-Means algorithm */
        kmeans::init::Distributed<step1Local, algorithmFPType, kmeans::init::randomDense> localInit(nClusters, nBlocks*nVectorsInBlock, i*nVectorsInBlock);

        localInit.input.set(kmeans::init::data, data[i]);
        localInit.compute();

        masterInit.input.add(kmeans::init::partialResults, localInit.getPartialResult());
    }
    masterInit.compute();
    masterInit.finalizeCompute();
    centroids = masterInit.getResult()->get(kmeans::init::centroids);

    /* Calculate centroids */
    for(size_t it = 0; it < nIterations; it++)
    {
        for (size_t i = 0; i < nBlocks; i++)
        {
            /* Create an algorithm object for the K-Means algorithm */
            kmeans::Distributed<step1Local> localAlgorithm(nClusters, false);

            /* Set the input data to the algorithm */
            localAlgorithm.input.set(kmeans::data,           data[i]);
            localAlgorithm.input.set(kmeans::inputCentroids, centroids);

            localAlgorithm.compute();

            masterAlgorithm.input.add(kmeans::partialResults, localAlgorithm.getPartialResult());
        }

        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        centroids = masterAlgorithm.getResult()->get(kmeans::centroids);
        objectiveFunction = masterAlgorithm.getResult()->get(kmeans::objectiveFunction);
    }

    /* Calculate assignments */
    for (size_t i = 0; i < nBlocks; i++)
    {
        /* Create an algorithm object for the K-Means algorithm */
        kmeans::Batch<> localAlgorithm(nClusters, 0);

        /* Set the input data to the algorithm */
        localAlgorithm.input.set(kmeans::data,           data[i]);
        localAlgorithm.input.set(kmeans::inputCentroids, centroids);

        localAlgorithm.compute();

        assignments[i] = localAlgorithm.getResult()->get(kmeans::assignments);
    }

    /* Print the clusterization results */
    printNumericTable(assignments[0], "First 10 cluster assignments from 1st node:", 10);
    printNumericTable(centroids, "First 10 dimensions of centroids:", 20, 10);
    printNumericTable(objectiveFunction,   "Objective function value:");

    return 0;
}
