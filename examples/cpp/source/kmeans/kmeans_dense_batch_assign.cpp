/* file: kmeans_dense_batch_assign.cpp */
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
!    C++ example of dense K-Means clustering in the batch processing mode
!    for calculation assignments without centroids update
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-KMEANS_DENSE_BATCH_ASSIGN"></a>
 * \example kmeans_dense_batch_assign.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
string datasetFileName     = "../data/batch/kmeans_dense.csv";

/* K-Means algorithm parameters */
const size_t nClusters   = 20;

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Get initial clusters for the K-Means algorithm */
    kmeans::init::Batch<float, kmeans::init::randomDense> init(nClusters);

    init.input.set(kmeans::init::data, dataSource.getNumericTable());
    init.compute();

    NumericTablePtr centroids = init.getResult()->get(kmeans::init::centroids);

    /* Create an algorithm object for the K-Means algorithm to calculate only assignments */
    kmeans::Batch<> algorithm(nClusters, 0);

    algorithm.input.set(kmeans::data,           dataSource.getNumericTable());
    algorithm.input.set(kmeans::inputCentroids, centroids);

    algorithm.compute();

    /* Print the clusterization results */
    printNumericTable(algorithm.getResult()->get(kmeans::assignments), "First 10 cluster assignments:", 10);

    return 0;
}
