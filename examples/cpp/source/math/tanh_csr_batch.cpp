/* file: tanh_csr_batch.cpp */
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
!    C++ example of tanh algorithm.
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-TANH_CSR_BATCH"></a>
 * \example tanh_csr_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::math;

/* Input data set parameters */
string datasetName = "../data/batch/covcormoments_csr.csv";

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetName);

    /* Read datasetFileName from a file and create a numeric table to store input data */
    CSRNumericTablePtr dataTable(createSparseTable<float>(datasetName));

    /* Create an algorithm */
    tanh::Batch<float, tanh::fastCSR> tanh;

    /* Set an input object for the algorithm */
    tanh.input.set(tanh::data, dataTable);

    /* Compute Abs function */
    tanh.compute();

    /* Print the results of the algorithm */
    tanh::ResultPtr res = tanh.getResult();
    printNumericTable(res->get(tanh::value), "Hyperbolic Tangent result (first 5 rows):", 5);

    return 0;
}
