/* file: CorDistDenseBatch.java */
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
 //  Content:
 //     Java example of computing a correlation distance matrix
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-CORRELATIONDISTANCEBATCH">
 * @example CorDistDenseBatch.java
 */

package com.intel.daal.examples.distance;

import com.intel.daal.algorithms.cordistance.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class CorDistDenseBatch {
    /* Input data set parameters */
    private static final String dataset       = "../data/batch/distance.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        /* Retrieve the input data */
        FileDataSource dataSource = new FileDataSource(context, dataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();

        /* Create an algorithm to compute a correlation distance matrix using the defaultDense method */
        Batch alg = new Batch(context, Float.class, Method.defaultDense);

        NumericTable input = dataSource.getNumericTable();
        alg.input.set(InputId.data, input);
        Result result = alg.compute();

        NumericTable res = result.get(ResultId.correlationDistance);

        Service.printNumericTable("Correlation distance", res, 15,15);

        context.dispose();
    }
}
