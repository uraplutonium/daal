/* file: NormalDenseBatch.java */
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
 //     Java example of normal distribution
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.distributions;

import com.intel.daal.algorithms.distributions.*;
import com.intel.daal.algorithms.distributions.normal.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-EXAMPLE-JAVA-NORMALDENSEBATCH">
 * @example NormalDenseBatch.java
 */
class NormalDenseBatch {
    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Create input table to fill with random numbers */
        HomogenNumericTable dataTable = new HomogenNumericTable(context, Float.class, 1, 10, NumericTable.AllocationFlag.DoAllocate);

        /* Create the algorithm */
        Batch normal = new Batch(context, Float.class, Method.defaultDense, 0.0, 1.0);

        /* Set the algorithm input */
        normal.input.set(InputId.tableToFill, dataTable);

        /* Set the Mersenne Twister engine to the distribution */
        com.intel.daal.algorithms.engines.mt19937.Batch eng = new com.intel.daal.algorithms.engines.mt19937.Batch(context, Float.class, com.intel.daal.algorithms.engines.mt19937.Method.defaultDense, 777);
        normal.parameter.setEngine(eng);

        /* Perform computations */
        normal.compute();

        /* Print the results */
        Service.printNumericTable("Normal distribution output:", dataTable);

        context.dispose();
    }
}
