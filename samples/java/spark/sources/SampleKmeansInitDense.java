/* file: SampleKmeansInitDense.java */
/*******************************************************************************
* Copyright 2017-2019 Intel Corporation.
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
*
* License:
* http://software.intel.com/en-us/articles/intel-sample-source-code-license-agr
* eement/
*******************************************************************************/

/*
 //  Content:
 //     Java sample of dense K-Means clustering
 ////////////////////////////////////////////////////////////////////////////////
 */

package DAAL;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.SparkConf;

import java.nio.DoubleBuffer;
import java.io.IOException;
import java.lang.ClassNotFoundException;

import com.intel.daal.data_management.data.*;
import com.intel.daal.data_management.data_source.*;
import com.intel.daal.services.*;

public class SampleKmeansInitDense {
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        DaalContext context = new DaalContext();

        /* Create JavaSparkContext that loads defaults from the system properties and the classpath and sets the name */
        JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("Spark Kmeans"));

        /* Read from the distributed HDFS data set at a specified path */
        StringDataSource templateDataSource = new StringDataSource( context, "" );
        DistributedHDFSDataSet dd = new DistributedHDFSDataSet( "/Spark/KmeansInitDense/data/", templateDataSource );
        JavaPairRDD<Integer, HomogenNumericTable> dataRDD = dd.getAsPairRDD(sc);

        /* Get initial centroids by plusPlusDense method */
        /* Compute k-means for dataRDD */
        SparkKmeansInitDense.KmeansInitResult initResult = SparkKmeansInitDense.initKmeansPlusPlus(sc, context, dataRDD);
        SparkKmeansInitDense.KmeansResult         result = SparkKmeansInitDense.runKmeans(context, dataRDD, initResult.centroids);

        /* Print the results */
        printNumericTable("Initial centroids (plusPlusDense method):", initResult.centroids, 20, 10);
        printNumericTable("Result centroids:", result.centroids, 20, 10);

        /* Get initial centroids by parallelPlusDense method */
        /* Compute k-means for dataRDD */
        initResult = SparkKmeansInitDense.initKmeansParallelPlus(sc, context, dataRDD);
        result     = SparkKmeansInitDense.runKmeans(context, dataRDD, initResult.centroids);

        /* Print the results */
        printNumericTable("Initial centroids (parallelPlusDense method):", initResult.centroids, 20, 10);
        printNumericTable("Result centroids:", result.centroids, 20, 10);

        context.dispose();
        sc.stop();
    }

    public static void printNumericTable(String header, NumericTable nt, long nPrintedRows, long nPrintedCols) {
        long nNtCols = nt.getNumberOfColumns();
        long nNtRows = nt.getNumberOfRows();
        long nRows = nNtRows;
        long nCols = nNtCols;

        if(nPrintedRows > 0) {
            nRows = Math.min(nNtRows, nPrintedRows);
        }

        DoubleBuffer result = DoubleBuffer.allocate((int)(nNtCols * nRows));
        result = nt.getBlockOfRows(0, nRows, result);

        if(nPrintedCols > 0) {
            nCols = Math.min(nNtCols, nPrintedCols);
        }

        StringBuilder builder = new StringBuilder();
        builder.append(header);
        builder.append("\n");
        for (long i = 0; i < nRows; i++) {
            for (long j = 0; j < nCols; j++) {
                String tmp = String.format("%-6.3f   ", result.get((int)(i * nNtCols + j)));
                builder.append(tmp);
            }
            builder.append("\n");
        }
        System.out.println(builder.toString());
    }
}
