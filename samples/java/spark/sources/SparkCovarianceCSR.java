/* file: SparkCovarianceCSR.java */
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
//      Java sample of sparse variance-covariance matrix computation in the
//      distributed processing mode
////////////////////////////////////////////////////////////////////////////////
*/

package DAAL;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Map;
import java.util.HashMap;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.SparkConf;

import com.intel.daal.algorithms.covariance.*;
import com.intel.daal.data_management.data.*;
import com.intel.daal.services.*;

public class SparkCovarianceCSR {
    /* Class containing the algorithm results */
    static class CovarianceResult {
        public HomogenNumericTable covariance;
        public HomogenNumericTable mean;
    }

    public static CovarianceResult runCovariance(DaalContext context, JavaRDD<CSRNumericTable> dataRDD) {
        JavaRDD<PartialResult> partsRDD = computeStep1Local(context, dataRDD);

        PartialResult reducedPres = reducePartialResults(context, partsRDD);

        CovarianceResult result = finalizeMergeOnMasterNode(context, reducedPres);

        return result;
    }

    private static JavaRDD<PartialResult> computeStep1Local(DaalContext context, JavaRDD<CSRNumericTable> dataRDD) {
        return dataRDD.map(new Function<CSRNumericTable, PartialResult>() {
            public PartialResult call(CSRNumericTable table) {
                DaalContext context = new DaalContext();

                /* Create an algorithm to compute a sparse variance-covariance matrix on local nodes*/
                DistributedStep1Local covarianceLocal = new DistributedStep1Local(context, Double.class, Method.fastCSR);

                /* Set the input data on local nodes */
                table.unpack(context);
                covarianceLocal.input.set(InputId.data, table);

                /* Compute a sparse variance-covariance matrix on local nodes */
                PartialResult pres = covarianceLocal.compute();
                pres.pack();

                context.dispose();
                return pres;
            }
        });
    }

    private static PartialResult reducePartialResults(DaalContext context, JavaRDD<PartialResult> partsRDD) {
        return partsRDD.reduce(new Function2<PartialResult, PartialResult, PartialResult>() {
            public PartialResult call(PartialResult pr1, PartialResult pr2) {
                DaalContext localContext = new DaalContext();

                /* Create an algorithm to compute new partial result from two partial results */
                DistributedStep2Master covarianceMaster = new DistributedStep2Master(localContext, Double.class, Method.fastCSR);

                /* Set the input data recieved from the local nodes */
                pr1.unpack(localContext);
                pr2.unpack(localContext);
                covarianceMaster.input.add(DistributedStep2MasterInputId.partialResults, pr1);
                covarianceMaster.input.add(DistributedStep2MasterInputId.partialResults, pr2);

                /* Compute a new partial result from two partial results */
                PartialResult reducedPresLocal = (PartialResult)covarianceMaster.compute();
                reducedPresLocal.pack();

                localContext.dispose();
                return reducedPresLocal;
            }
        });
    }

    private static CovarianceResult finalizeMergeOnMasterNode(DaalContext context, PartialResult reducedPres) {

        /* Create an algorithm to compute a dense variance-covariance matrix on the master node */
        DistributedStep2Master covarianceMaster = new DistributedStep2Master(context, Double.class, Method.fastCSR);

        /* Set the reduced partial result to the master algorithm to compute the final result */
        reducedPres.unpack(context);
        covarianceMaster.input.add(DistributedStep2MasterInputId.partialResults, reducedPres);

        /* Compute a dense variance-covariance matrix on the master node */
        covarianceMaster.compute();

        /* Finalize computations and retrieve the results */
        Result res = covarianceMaster.finalizeCompute();

        CovarianceResult covResult = new CovarianceResult();
        covResult.covariance = (HomogenNumericTable)res.get(ResultId.covariance);
        covResult.mean = (HomogenNumericTable)res.get(ResultId.mean);
        return covResult;
    }
}
