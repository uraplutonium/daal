/* file: Batch.java */
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

/**
 * @defgroup covariance Correlation and Variance-Covariance Matrices
 * @brief Contains classes for computing the correlation or variance-covariance matrix
 * @ingroup analysis
 * @{
 */
/**
 * @defgroup covariance_batch Batch
 * @ingroup covariance
 * @{
 */
/**
 * @brief Contains classes for computing the correlation or variance-covariance matrix
 * in the batch processing mode
 */
package com.intel.daal.algorithms.covariance;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;
import java.nio.DoubleBuffer;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__BATCH"></a>
 * @brief Computes the correlation or variance-covariance matrix in the batch processing mode
 * <!-- \n<a href="DAAL-REF-COVARIANCE-ALGORITHM">Correlation or variance-covariance matrix algorithm description and usage models</a> -->
 *
 * @tparam algorithmFPType  Data type to use in intermediate computations of the correlation or variance-covariance matrix, double or float
 * @tparam method           Computation method, @ref daal::algorithms::covariance::Method
 *
 * @par Enumerations
 *      - @ref Method   Computation methods of the correlation or variance-covariance matrix algorithm
 *      - @ref InputId  Identifiers of input objects
 *      - @ref ResultId Identifiers of the results
 *
 * @par References
 *      - Input class
 *      - Parameter class
 *      - Result class
 */
public class Batch extends BatchImpl {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the correlation or variance-covariance matrix algorithm
     * by copying input objects and parameters of another algorithm for correlation or variance-covariance
     * matrix computation
     *
     * @param context    Context to manage the correlation or variance-covariance matrix algorithm
     * @param other      An algorithm to be used as the source to initialize the input objects
     *                   and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;
        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());

        input = new Input(getContext(), cObject, prec, method, ComputeMode.batch);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue(), ComputeMode.batch.getValue()), cObject);
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHM__COVARIANCE__BATCH__BATCH"></a>
     * Constructs the correlation or variance-covariance matrix algorithm
     *
     * @param context    Context to manage the correlation or variance-covariance matrix algorithm
     * @param cls        Data type to use in intermediate computations for correlation or variance-covariance matrix, Double.class or Float.class
     * @param method     Correlation or variance-covariance matrix computation method, @ref Method
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);
        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != Method.defaultDense && this.method != Method.singlePassDense
            && this.method != Method.sumDense && this.method != Method.fastCSR
            && this.method != Method.singlePassCSR && this.method != Method.sumCSR) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue());

        input = new Input(getContext(), cObject, prec, method, ComputeMode.batch);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue(), ComputeMode.batch.getValue()), cObject);
    }

    /**
     * Returns the newly allocated correlation or variance-covariance matrix algorithm
     * with a copy of input objects and parameters of this correlation or variance-covariance matrix algorithm
     * @param context    Context to manage the correlation or variance-covariance matrix algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long algAddr, int prec, int method, int cmode);
    private native long cClone(long cAlgorithm, int prec, int method);
}
/** @} */
/** @} */
