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
 * @defgroup normalization Normalization
 * @brief Contains classes for computing normalization algorithms
 * @ingroup analysis
 * @{
 */
/**
 * @defgroup zscore Z-score
 * @brief Contains classes for computing Z-score normalization algorithms
 * @ingroup normalization
 * @{
 */
/**
 * @defgroup zscore_batch Batch
 * @ingroup zscore
 * @{
 */
/**
 * @brief Contains classes for computing Z-score normalization solvers
 */
package com.intel.daal.algorithms.normalization.zscore;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.ComputeMode;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ZSCORE__BATCH"></a>
 * \brief Computes Z-score normalization in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-ZSCORE-ALGORITHM">Z-score normalization algorithm description and usage models</a> -->
 *
 * \par References
 *      - @ref InputId class
 *      - @ref ResultId class
 *
 */
public class Batch extends AnalysisBatch {
    public Input          input;     /*!< %Input data */
    public Method     method; /*!< Computation method for the algorithm */
    private Precision prec;          /*!< Precision of computations */
    public Parameter  parameter;      /*!< Parameters of the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHM__ZSCORE__BATCH__BATCH"></a>
     * Constructs the Z-score normalization algorithm
     *
     * @param context    Context to manage the Z-score normalization algorithm
     * @param cls        Data type to use in intermediate computations for Z-score normalization, Double.class or Float.class
     * @param method     Z-score normalization computation method, @ref Method
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);

        this.method = method;

        if (method != Method.defaultDense && method != Method.sumDense) {
            throw new IllegalArgumentException("method unsupported");
        }
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), method.getValue());
        input = new Input(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()),
                                  cObject, prec, method, ComputeMode.batch);
    }

    /**
    * Constructs algorithm that computes normalization by copying input objects and parameters
    * of another algorithm
    * @param context      Context to manage the normalization algorithms
    * @param other        An algorithm to be used as the source to initialize the input objects
    *                     and parameters of the algorithm
    */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());

        input = new Input(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()),
                                  cObject, prec, method, ComputeMode.batch);
    }

    /**
     * Computes Z-score normalization
     * @return  Z-score normalization results
    */
    @Override
    public Result compute() {
        super.compute();
        Result result = new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }


    /**
     * Returns the newly allocated algorithm that computes normalization
     * with a copy of input objects and parameters of this algorithm
     * @param context      Context to manage the normalization algorithms
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
/** @} */
/** @} */
