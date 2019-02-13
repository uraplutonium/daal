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
 * @defgroup low_order_moments Moments of Low Order
 * @brief Contains classes for computing the results of the low order moments algorithm
 * @ingroup analysis
 * @{
 */
/**
 * @defgroup low_order_moments_batch Batch
 * @ingroup low_order_moments
 * @{
 */
/**
 * @brief Contains classes for computing %moments of low order
 */
package com.intel.daal.algorithms.low_order_moments;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__BATCH"></a>
 * @brief Computes %moments of low order in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-LOW_ORDER_MOMENTS-ALGORITHM">Low order %moments algorithm description and usage models</a> -->
 *
 * @par References
 *      - InputId class. Identifiers of the input objects for the low order moments algorithm
 *      - ResultId class. Identifiers of the results of the low order moments algorithm
 *      - Input class
 *      - Result class
 */
public class Batch extends BatchImpl {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs low order %moments algorithm by copying input objects
     * of another low order %moments algorithm
     * @param context   Context to manage the low order %moments algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());

        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()), cObject, prec, method, ComputeMode.batch);
    }

    /**
     * Constructs low order %moments algorithm
     * @param context   Context to manage the low order %moments algorithm
     * @param cls       Data type to use in intermediate computations of the low order %moments algorithm,
     *                  Double.class or Float.class
     * @param method    Computation method of the algorithm, @ref Method
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

        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()), cObject, prec, method, ComputeMode.batch);
    }

    /**
     * Returns the newly allocated low order %moments algorithm
     * with a copy of input objects of this algorithm
     * @param context   Context to manage the low order %moments algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
/** @} */
