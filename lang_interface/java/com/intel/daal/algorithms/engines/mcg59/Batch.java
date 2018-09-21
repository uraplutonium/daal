/* file: Batch.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
 * @defgroup engines_mcg59_batch Batch
 * @ingroup engines_mcg59
 * @{
 */
/**
 * @brief Contains classes for the mcg59 engine
 */
package com.intel.daal.algorithms.engines.mcg59;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.engines.Input;
import com.intel.daal.algorithms.engines.Result;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ENGINES__MCG59__BATCH"></a>
 * \brief Provides methods for mcg59 engine computations in the batch processing mode
 *
 * \par References
 *      - @ref com.intel.daal.algorithms.engines.Input class
 */
public class Batch extends com.intel.daal.algorithms.engines.BatchBase {
    public  Method       method;    /*!< Computation method for the engine */
    private Precision    prec;      /*!< Data type to use in intermediate computations for the engine */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs mcg59 engine by copying input objects and parameters of another mcg59 engine
     * @param context Context to manage the mcg59 engine
     * @param other   A engines to be used as the source to initialize the input objects
     *                and parameters of this engine
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
    }

    /**
     * Constructs the mcg59 engine
     * @param context    Context to manage the engine
     * @param cls        Data type to use in intermediate computations for the engine, Double.class or Float.class
     * @param method     The engine computation method, @ref Method
     * @param seed       Initial condition
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method, int seed) {
        super(context);
        constructBatch(context, cls, method, seed);
    }

    private void constructBatch(DaalContext context, Class<? extends Number> cls, Method method, int seed) {
        this.method = method;

        if (method != Method.defaultDense) {
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

        this.cObject = cInit(prec.getValue(), method.getValue(), seed);
    }

    /**
     * Computes the result of the mcg59 engine
     * @return  Mcg59 engine result
     */
    @Override
    public Result compute() {
        super.compute();
        return new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the newly allocated mcg59 engine
     * with a copy of input objects and parameters of this mcg59 engine
     * @param context    Context to manage the engine
     * @return The newly allocated mcg59 engine
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method, int seed);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */