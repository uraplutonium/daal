/* file: Batch.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/**
 * @brief Contains classes for computing limited memory Broyden-Fletcher-Goldfarb-Shanno algorithm
 */
package com.intel.daal.algorithms.optimization_solver.lbfgs;

import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__LBFGS__BATCH"></a>
 * @brief Computes the results of LBFGS algorithm in the batch processing mode
 *
 * @par References
 *      - <a href="DAAL-REF-LBFGS-ALGORITHM">LBFGS algorithm description and usage models</a>
 *      - @ref com.intel.daal.algorithms.optimization_solver.lbfgs.Method class
 *      - @ref com.intel.daal.algorithms.optimization_solver.lbfgs.Parameter class
 *      - @ref com.intel.daal.algorithms.optimization_solver.lbfgs.InputId class
 *      - @ref com.intel.daal.algorithms.optimization_solver.lbfgs.Input class
 *      - @ref com.intel.daal.algorithms.optimization_solver.lbfgs.ResultId class
 *      - @ref com.intel.daal.algorithms.optimization_solver.lbfgs.Result class
 *
 */
public class Batch extends com.intel.daal.algorithms.optimization_solver.Batch {

    public Input          input;     /*!< %Input data */
    public Method   method;  /*!< Computation method for the algorithm */
    public Parameter  parameter;     /*!< Parameters of the algorithm */
    private Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the LBFGS algorithm by copying input objects and parameters of another LBFGS algorithm
     * @param context    Context to manage the LBFGS algorithm
     * @param other      An algorithm to be used as the source to initialize the input objects
     *                   and parameters of this algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(context, cGetParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the LBFGS algorithm
     *
     * @param context      Context to manage the LBFGS algorithm
     * @param cls          Data type to use in intermediate computations for the LBFGS algorithm, Double.class or Float.class
     * @param method       LBFGS computation method, @ref Method
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);

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

        this.cObject = cInit(prec.getValue(), method.getValue());
        input = new Input(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(context, cGetParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the results of LBFGS algorithm in the batch processing mode
     * @return  Results of the computation
     */
    @Override
    public Result compute() {
        super.compute();
        return new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Registers user-allocated memory to store the results of computing the LBFGS
     * in the batch processing mode
     * @param result    Structure to store results of computing the LBFGS
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Return the input of the algorithm
     * @return Input of the algorithm
     */
    public Input getInput() {
        return (Input)input;
    }

    /**
     * Returns the newly allocated LBFGS algorithm
     * with a copy of input objects and parameters of this LBFGS algorithm
     * @param context    Context to manage the LBFGS algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cClone(long algAddr, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetParameter(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cGetResult(long cAlgorithm, int prec, int method);
}
