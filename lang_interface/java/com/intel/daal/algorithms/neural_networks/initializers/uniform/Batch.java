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
 * @brief Contains classes for the uniform initializer
 */
package com.intel.daal.algorithms.neural_networks.initializers.uniform;

import com.intel.daal.algorithms.neural_networks.initializers.Input;
import com.intel.daal.algorithms.neural_networks.initializers.Result;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__INITIALIZERS__UNIFORM__BATCH"></a>
 * \brief Provides methods for uniform initializer computations in the batch processing mode
 *
 * \par References
 *      - @ref Method class
 *      - @ref Parameter class
 *      - @ref com.intel.daal.algorithms.neural_networks.initializers.Input class
 *      - @ref com.intel.daal.algorithms.neural_networks.initializers.Result class
 */
public class Batch extends com.intel.daal.algorithms.neural_networks.initializers.InitializerIface {
    public  Parameter    parameter; /*!< Parameters of the uniform initializer */
    public  Method       method;    /*!< Computation method for the initializer */
    private Precision    prec;      /*!< Data type to use in intermediate computations for the initializer */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs uniform initializer by copying input objects and parameters of another uniform initializer
     * @param context Context to manage the uniform initializer
     * @param other   An initializer to be used as the source to initialize the input objects
     *                and parameters of this initializer
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(context, cGetInput(cObject));
        parameter = new Parameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the uniform initializer
     * @param context    Context to manage the initializer
     * @param cls        Data type to use in intermediate computations for the initializer, Double.class or Float.class
     * @param method     The initializer computation method, @ref Method
     * @param a          Left bound of the interval
     * @param b          Right bound of the interval
     * @param seed       The seed for generating random values
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method, double a, double b, long seed) {
        super(context);
        constructBatch(context, cls, method, a, b, seed);
    }

    /**
    * Constructs the uniform initializer
    * @param context    Context to manage the initializer
    * @param cls        Data type to use in intermediate computations for the initializer, Double.class or Float.class
    * @param method     The initializer computation method, @ref Method
    * @param a          Left bound of the interval
    * @param b          Right bound of the interval
    */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method, double a, double b) {
        super(context);

        long seed = 777;
        constructBatch(context, cls, method, a, b, seed);
    }

    private void constructBatch(DaalContext context, Class<? extends Number> cls, Method method, double a, double b, long seed) {
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

        this.cObject = cInit(prec.getValue(), method.getValue(), a, b, seed);
        input = new Input(context, cGetInput(cObject));
        parameter = new Parameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
        parameter.setA(a);
        parameter.setB(b);
        parameter.setSeed(seed);
    }

    /**
     * Computes the result of the uniform initializer
     * @return  Uniform initializer result
     */
    @Override
    public Result compute() {
        super.compute();
        return new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the newly allocated uniform initializer
     * with a copy of input objects and parameters of this uniform initializer
     * @param context    Context to manage the initializer
     * @return The newly allocated uniform initializer
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method, double a, double b, long seed);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native long cClone(long algAddr, int prec, int method);
}
