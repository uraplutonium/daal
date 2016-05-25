/* file: BackwardBatch.java */
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

package com.intel.daal.algorithms.neural_networks.layers.maximum_pooling2d;

import com.intel.daal.algorithms.neural_networks.layers.pooling2d.SpatialDimensions;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__MAXIMUM_POOLING2D__BACKWARDBATCH"></a>
 * @brief Class that computes the results of the two-dimensional maximum pooling layer in the batch processing mode
 *
 * @par References
 *      - <a href="DAAL-REF-MAXIMUMPOOLING2DBACKWARD-ALGORITHM">Backward two-dimensional maximum pooling layer description and usage models</a>
 *      - @ref Method class
 *      - @ref LayerDataId class
 *      - @ref BackwardInput class
 *      - @ref BackwardResult class
 */
public class BackwardBatch extends com.intel.daal.algorithms.neural_networks.layers.BackwardLayer {
    public  BackwardInput input;     /*!< %Input data */
    public  Method        method;    /*!< Computation method for the layer */
    public  Parameter     parameter; /*!< Parameters of the layer */
    private Precision     prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the backward two-dimensional maximum pooling layer by copying input objects of backward two-dimensional maximum pooling layer
     * @param context    Context to manage the backward two-dimensional maximum pooling layer
     * @param other      A backward two-dimensional maximum pooling layer to be used as the source to initialize the input objects of
     *                   the backward two-dimensional maximum pooling layer
     */
    public BackwardBatch(DaalContext context, BackwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new BackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward two-dimensional maximum pooling layer
     * @param context    Context to manage the backward two-dimensional maximum pooling layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref Method
     * @param nDim       Number of dimensions in input data
     */
    public BackwardBatch(DaalContext context, Class<? extends Number> cls, Method method, long nDim) {
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

        this.cObject = cInit(prec.getValue(), method.getValue(), nDim);
        input = new BackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    BackwardBatch(DaalContext context, Class<? extends Number> cls, Method method, long cObject, long nDim) {
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

        this.cObject = cObject;
        input = new BackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
        SpatialDimensions sd = new SpatialDimensions(nDim - 2, nDim - 1);
        parameter.setSpatialDimensions(sd);
    }

    /**
     * Computes the result of the backward two-dimensional maximum pooling layer
     * @return  Backward two-dimensional maximum pooling layer result
     */
    @Override
    public BackwardResult compute() {
        super.compute();
        BackwardResult result = new BackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the backward two-dimensional maximum pooling layer
     * @param result    Structure to store the result of the backward two-dimensional maximum pooling layer
     */
    public void setResult(BackwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the backward layer
     * @return Structure that contains result of the backward layer
     */
    @Override
    public BackwardResult getLayerResult() {
        return new BackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * @return Structure that contains input object of the backward layer
     */
    @Override
    public BackwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the backward layer
     * @return Structure that contains parameters of the backward layer
     */
    @Override
    public Parameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated backward two-dimensional maximum pooling layer
     * with a copy of input objects of this backward two-dimensional maximum pooling layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated backward two-dimensional maximum pooling layer
     */
    @Override
    public BackwardBatch clone(DaalContext context) {
        return new BackwardBatch(context, this);
    }

    private native long cInit(int prec, int method, long nDim);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
