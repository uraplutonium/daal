/* file: TanhBackwardBatch.java */
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
 * @defgroup tanh_layers_backward_batch Batch
 * @ingroup tanh_layers_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.tanh;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.Parameter;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__TANH__TANHBACKWARDBATCH"></a>
 * \brief Class that computes the results of the backward hyperbolic tangent (tanh) layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-TANHBACKWARD">Backward tanh layer description and usage models</a> -->
 *
 * \par References
 *      - @ref TanhLayerDataId class
 */
public class TanhBackwardBatch extends com.intel.daal.algorithms.neural_networks.layers.BackwardLayer {
    public  TanhBackwardInput input;    /*!< %Input data */
    public  TanhMethod        method;   /*!< Computation method for the layer */
    private Precision     prec;     /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the backward tanh layer by copying input objects from another backward tanh layer
     * @param context    Context to manage the backward tanh layer
     * @param other      An algorithm to be used as the source to initialize the input objects
     *                   and parameters of the backward tanh layer
     */
    public TanhBackwardBatch(DaalContext context, TanhBackwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new TanhBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward tanh layer
     * @param context    Context to manage the backward tanh layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref TanhMethod
     */
    public TanhBackwardBatch(DaalContext context, Class<? extends Number> cls, TanhMethod method) {
        super(context);

        this.method = method;

        if (method != TanhMethod.defaultDense) {
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
        input = new TanhBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward tanh layer
     * @param context    Context to manage the backward tanh layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref TanhMethod
     */
    TanhBackwardBatch(DaalContext context, Class<? extends Number> cls, TanhMethod method, long cObject) {
        super(context);

        this.method = method;

        if (method != TanhMethod.defaultDense) {
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
        input = new TanhBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the backward tanh layer
     * @return  Backward tanh layer result
     */
    @Override
    public TanhBackwardResult compute() {
        super.compute();
        TanhBackwardResult result = new TanhBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the backward tanh layer
     * @param result    Structure to store the result of the backward tanh layer
     */
    public void setResult(TanhBackwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the backward layer
     * @return Structure that contains result of the backward layer
     */
    @Override
    public TanhBackwardResult getLayerResult() {
        return new TanhBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * @return Structure that contains input object of the backward layer
     */
    @Override
    public TanhBackwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the backward layer
     * @return Structure that contains parameters of the backward layer
     */
    @Override
    public Parameter getLayerParameter() {
        return null;
    }

    /**
     * Returns the newly allocated backward tanh layer
     * with a copy of input objects of this backward tanh layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated backward tanh layer
     */
    @Override
    public TanhBackwardBatch clone(DaalContext context) {
        return new TanhBackwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
