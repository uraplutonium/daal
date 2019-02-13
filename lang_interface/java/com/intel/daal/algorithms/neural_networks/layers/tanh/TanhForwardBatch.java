/* file: TanhForwardBatch.java */
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
 * @defgroup tanh_layers_forward_batch Batch
 * @ingroup tanh_layers_forward
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
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__TANH__TANHFORWARDBATCH"></a>
 * \brief Class that computes the results of the forward hyperbolic tangent (tanh) layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-TANHFORWARD">Forward tanh layer description and usage models</a> -->
 *
 * \par References
 *      - @ref TanhLayerDataId class
 */
public class TanhForwardBatch extends com.intel.daal.algorithms.neural_networks.layers.ForwardLayer {
    public  TanhForwardInput input;    /*!< %Input data */
    public  TanhMethod       method;   /*!< Computation method for the layer */
    private Precision    prec;     /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the forward tanh layer by copying input objects of another forward tanh layer
     * @param context    Context to manage the forward tanh layer
     * @param other      A forward tanh layer to be used as the source to initialize the input objects of the forward tanh layer
     */
    public TanhForwardBatch(DaalContext context, TanhForwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new TanhForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the forward tanh layer
     * @param context    Context to manage the layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref TanhMethod
     */
    public TanhForwardBatch(DaalContext context, Class<? extends Number> cls, TanhMethod method) {
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
        input = new TanhForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the forward tanh layer
     * @param context    Context to manage the layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref TanhMethod
     */
    TanhForwardBatch(DaalContext context, Class<? extends Number> cls, TanhMethod method, long cObject) {
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
        input = new TanhForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the forward tanh layer
     * @return  Forward tanh layer result
     */
    @Override
    public TanhForwardResult compute() {
        super.compute();
        TanhForwardResult result = new TanhForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the forward tanh layer
     * @param result    Structure to store the result of the forward tanh layer
     */
    public void setResult(TanhForwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the forward layer
     * @return Structure that contains result of the forward layer
     */
    @Override
    public TanhForwardResult getLayerResult() {
        return new TanhForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the forward layer
     * @return Structure that contains input object of the forward layer
     */
    @Override
    public TanhForwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the forward layer
     * @return Structure that contains parameters of the forward layer
     */
    @Override
    public Parameter getLayerParameter() {
        return null;
    }

    /**
     * Returns the newly allocated forward tanh layer
     * with a copy of input objects of this forward tanh layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated forward tanh layer
     */
    @Override
    public TanhForwardBatch clone(DaalContext context) {
        return new TanhForwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
