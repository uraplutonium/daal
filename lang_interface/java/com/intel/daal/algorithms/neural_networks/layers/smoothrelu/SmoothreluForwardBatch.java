/* file: SmoothreluForwardBatch.java */
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
 * @defgroup smoothrelu_layers_forward_batch Batch
 * @ingroup smoothrelu_layers_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.smoothrelu;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.Parameter;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SMOOTHRELU__SMOOTHRELUFORWARDBATCH"></a>
 * \brief Class that computes the results of the forward smooth rectified linear unit (smoothrelu) layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-SMOOTHRELUFORWARD">Forward smooth relu layer description and usage models</a> -->
 *
 * \par References
 *      - @ref SmoothreluLayerDataId class
 */
public class SmoothreluForwardBatch extends com.intel.daal.algorithms.neural_networks.layers.ForwardLayer {
    public  SmoothreluForwardInput input;    /*!< %Input data */
    public  SmoothreluMethod       method;   /*!< Computation method for the layer */
    private Precision    prec;     /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the forward smoothrelu layer by copying input objects of another forward smoothrelu layer
     * @param context    Context to manage the forward smoothrelu layer
     * @param other      A forward smoothrelu layer to be used as the source to initialize the input objects of the forward smoothrelu layer
     */
    public SmoothreluForwardBatch(DaalContext context, SmoothreluForwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new SmoothreluForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the forward smoothrelu layer
     * @param context    Context to manage the layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref SmoothreluMethod
     */
    public SmoothreluForwardBatch(DaalContext context, Class<? extends Number> cls, SmoothreluMethod method) {
        super(context);

        this.method = method;

        if (method != SmoothreluMethod.defaultDense) {
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
        input = new SmoothreluForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the forward smoothrelu layer
     * @param context    Context to manage the layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref SmoothreluMethod
     */
    SmoothreluForwardBatch(DaalContext context, Class<? extends Number> cls, SmoothreluMethod method, long cObject) {
        super(context);

        this.method = method;

        if (method != SmoothreluMethod.defaultDense) {
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
        input = new SmoothreluForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the forward smoothrelu layer
     * @return  Forward smoothrelu layer result
     */
    @Override
    public SmoothreluForwardResult compute() {
        super.compute();
        SmoothreluForwardResult result = new SmoothreluForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the forward smoothrelu layer
     * @param result    Structure to store the result of the forward smoothrelu layer
     */
    public void setResult(SmoothreluForwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the forward layer
     * @return Structure that contains result of the forward layer
     */
    @Override
    public SmoothreluForwardResult getLayerResult() {
        return new SmoothreluForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the forward layer
     * @return Structure that contains input object of the forward layer
     */
    @Override
    public SmoothreluForwardInput getLayerInput() {
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
     * Returns the newly allocated forward smoothrelu layer
     * with a copy of input objects of this forward smoothrelu layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated forward smoothrelu layer
     */
    @Override
    public SmoothreluForwardBatch clone(DaalContext context) {
        return new SmoothreluForwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
