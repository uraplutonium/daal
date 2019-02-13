/* file: LcnForwardBatch.java */
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
 * @defgroup lcn_layers_forward_batch Batch
 * @ingroup lcn_layers_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.lcn;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LCN__LCNFORWARDBATCH"></a>
 * \brief Class that computes the results of the forward local contrast normalization layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-LCNFORWARD">Forward local contrast normalization layer description and usage models</a> -->
 *
 * \par References
 *      - @ref LcnMethod class
 *      - @ref LcnLayerDataId class
 *      - @ref LcnForwardInput class
 *      - @ref LcnForwardResult class
 */
public class LcnForwardBatch extends com.intel.daal.algorithms.neural_networks.layers.ForwardLayer {
    public  LcnForwardInput input;     /*!< %Input data */
    public  LcnMethod       method;    /*!< Computation method for the layer */
    public  LcnParameter    parameter; /*!< LcnParameters of the layer */
    private Precision    prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the forward local contrast normalization layer by copying input objects of another forward local contrast normalization layer
     * @param context    Context to manage the forward local contrast normalization layer
     * @param other      A forward local contrast normalization layer to be used as the source to initialize the input objects of the forward local contrast normalization layer
     */
    public LcnForwardBatch(DaalContext context, LcnForwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new LcnForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new LcnParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the forward local contrast normalization layer
     * @param context    Context to manage the layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref LcnMethod
     */
    public LcnForwardBatch(DaalContext context, Class<? extends Number> cls, LcnMethod method) {
        super(context);

        this.method = method;

        if (method != LcnMethod.defaultDense) {
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
        input = new LcnForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new LcnParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the forward local contrast normalization layer
     * @param context    Context to manage the layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref LcnMethod
     * @param cObject    Address of C++ forward batch
     */
    LcnForwardBatch(DaalContext context, Class<? extends Number> cls, LcnMethod method, long cObject) {
        super(context);

        this.method = method;

        if (method != LcnMethod.defaultDense) {
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
        input = new LcnForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new LcnParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the forward local contrast normalization layer
     * @return  Forward local contrast normalization layer result
     */
    @Override
    public LcnForwardResult compute() {
        super.compute();
        LcnForwardResult result = new LcnForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the forward local contrast normalization layer
     * @param result    Structure to store the result of the forward local contrast normalization layer
     */
    public void setResult(LcnForwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the forward layer
     * @return Structure that contains result of the forward layer
     */
    @Override
    public LcnForwardResult getLayerResult() {
        return new LcnForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the forward layer
     * @return Structure that contains input object of the forward layer
     */
    @Override
    public LcnForwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the forward layer
     * @return Structure that contains parameters of the forward layer
     */
    @Override
    public LcnParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated forward local contrast normalization layer
     * with a copy of input objects of this forward local contrast normalization layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated forward local contrast normalization layer
     */
    @Override
    public LcnForwardBatch clone(DaalContext context) {
        return new LcnForwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
