/* file: DropoutBackwardBatch.java */
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
 * @defgroup dropout_backward_batch Batch
 * @ingroup dropout_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.dropout;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__DROPOUT__DROPOUTBACKWARDBATCH"></a>
 * \brief Class that computes the results of the backward dropout layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-DROPOUTBACKWARD">Backward dropout layer description and usage models</a> -->
 *
 * \par References
 *      - @ref DropoutLayerDataId class
 */
public class DropoutBackwardBatch extends com.intel.daal.algorithms.neural_networks.layers.BackwardLayer {
    public  DropoutBackwardInput input;     /*!< %Input data */
    public  DropoutMethod        method;    /*!< Computation method for the layer */
    public  DropoutParameter     parameter; /*!< DropoutParameters of the layer */
    private Precision     prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the backward dropout layer by copying input objects of backward dropout layer
     * @param context    Context to manage the backward dropout layer
     * @param other      A backward dropout layer to be used as the source
     *                   to initialize the input objects of the backward dropout layer
     */
    public DropoutBackwardBatch(DaalContext context, DropoutBackwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new DropoutBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new DropoutParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward dropout layer
     * @param context    Context to manage the backward dropout layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref DropoutMethod
     */
    public DropoutBackwardBatch(DaalContext context, Class<? extends Number> cls, DropoutMethod method) {
        super(context);

        this.method = method;

        if (method != DropoutMethod.defaultDense) {
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
        input = new DropoutBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new DropoutParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    DropoutBackwardBatch(DaalContext context, Class<? extends Number> cls, DropoutMethod method, long cObject) {
        super(context);

        this.method = method;

        if (method != DropoutMethod.defaultDense) {
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
        input = new DropoutBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new DropoutParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the backward dropout layer
     * @return  Backward dropout layer result
     */
    @Override
    public DropoutBackwardResult compute() {
        super.compute();
        DropoutBackwardResult result = new DropoutBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the backward dropout layer
     * @param result    Structure to store the result of the backward dropout layer
     */
    public void setResult(DropoutBackwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the backward layer
     * @return Structure that contains result of the backward layer
     */
    @Override
    public DropoutBackwardResult getLayerResult() {
        return new DropoutBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * @return Structure that contains input object of the backward layer
     */
    @Override
    public DropoutBackwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the backward layer
     * @return Structure that contains parameters of the backward layer
     */
    @Override
    public DropoutParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated backward dropout layer
     * with a copy of input objects of this backward dropout layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated backward dropout layer
     */
    @Override
    public DropoutBackwardBatch clone(DaalContext context) {
        return new DropoutBackwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
