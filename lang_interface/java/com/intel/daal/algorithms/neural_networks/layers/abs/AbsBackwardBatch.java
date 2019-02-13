/* file: AbsBackwardBatch.java */
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
 * @defgroup abs_layers_backward_batch Batch
 * @ingroup abs_layers_backward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.abs;

import com.intel.daal.utils.*;
import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.Parameter;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ABS__ABSBACKWARDBATCH"></a>
 * \brief Class that computes the results of the backward abs layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-ABSBACKWARD">Backward abs layer description and usage models</a> -->
 *
 * \par References
 *      - @ref AbsLayerDataId class
 */
public class AbsBackwardBatch extends com.intel.daal.algorithms.neural_networks.layers.BackwardLayer {
    public  AbsBackwardInput input;    /*!< %Input data */
    public  AbsMethod        method;   /*!< Computation method for the layer */
    private Precision     prec;     /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the backward abs layer by copying input objects of backward abs layer
     * @param context    Context to manage the backward abs layer
     * @param other      A backward abs layer to be used as the source to initialize the input objects of the backward abs layer
     */
    public AbsBackwardBatch(DaalContext context, AbsBackwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new AbsBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the backward abs layer
     * @param context    Context to manage the backward abs layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref AbsMethod
     */
    public AbsBackwardBatch(DaalContext context, Class<? extends Number> cls, AbsMethod method) {
        super(context);

        this.method = method;

        if (method != AbsMethod.defaultDense) {
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
        input = new AbsBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    AbsBackwardBatch(DaalContext context, Class<? extends Number> cls, AbsMethod method, long cObject) {
        super(context);

        this.method = method;

        if (method != AbsMethod.defaultDense) {
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
        input = new AbsBackwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the backward abs layer
     * @return  Backward abs layer result
     */
    @Override
    public AbsBackwardResult compute() {
        super.compute();
        AbsBackwardResult result = new AbsBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the backward abs layer
     * @param result    Structure to store the result of the backward abs layer
     */
    public void setResult(AbsBackwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the backward layer
     * @return Structure that contains result of the backward layer
     */
    @Override
    public AbsBackwardResult getLayerResult() {
        return new AbsBackwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * @return Structure that contains input object of the backward layer
     */
    @Override
    public AbsBackwardInput getLayerInput() {
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
     * Returns the newly allocated backward abs layer
     * with a copy of input objects of this backward abs layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated backward abs layer
     */
    @Override
    public AbsBackwardBatch clone(DaalContext context) {
        return new AbsBackwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
