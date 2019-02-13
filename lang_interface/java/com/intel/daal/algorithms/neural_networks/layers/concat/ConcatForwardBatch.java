/* file: ConcatForwardBatch.java */
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
 * @defgroup concat_forward_batch Batch
 * @ingroup concat_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.concat;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONCAT__CONCATFORWARDBATCH"></a>
 * \brief Class that computes the results of the forward concat layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-CONCATFORWARD">Forward concat layer description and usage models</a> -->
 *
 * \par References
 *      - @ref ConcatLayerDataId class
 */
public class ConcatForwardBatch extends com.intel.daal.algorithms.neural_networks.layers.ForwardLayer {
    public  ConcatForwardInput input;     /*!< %Input data */
    public  ConcatMethod       method;    /*!< Computation method for the layer */
    public  ConcatParameter    parameter; /*!< ConcatParameters of the layer */
    private Precision    prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the forward concat layer by copying input objects of another forward concat layer
     * @param context    Context to manage the forward concat layer
     * @param other      A forward concat layer to be used as the source to
     *                   initialize the input objects of the forward concat layer
     */
    public ConcatForwardBatch(DaalContext context, ConcatForwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new ConcatForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new ConcatParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the forward concat layer
     * @param context    Context to manage the layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref ConcatMethod
     */
    public ConcatForwardBatch(DaalContext context, Class<? extends Number> cls, ConcatMethod method) {
        super(context);

        this.method = method;

        if (method != ConcatMethod.defaultDense) {
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
        input = new ConcatForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new ConcatParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    ConcatForwardBatch(DaalContext context, Class<? extends Number> cls, ConcatMethod method, long cObject) {
        super(context);

        this.method = method;

        if (method != ConcatMethod.defaultDense) {
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
        input = new ConcatForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new ConcatParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the forward concat layer
     * @return  Forward concat layer result
     */
    @Override
    public ConcatForwardResult compute() {
        super.compute();
        ConcatForwardResult result = new ConcatForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the forward concat layer
     * @param result    Structure to store the result of the forward concat layer
     */
    public void setResult(ConcatForwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the forward layer
     * @return Structure that contains result of the forward layer
     */
    @Override
    public ConcatForwardResult getLayerResult() {
        return new ConcatForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the forward layer
     * @return Structure that contains input object of the forward layer
     */
    @Override
    public ConcatForwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the forward layer
     * @return Structure that contains parameters of the forward layer
     */
    @Override
    public ConcatParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated forward concat layer
     * with a copy of input objects of this forward concat layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated forward concat layer
     */
    @Override
    public ConcatForwardBatch clone(DaalContext context) {
        return new ConcatForwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
