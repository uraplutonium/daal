/* file: LogisticCrossForwardBatch.java */
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
 * @defgroup logistic_cross_forward_batch Batch
 * @ingroup logistic_cross_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.logistic_cross;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOGISTIC_CROSS__LOGISTICCROSSFORWARDBATCH"></a>
 * \brief Class that computes the results of the forward logistic cross-entropy layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-LOGISTIC_CROSSFORWARD-ALGORITHM">Forward logistic cross-entropy layer description and usage models</a> -->
 *
 * \par References
 *      - @ref LogisticCrossLayerDataId class
 */
public class LogisticCrossForwardBatch extends com.intel.daal.algorithms.neural_networks.layers.loss.LossForwardBatch {
    public  LogisticCrossForwardInput input;     /*!< %Input data */
    public  LogisticCrossMethod       method;    /*!< Computation method for the layer */
    public  LogisticCrossParameter    parameter; /*!< LogisticCrossParameters of the layer */
    private Precision    prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the forward logistic cross-entropy layer by copying input objects of another forward logistic cross-entropy layer
     * @param context    Context to manage the forward logistic cross-entropy layer
     * @param other      A forward logistic cross-entropy layer to be used as the source to initialize the input objects of the forward logistic cross-entropy layer
     */
    public LogisticCrossForwardBatch(DaalContext context, LogisticCrossForwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new LogisticCrossForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new LogisticCrossParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the forward logistic cross-entropy layer
     * @param context    Context to manage the layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref LogisticCrossMethod
     */
    public LogisticCrossForwardBatch(DaalContext context, Class<? extends Number> cls, LogisticCrossMethod method) {
        super(context);

        this.method = method;

        if (method != LogisticCrossMethod.defaultDense) {
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
        input = new LogisticCrossForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new LogisticCrossParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    LogisticCrossForwardBatch(DaalContext context, Class<? extends Number> cls, LogisticCrossMethod method, long cObject) {
        super(context, cObject);

        this.method = method;

        if (method != LogisticCrossMethod.defaultDense) {
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
        input = new LogisticCrossForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new LogisticCrossParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of the forward logistic cross-entropy layer
     * @return  Forward logistic cross-entropy layer result
     */
    @Override
    public LogisticCrossForwardResult compute() {
        super.compute();
        LogisticCrossForwardResult result = new LogisticCrossForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the forward logistic cross-entropy layer
     * @param result    Structure to store the result of the forward logistic cross-entropy layer
     */
    public void setResult(LogisticCrossForwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the forward layer
     * @return Structure that contains result of the forward layer
     */
    @Override
    public LogisticCrossForwardResult getLayerResult() {
        return new LogisticCrossForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the forward layer
     * @return Structure that contains input object of the forward layer
     */
    @Override
    public LogisticCrossForwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the forward layer
     * @return Structure that contains parameters of the forward layer
     */
    @Override
    public LogisticCrossParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated forward logistic cross-entropy layer
     * with a copy of input objects of this forward logistic cross-entropy layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated forward logistic cross-entropy layer
     */
    @Override
    public LogisticCrossForwardBatch clone(DaalContext context) {
        return new LogisticCrossForwardBatch(context, this);
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
