/* file: LcnBatch.java */
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
 * @defgroup lcn_layers Local contrast normalization (LCN) Layer
 * @brief Contains classes for neural network local contrast normalization layer
 * @ingroup layers
 * @{
 */
/**
 * @brief Contains classes of the local contrast normalization layer
 */
package com.intel.daal.algorithms.neural_networks.layers.lcn;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LCN__LCNBATCH"></a>
 * @brief Provides methods for the local contrast normalization layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-LCNFORWARD-ALGORITHM">Forward local contrast normalization layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-LCNBACKWARD-ALGORITHM">Backward local contrast normalization layer description and usage models</a> -->
 *
 * @par References
 *      - @ref LcnForwardBatch class
 *      - @ref LcnBackwardBatch class
 */
public class LcnBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    public    LcnMethod        method;      /*!< Computation method for the layer */
    public    LcnParameter     parameter;   /*!< LCN layer parameters */
    protected Precision     prec;        /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the local contrast normalization layer
     * @param context    Context to manage the local contrast normalization layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref LcnMethod
     */
    public LcnBatch(DaalContext context, Class<? extends Number> cls, LcnMethod method) {
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
        parameter = new LcnParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));

        forwardLayer = (ForwardLayer)(new LcnForwardBatch(context, cls, method, cGetForwardLayer(cObject, prec.getValue(), method.getValue())));
        backwardLayer = (BackwardLayer)(new LcnBackwardBatch(context, cls, method, cGetBackwardLayer(cObject, prec.getValue(), method.getValue())));
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetForwardLayer(long cAlgorithm, int prec, int method);
    private native long cGetBackwardLayer(long cAlgorithm, int prec, int method);
}
/** @} */
