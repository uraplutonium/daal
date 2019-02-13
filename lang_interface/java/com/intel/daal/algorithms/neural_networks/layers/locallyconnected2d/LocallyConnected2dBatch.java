/* file: LocallyConnected2dBatch.java */
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
 * @defgroup locallyconnected2d Two-dimensional Locally Connected Layer
 * @brief Contains classes for neural network 2D locally connected layer
 * @ingroup layers
 * @{
 */
/**
 * @brief Contains classes of the two-dimensional (2D) locally connected layer
 */
package com.intel.daal.algorithms.neural_networks.layers.locallyconnected2d;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOCALLYCONNECTED2D__LOCALLYCONNECTED2DBATCH"></a>
 * @brief Provides methods for the 2D locally connected layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-LOCALLYCONNECTED2DFORWARD-ALGORITHM">Forward 2D locally connected layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-LOCALLYCONNECTED2DBACKWARD-ALGORITHM">Backward 2D locally connected layer description and usage models</a> -->
 *
 * @par References
 *      - @ref LocallyConnected2dForwardBatch class
 *      - @ref LocallyConnected2dBackwardBatch class
 */
public class LocallyConnected2dBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    public    LocallyConnected2dMethod        method;      /*!< Computation method for the layer */
    public    LocallyConnected2dParameter     parameter;   /*!< Locally connected layer parameters */
    protected Precision     prec;        /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the 2D locally connected layer
     * @param context    Context to manage the 2D locally connected layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref LocallyConnected2dMethod
     */
    public LocallyConnected2dBatch(DaalContext context, Class<? extends Number> cls, LocallyConnected2dMethod method) {
        super(context);

        this.method = method;

        if (method != LocallyConnected2dMethod.defaultDense) {
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
        parameter = new LocallyConnected2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));

        forwardLayer = (ForwardLayer)(new LocallyConnected2dForwardBatch(context, cls, method, cGetForwardLayer(cObject, prec.getValue(), method.getValue())));
        backwardLayer = (BackwardLayer)(new LocallyConnected2dBackwardBatch(context, cls, method, cGetBackwardLayer(cObject, prec.getValue(), method.getValue())));
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetForwardLayer(long cAlgorithm, int prec, int method);
    private native long cGetBackwardLayer(long cAlgorithm, int prec, int method);
}
/** @} */
