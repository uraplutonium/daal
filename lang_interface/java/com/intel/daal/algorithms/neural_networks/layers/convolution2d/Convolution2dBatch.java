/* file: Convolution2dBatch.java */
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
 * @defgroup convolution2d Two-dimensional Convolution Layer
 * @brief Contains classes for neural network 2D convolution layer
 * @ingroup layers
 * @{
 */
/**
 * @brief Contains classes of the two-dimensional (2D) convolution layer
 */
package com.intel.daal.algorithms.neural_networks.layers.convolution2d;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONVOLUTION2D__CONVOLUTION2DBATCH"></a>
 * @brief Provides methods for the 2D convolution layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-CONVOLUTION2DFORWARD-ALGORITHM">Forward 2D convolution layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-CONVOLUTION2DBACKWARD-ALGORITHM">Backward 2D convolution layer description and usage models</a> -->
 *
 * @par References
 *      - @ref Convolution2dForwardBatch class
 *      - @ref Convolution2dBackwardBatch class
 */
public class Convolution2dBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    public    Convolution2dMethod        method;      /*!< Computation method for the layer */
    public    Convolution2dParameter     parameter;   /*!< Convolution layer parameters */
    protected Precision                  prec;        /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the 2D convolution layer
     * @param context    Context to manage the 2D convolution layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref Convolution2dMethod
     */
    public Convolution2dBatch(DaalContext context, Class<? extends Number> cls, Convolution2dMethod method) {
        super(context);

        this.method = method;

        if (method != Convolution2dMethod.defaultDense) {
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
        parameter = new Convolution2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));

        forwardLayer = (ForwardLayer)(new Convolution2dForwardBatch(context, cls, method, cGetForwardLayer(cObject, prec.getValue(), method.getValue())));
        backwardLayer = (BackwardLayer)(new Convolution2dBackwardBatch(context, cls, method, cGetBackwardLayer(cObject, prec.getValue(), method.getValue())));
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetForwardLayer(long cAlgorithm, int prec, int method);
    private native long cGetBackwardLayer(long cAlgorithm, int prec, int method);
}
/** @} */
