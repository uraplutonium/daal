/* file: SpatialStochasticPooling2dBatch.java */
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
 * @defgroup spatial_stochastic_pooling2d Two-dimensional Spatial pyramid stochastic Pooling Layer
 * @brief Contains classes for spatial pyramid stochastic two-dimensional (2D) pooling layer
 * @ingroup spatial_pooling2d
 * @{
 */
/**
 * @brief Contains classes of the two-dimensional (2D) spatial stochastic pooling layer
 */
package com.intel.daal.algorithms.neural_networks.layers.spatial_stochastic_pooling2d;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_STOCHASTIC_POOLING2D__SPATIALSTOCHASTICPOOLING2DBATCH"></a>
 * @brief Provides methods for the two-dimensional spatial stochastic pooling layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-STOCHASTICPOOLING2DFORWARD-ALGORITHM">Forward two-dimensional spatial stochastic pooling layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-STOCHASTICPOOLING2DBACKWARD-ALGORITHM">Backward two-dimensional spatial stochastic pooling layer description and usage models</a> -->
 *
 * @par References
 *      - @ref SpatialStochasticPooling2dForwardBatch class
 *      - @ref SpatialStochasticPooling2dBackwardBatch class
 */
public class SpatialStochasticPooling2dBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    public    SpatialStochasticPooling2dMethod        method;      /*!< Computation method for the layer */
    public    SpatialStochasticPooling2dParameter     parameter;   /*!< Pooling layer parameters */
    protected Precision     prec;        /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the two-dimensional spatial stochastic pooling layer
     * @param context        Context to manage the two-dimensional spatial stochastic pooling layer
     * @param cls            Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method         The layer computation method, @ref SpatialStochasticPooling2dMethod
     * @param pyramidHeight  The value of pyramid height
     * @param nDim           Number of dimensions in input data
     */
    public SpatialStochasticPooling2dBatch(DaalContext context, Class<? extends Number> cls, SpatialStochasticPooling2dMethod method, long pyramidHeight, long nDim) {
        super(context);

        this.method = method;

        if (method != SpatialStochasticPooling2dMethod.defaultDense) {
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

        this.cObject = cInit(prec.getValue(), method.getValue(), pyramidHeight, nDim);
        parameter = new SpatialStochasticPooling2dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));

        forwardLayer = (ForwardLayer)(new SpatialStochasticPooling2dForwardBatch(context, cls, method, cGetForwardLayer(cObject, prec.getValue(), method.getValue()), pyramidHeight, nDim));
        backwardLayer = (BackwardLayer)(new SpatialStochasticPooling2dBackwardBatch(context, cls, method, cGetBackwardLayer(cObject, prec.getValue(),
                                                                                                  method.getValue()), pyramidHeight, nDim));
    }

    private native long cInit(int prec, int method, long pyramidHeight, long nDim);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetForwardLayer(long cAlgorithm, int prec, int method);
    private native long cGetBackwardLayer(long cAlgorithm, int prec, int method);
}
/** @} */
