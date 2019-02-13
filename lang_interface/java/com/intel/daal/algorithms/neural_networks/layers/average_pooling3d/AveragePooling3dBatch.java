/* file: AveragePooling3dBatch.java */
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
 * @defgroup average_pooling3d Three-dimensional Average Pooling Layer
 * @brief Contains classes for average three-dimensional (3D) pooling layer
 * @ingroup pooling3d
 * @{
 */
/**
 * @brief Contains classes of the three-dimensional (3D) average pooling layer
 */
package com.intel.daal.algorithms.neural_networks.layers.average_pooling3d;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__AVERAGE_POOLING3D__AVERAGEPOOLING3DBATCH"></a>
 * @brief Provides methods for the three-dimensional average pooling layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-AVERAGEPOOLING3DFORWARD-ALGORITHM">Forward three-dimensional average pooling layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-AVERAGEPOOLING3DBACKWARD-ALGORITHM">Backward three-dimensional average pooling layer description and usage models</a> -->
 *
 * @par References
 *      - @ref AveragePooling3dForwardBatch class
 *      - @ref AveragePooling3dBackwardBatch class
 */
public class AveragePooling3dBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    public    AveragePooling3dMethod        method;      /*!< Computation method for the layer */
    public    AveragePooling3dParameter     parameter;   /*!< Pooling layer parameters */
    protected Precision                     prec;        /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the three-dimensional average pooling layer
     * @param context    Context to manage the three-dimensional average pooling layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref AveragePooling3dMethod
     * @param nDim       Number of dimensions in input data
     */
    public AveragePooling3dBatch(DaalContext context, Class<? extends Number> cls, AveragePooling3dMethod method, long nDim) {
        super(context);

        this.method = method;

        if (method != AveragePooling3dMethod.defaultDense) {
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

        this.cObject = cInit(prec.getValue(), method.getValue(), nDim);
        parameter = new AveragePooling3dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));

        forwardLayer = (ForwardLayer)(new AveragePooling3dForwardBatch(context, cls, method, cGetForwardLayer(cObject, prec.getValue(), method.getValue()), nDim));
        backwardLayer = (BackwardLayer)(new AveragePooling3dBackwardBatch(context, cls, method, cGetBackwardLayer(cObject, prec.getValue(),
                                                                                                  method.getValue()), nDim));
    }

    private native long cInit(int prec, int method, long nDim);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetForwardLayer(long cAlgorithm, int prec, int method);
    private native long cGetBackwardLayer(long cAlgorithm, int prec, int method);
}
/** @} */
