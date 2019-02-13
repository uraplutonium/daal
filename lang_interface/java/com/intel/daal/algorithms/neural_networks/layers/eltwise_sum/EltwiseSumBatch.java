/* file: EltwiseSumBatch.java */
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
 * @defgroup eltwise_sum Element-wise Sum Layer
 * @brief Contains classes for the element-wise sum layer
 * @ingroup layers
 * @{
 */
/**
 * @brief Contains classes of the element-wise sum layer
 */
package com.intel.daal.algorithms.neural_networks.layers.eltwise_sum;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELTWISE_SUM__ELTWISESUMBATCH"></a>
 * @brief Provides methods for the element-wise sum layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-ELTWISESUMFORWARD-ALGORITHM">Forward element-wise sum layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-ELTWISESUMBACKWARD-ALGORITHM">Backward element-wise sum layer description and usage models</a> -->
 *
 * @par References
 *      - @ref EltwiseSumForwardBatch class
 *      - @ref EltwiseSumBackwardBatch class
 */
public class EltwiseSumBatch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    public    EltwiseSumMethod        method;      /*!< Computation method for the layer */
    public    EltwiseSumParameter     parameter;   /*!< Element-wise sum layer parameters */
    protected Precision               prec;        /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the element-wise sum layer
     * @param context    Context to manage the element-wise sum layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref EltwiseSumMethod
     */
    public EltwiseSumBatch(DaalContext context, Class<? extends Number> cls, EltwiseSumMethod method) {
        super(context);

        this.method = method;

        if (method != EltwiseSumMethod.defaultDense) {
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

        parameter = new EltwiseSumParameter(context,
            cInitParameter(cObject, prec.getValue(), method.getValue()) );

        forwardLayer = (ForwardLayer)(new EltwiseSumForwardBatch(context, cls, method,
            cGetForwardLayer(cObject, prec.getValue(), method.getValue()) ));

        backwardLayer = (BackwardLayer)(new EltwiseSumBackwardBatch(context, cls, method,
            cGetBackwardLayer(cObject, prec.getValue(), method.getValue()) ));
    }

    private native long cInit(int prec, int method);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetForwardLayer(long cAlgorithm, int prec, int method);
    private native long cGetBackwardLayer(long cAlgorithm, int prec, int method);
}
/** @} */
