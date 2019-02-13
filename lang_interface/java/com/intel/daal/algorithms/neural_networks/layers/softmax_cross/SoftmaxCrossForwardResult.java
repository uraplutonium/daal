/* file: SoftmaxCrossForwardResult.java */
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
 * @ingroup softmax_cross_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.softmax_cross;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SOFTMAX_CROSS__SOFTMAXCROSSFORWARDRESULT"></a>
 * @brief Class that provides methods to access the result obtained with the compute() method of the forward softmax cross-entropy layer
 */
public final class SoftmaxCrossForwardResult extends com.intel.daal.algorithms.neural_networks.layers.loss.LossForwardResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the forward softmax cross-entropy layer result
     * @param context   Context to manage the forward softmax cross-entropy layer result
     */
    public SoftmaxCrossForwardResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public SoftmaxCrossForwardResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result of the forward softmax cross-entropy layer
     * @param  id   Identifier of the result
     * @return Result that corresponds to the given identifier
     */
    public Tensor get(SoftmaxCrossLayerDataId id) {
        if (id == SoftmaxCrossLayerDataId.auxProbabilities || id == SoftmaxCrossLayerDataId.auxGroundTruth) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetValue(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the result of the forward softmax cross-entropy layer
     * @param id   Identifier of the result
     * @param val  Result that corresponds to the given identifier
     */
    public void set(SoftmaxCrossLayerDataId id, Tensor val) {
        if (id == SoftmaxCrossLayerDataId.auxProbabilities || id == SoftmaxCrossLayerDataId.auxGroundTruth) {
            cSetValue(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native long cNewResult();

    private native long cGetValue(long cObject, int id);

    private native void cSetValue(long cObject, int id, long ntAddr);
}
/** @} */
