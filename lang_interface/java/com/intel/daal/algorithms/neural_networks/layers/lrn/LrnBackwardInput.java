/* file: LrnBackwardInput.java */
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
 * @defgroup lrn_forward Forward Local Response Normalization Layer
 * @brief Contains classes for the forward local response normalization layer
 * @ingroup lrn
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.lrn;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LRN__LRNBACKWARDINPUT"></a>
 * @brief Input object for the backward local response normalization layer
 */
public final class LrnBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.BackwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public LrnBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object of the backward local response normalization layer
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(LrnLayerDataId id, Tensor val) {
        if (id == LrnLayerDataId.auxData || id == LrnLayerDataId.auxSmBeta) {
            cSetInput(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("Incorrect LrnBackwardInputId");
        }
    }

    /**
     * Returns the input object of the backward local response normalization layer
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public Tensor get(LrnLayerDataId id) {
        if (id == LrnLayerDataId.auxData || id == LrnLayerDataId.auxSmBeta) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetInput(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetInput(long cObject, int id, long ntAddr);
    private native long cGetInput(long cObject, int id);
}
/** @} */
