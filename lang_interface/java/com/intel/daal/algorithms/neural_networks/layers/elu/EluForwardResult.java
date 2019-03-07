/* file: EluForwardResult.java */
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
 * @ingroup elu_layers_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.elu;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELU__ELUFORWARDRESULT"></a>
 * @brief Class that provides methods to access the result obtained with the compute() method of the forward ELU layer
 */
public final class EluForwardResult extends com.intel.daal.algorithms.neural_networks.layers.ForwardResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the forward ELU layer result
     * @param context   Context to manage the forward ELU layer result
     */
    public EluForwardResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public EluForwardResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result of the forward ELU layer
     * @param  id   Identifier of the result
     * @return Result that corresponds to the given identifier
     */
    public Tensor get(EluLayerDataId id) {
        if (id == EluLayerDataId.auxData || id == EluLayerDataId.auxIntermediateValue) {
            return (Tensor)Factory.instance().createObject(getContext(), cGetValue(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the result of the forward ELU layer
     * @param id   Identifier of the result
     * @param val  Result that corresponds to the given identifier
     */
    public void set(EluLayerDataId id, Tensor val) {
        if (id == EluLayerDataId.auxData || id == EluLayerDataId.auxIntermediateValue) {
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
