/* file: FullyConnectedParameter.java */
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
 * @ingroup fullyconnected
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.fullyconnected;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FULLYCONNECTED__FULLYCONNECTEDPARAMETER"></a>
 * \brief Class that specifies parameters of the fully-connected layer
 */
public class FullyConnectedParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {

    public FullyConnectedParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Gets the number of layer outputs
     */
    public long getNOutputs() {
        return cGetNOutputs(cObject);
    }

    /**
     *  Sets the number of layer outputs
     *  @param nOutputs A number of layer outputs
     */
    public void setNOutputs(long nOutputs) {
        cSetNOutputs(cObject, nOutputs);
    }

    private native long cInit(long nOutputs);
    private native long cGetNOutputs(long cParameter);
    private native void cSetNOutputs(long cParameter, long nOutputs);
}
/** @} */
