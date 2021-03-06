/* file: MaximumPooling2dForwardInput.java */
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
 * @defgroup maximum_pooling2d_forward Forward Two-dimensional Max Pooling Layer
 * @brief Contains classes for forward maximum 2D pooling layer
 * @ingroup maximum_pooling2d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.maximum_pooling2d;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__MAXIMUM_POOLING2D__MAXIMUMPOOLING2DFORWARDINPUT"></a>
 * @brief %Input object for the forward two-dimensional maximum pooling layer
 */
public class MaximumPooling2dForwardInput extends com.intel.daal.algorithms.neural_networks.layers.pooling2d.Pooling2dForwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public MaximumPooling2dForwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */
