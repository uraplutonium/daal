/* file: Pooling2dBackwardInput.java */
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
 * @defgroup pooling2d_backward Backward Two-dimensional Pooling Layer
 * @brief Contains classes for backward two-dimensional (2D) pooling layer
 * @ingroup pooling2d
 * @{
 */
/**
 * @brief Contains classes of the two-dimensional (2D) pooling layers
 */
package com.intel.daal.algorithms.neural_networks.layers.pooling2d;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING2D__POOLING2DBACKWARDINPUT"></a>
 * @brief Input object for the backward two-dimensional pooling layer
 */
public class Pooling2dBackwardInput extends com.intel.daal.algorithms.neural_networks.layers.BackwardInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Pooling2dBackwardInput(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */
