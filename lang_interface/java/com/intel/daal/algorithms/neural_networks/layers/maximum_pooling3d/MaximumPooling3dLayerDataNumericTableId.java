/* file: MaximumPooling3dLayerDataNumericTableId.java */
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
 * @ingroup maximum_pooling3d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.maximum_pooling3d;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__MAXIMUM_POOLING3D__MAXIMUMPOOLING3DLAYERDATANUMERICTABLEID"></a>
 * \brief Identifiers of input objects for the backward three-dimensional maximum pooling layer and
 *        results for the forward three-dimensional maximum pooling layer
 */
public final class MaximumPooling3dLayerDataNumericTableId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public MaximumPooling3dLayerDataNumericTableId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int auxInputDimensionsId = 1;

    public static final MaximumPooling3dLayerDataNumericTableId auxInputDimensions = new MaximumPooling3dLayerDataNumericTableId(
        auxInputDimensionsId);    /*!< Numeric table of size 1 x p that contains the sizes
                                       of the dimensions of the input data tensor */
}
/** @} */
