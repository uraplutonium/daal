/* file: Pooling1dIndex.java */
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
 * @defgroup pooling1d One-dimensional Pooling Layer
 * @brief Contains classes for the one-dimensional (1D) pooling layer
 * @ingroup layers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.pooling1d;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING1D__POOLING1DINDEX"></a>
 * \brief Data structure representing the indices of the dimension on which one-dimensional pooling is performed
 */
public final class Pooling1dIndex {
    private long[] size;     /*!< Array of indices of the dimension on which one-dimensional pooling is performed */

    /**
    * Constructs Index with parameters
    * @param first  The first dimension index
    */
    public Pooling1dIndex(long first) {
        size = new long[1];
        size[0] = first;
    }

    /**
     *  Sets the array of indices of the dimension on which one-dimensional pooling is performed
    * @param first  The first dimension index
    */
    public void setSize(long first) {
        size[0] = first;
    }

    /**
    *  Gets the array of indices of the dimension on which one-dimensional pooling is performed
    * @return Array of indices of the dimension on which one-dimensional pooling is performed
    */
    public long[] getSize() {
        return size;
    }
}
/** @} */
