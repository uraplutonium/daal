/* file: Convolution2dIndices.java */
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
 * @ingroup convolution2d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.convolution2d;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONVOLUTION2D__CONVOLUTION2DINDICES"></a>
 * \brief Data structure representing the dimension for convolution kernels
 */
public final class Convolution2dIndices {
    private long[] size;     /*!< Array of dimensions for convolution kernels */

    /**
    * Constructs Convolution2dIndices with parameters
    * @param first  The first dimension for convolution kernels
    * @param second The second dimension for convolution kernels
    */
    public Convolution2dIndices(long first, long second) {
        size = new long[2];
        size[0] = first;
        size[1] = second;
    }

    /**
     *  Sets the array of dimensions for convolution kernels
    * @param first  The first dimension for convolution kernels
    * @param second The second dimension for convolution kernels
    */
    public void setSize(long first, long second) {
        size[0] = first;
        size[1] = second;
    }

    /**
    *  Gets the array of dimensions for convolution kernels
    * @return Array of dimensions for convolution kernels
    */
    public long[] getSize() {
        return size;
    }
}
/** @} */
