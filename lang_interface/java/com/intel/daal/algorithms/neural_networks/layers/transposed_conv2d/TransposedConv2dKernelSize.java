/* file: TransposedConv2dKernelSize.java */
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
 * @ingroup transposed_conv2d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.transposed_conv2d;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__TRANSPOSED_CONV2D__TRANSPOSEDCONV2DKERNELSIZE"></a>
 * \brief Data structure representing the sizes of the two-dimensional kernel subtensor for the backward 2D transposed convolution
  *       layer and results for the forward 2D transposed convolution layer
 */
public final class TransposedConv2dKernelSize {
    private long[] size; /*!< Array of sizes of the two-dimensional kernel subtensor */

    /**
    * Constructs TransposedConv2dKernelSize with parameters
    * @param first  The first size of the two-dimensional kernel subtensor
    * @param second The second size of the two-dimensional kernel subtensor
    */
    public TransposedConv2dKernelSize(long first, long second) {
        size = new long[2];
        size[0] = first;
        size[1] = second;
    }

    /**
     *  Sets the the array of sizes of the two-dimensional kernel subtensor
    * @param first  The first size of the two-dimensional kernel subtensor
    * @param second The second size of the two-dimensional kernel subtensor
     */
    public void setSize(long first, long second) {
        size[0] = first;
        size[1] = second;
    }

    /**
    *  Gets the array of sizes of the two-dimensional kernel subtensor
    * @return Array of sizes of the two-dimensional kernel subtensor
    */
    public long[] getSize() {
        return size;
    }
}
/** @} */
