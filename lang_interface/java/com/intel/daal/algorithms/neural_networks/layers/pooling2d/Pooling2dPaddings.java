/* file: Pooling2dPaddings.java */
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
 * @ingroup pooling2d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.pooling2d;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING2D__POOLING2DPADDINGS"></a>
 * \brief Data structure representing the number of data elements to implicitly add
 *        to each side of the 2D subtensor on which two-dimensional pooling is performed
 */
public final class Pooling2dPaddings {
    private long[] size;     /*!< Array of numbers of data elements to implicitly add to each size of
                                  the 2D subtensor on which two-dimensional pooling is performed */
    /**
    * Constructs Paddings with parameters
    * @param first   Number of data elements to add to the the first dimension of the 2D subtensor
    * @param second  Number of data elements to add to the the second dimension of the 2D subtensor
    */
    public Pooling2dPaddings(long first, long second) {
        size = new long[2];
        size[0] = first;
        size[1] = second;
    }

    /**
    *  Sets the array of numbers of data elements to implicitly add to each size of
    *  the two-dimensional subtensor on which two-dimensional pooling is performed
    * @param first   Number of data elements to add to the the first dimension of the 2D subtensor
    * @param second  Number of data elements to add to the the second dimension of the 2D subtensor
    */
    public void setSize(long first, long second) {
        size[0] = first;
        size[1] = second;
    }

    /**
    *  Gets the array of numbers of data elements to implicitly add to each size of
    *  the two-dimensional subtensor on which two-dimensional pooling is performed
    * @return Array of numbers of data elements to implicitly add to each size of
    *         he two-dimensional subtensor on which two-dimensional pooling is performed
    */
    public long[] getSize() {
        return size;
    }
}
/** @} */
