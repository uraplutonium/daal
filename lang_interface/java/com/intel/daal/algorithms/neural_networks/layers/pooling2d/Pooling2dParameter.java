/* file: Pooling2dParameter.java */
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

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING2D__POOLING2DPARAMETER"></a>
 * \brief Class that specifies parameters of the two-dimensional pooling layer
 */
public class Pooling2dParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {
    /** @private */
    public Pooling2dParameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
    *  Gets the data structure representing the size of the 2D subtensor from which the element is computed
    * @return Data structure representing the size of the 2D subtensor from which the element is computed
    */
    public Pooling2dKernelSizes getKernelSizes() {
        long[] size = cGetKernelSizes(cObject);
        return new Pooling2dKernelSizes(size[0], size[1]);
    }

    /**
     *  Sets the data structure representing the size of the 2D subtensor from which the element is computed
     *  @param ks   The data structure representing the size of the 2D subtensor from which the element is computed
     */
    public void setKernelSizes(Pooling2dKernelSizes ks) {
        long[] size = ks.getSize();
        cSetKernelSizes(cObject, size[0], size[1]);
    }

    /**
    *  Gets the data structure representing the intervals on which the subtensors for two-dimensional pooling are computed
    * @return Data structure representing the intervals on which the subtensors for two-dimensional pooling are selected
    */
    public Pooling2dStrides getStrides() {
        long[] size = cGetStrides(cObject);
        return new Pooling2dStrides(size[0], size[1]);
    }

    /**
     *  Sets the data structure representing the intervals on which the subtensors for two-dimensional pooling are selected
     *  @param str   The data structure representing the intervals on which the subtensors for two-dimensional pooling are selected
     */
    public void setStrides(Pooling2dStrides str) {
        long[] size = str.getSize();
        cSetStrides(cObject, size[0], size[1]);
    }

    /**
    *  Gets the structure representing the number of data elements to implicitly add
    *        to each side of the 2D subtensor on which two-dimensional pooling is performed
    * @return Data structure representing the number of data elements to implicitly add to each size
    *         of the two-dimensional subtensor on which two-dimensional pooling is performed
    */
    public Pooling2dPaddings getPaddings() {
        long[] size = cGetPaddings(cObject);
        return new Pooling2dPaddings(size[0], size[1]);
    }

    /**
    *  Sets the data structure representing the number of data elements to implicitly add to each size
    *  of the two-dimensional subtensor on which two-dimensional pooling is performed
    *  @param pad   The data structure representing the number of data elements to implicitly add to each size
    *                      of the two-dimensional subtensor on which two-dimensional pooling is performed
    */
    public void setPaddings(Pooling2dPaddings pad) {
        long[] size = pad.getSize();
        cSetPaddings(cObject, size[0], size[1]);
    }

    /**
    *  Gets the data structure representing the indices of the dimension on which two-dimensional pooling is performed
    * @return Data structure representing the indices of the dimension on which two-dimensional pooling is performed
    */
    public Pooling2dIndices getIndices() {
        long[] size = cGetSD(cObject);
        return new Pooling2dIndices(size[0], size[1]);
    }

    /**
     *  Sets the data structure representing the indices of the dimension on which two-dimensional pooling is performed
     *  @param sd   The data structure representing the indices of the dimension on which two-dimensional pooling is performed
     */
    public void setIndices(Pooling2dIndices sd) {
        long[] size = sd.getSize();
        cSetSD(cObject, size[0], size[1]);
    }

    private native long cInit(long index, long kernelSize, long stride, long padding, boolean predictionStage);
    private native void cSetKernelSizes(long cObject, long first, long second);
    private native void cSetStrides(long cObject, long first, long second);
    private native void cSetPaddings(long cObject, long first, long second);
    private native void cSetSD(long cObject, long first, long second);
    private native long[] cGetKernelSizes(long cObject);
    private native long[] cGetStrides(long cObject);
    private native long[] cGetPaddings(long cObject);
    private native long[] cGetSD(long cObject);
}
/** @} */
