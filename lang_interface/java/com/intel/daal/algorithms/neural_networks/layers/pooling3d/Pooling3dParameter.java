/* file: Pooling3dParameter.java */
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
 * @ingroup pooling3d
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.pooling3d;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING3D__POOLING3DPARAMETER"></a>
 * \brief Class that specifies parameters of the pooling layer
 */
public class Pooling3dParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {
    /** @private */
    public Pooling3dParameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
    *  Gets the data structure representing the sizes of the three-dimensional kernel subtensor
    * @return Data structure representing the sizes of the three-dimensional kernel subtensor
    */
    public Pooling3dKernelSizes getKernelSizes() {
        long[] size = cGetKernelSizes(cObject);
        return new Pooling3dKernelSizes(size[0], size[1], size[2]);
    }

    /**
     *  Sets the data structure representing the sizes of the three-dimensional kernel subtensor
     *  @param ks   The data structure representing the sizes of the three-dimensional kernel subtensor
     */
    public void setKernelSizes(Pooling3dKernelSizes ks) {
        long[] size = ks.getSize();
        cSetKernelSizes(cObject, size[0], size[1], size[2]);
    }

    /**
    *  Gets the data structure representing the intervals on which the subtensors for pooling are selected
    * @return Data structure representing the intervals on which the subtensors for pooling are selected
    */
    public Pooling3dStrides getStrides() {
        long[] size = cGetStrides(cObject);
        return new Pooling3dStrides(size[0], size[1], size[2]);
    }

    /**
     *  Sets the data structure representing the intervals on which the subtensors for pooling are selected
     *  @param str   The data structure representing the intervals on which the subtensors for pooling are selected
     */
    public void setStrides(Pooling3dStrides str) {
        long[] size = str.getSize();
        cSetStrides(cObject, size[0], size[1], size[2]);
    }

    /**
    *  Gets the data structure representing the number of data elements to implicitly add to each size
    *  of the three-dimensional subtensor on which pooling is performed
    * @return Data structure representing the number of data elements to implicitly add to each size
    *         of the three-dimensional subtensor on which pooling is performed
    */
    public Pooling3dPaddings getPaddings() {
        long[] size = cGetPaddings(cObject);
        return new Pooling3dPaddings(size[0], size[1], size[2]);
    }

    /**
    *  Sets the data structure representing the number of data elements to implicitly add to each size
    *  of the three-dimensional subtensor on which pooling is performed
    *  @param pad   The data structure representing the number of data elements to implicitly add to each size
    *               of the three-dimensional subtensor on which pooling is performed
    */
    public void setPaddings(Pooling3dPaddings pad) {
        long[] size = pad.getSize();
        cSetPaddings(cObject, size[0], size[1], size[2]);
    }

    /**
    *  Gets the data structure representing the dimension for convolution kernels
    * @return Data structure representing the dimension for convolution kernels
    */
    public Pooling3dIndices getIndices() {
        long[] size = cGetSD(cObject);
        return new Pooling3dIndices(size[0], size[1], size[2]);
    }

    /**
     *  Sets the data structure representing the dimension for convolution kernels
     *  @param sd   The data structure representing the dimension for convolution kernels
     */
    public void setIndices(Pooling3dIndices sd) {
        long[] size = sd.getSize();
        cSetSD(cObject, size[0], size[1], size[2]);
    }

    private native long cInit(long index, long kernelSize, long stride, long padding, boolean predictionStage);
    private native void cSetKernelSizes(long cObject, long first, long second, long third);
    private native void cSetStrides(long cObject, long first, long second, long third);
    private native void cSetPaddings(long cObject, long first, long second, long third);
    private native void cSetSD(long cObject, long first, long second, long third);
    private native long[] cGetKernelSizes(long cObject);
    private native long[] cGetStrides(long cObject);
    private native long[] cGetPaddings(long cObject);
    private native long[] cGetSD(long cObject);
}
/** @} */
