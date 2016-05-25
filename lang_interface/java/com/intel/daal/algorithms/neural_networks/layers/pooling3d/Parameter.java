/* file: Parameter.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

package com.intel.daal.algorithms.neural_networks.layers.pooling3d;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING3D__PARAMETER"></a>
 * \brief Class that specifies parameters of the pooling layer
 */
public class Parameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {
    /** @private */
    public Parameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
    *  Gets the number of groups which the input data is split in groupDimension dimension
    * @return Number of groups which the input data is split in groupDimension dimension
    */
    public boolean getPredictionStage() {
        return cGetPredictionStage(cObject);
    }

    /**
     *  Sets the number of groups which the input data is split in groupDimension dimension
     *  @param predictionStage   The number of groups which the input data is split in groupDimension dimension
     */
    public void setPredictionStage(boolean predictionStage) {
        cSetPredictionStage(cObject, predictionStage);
    }

    /**
    *  Gets the data structure representing the sizes of the three-dimensional kernel subtensor
    * @return Data structure representing the sizes of the three-dimensional kernel subtensor
    */
    public KernelSize getKernelSize() {
        long[] size = cGetKernelSize(cObject);
        return new KernelSize(size[0], size[1], size[2]);
    }

    /**
     *  Sets the data structure representing the sizes of the three-dimensional kernel subtensor
     *  @param ks   The data structure representing the sizes of the three-dimensional kernel subtensor
     */
    public void setKernelSize(KernelSize ks) {
        long[] size = ks.getSize();
        cSetKernelSize(cObject, size[0], size[1], size[2]);
    }

    /**
    *  Gets the data structure representing the intervals on which the subtensors for pooling are selected
    * @return Data structure representing the intervals on which the subtensors for pooling are selected
    */
    public Stride getStride() {
        long[] size = cGetStride(cObject);
        return new Stride(size[0], size[1], size[2]);
    }

    /**
     *  Sets the data structure representing the intervals on which the subtensors for pooling are selected
     *  @param str   The data structure representing the intervals on which the subtensors for pooling are selected
     */
    public void setStride(Stride str) {
        long[] size = str.getSize();
        cSetStride(cObject, size[0], size[1], size[2]);
    }

    /**
    *  Gets the data structure representing the number of data elements to implicitly add to each size
    *  of the three-dimensional subtensor on which pooling is performed
    * @return Data structure representing the number of data elements to implicitly add to each size
    *         of the three-dimensional subtensor on which pooling is performed
    */
    public Padding getPadding() {
        long[] size = cGetPadding(cObject);
        return new Padding(size[0], size[1], size[2]);
    }

    /**
    *  Sets the data structure representing the number of data elements to implicitly add to each size
    *  of the three-dimensional subtensor on which pooling is performed
    *  @param pad   The data structure representing the number of data elements to implicitly add to each size
    *               of the three-dimensional subtensor on which pooling is performed
    */
    public void setPadding(Padding pad) {
        long[] size = pad.getSize();
        cSetPadding(cObject, size[0], size[1], size[2]);
    }

    /**
    *  Gets the data structure representing the dimension for convolution kernels
    * @return Data structure representing the dimension for convolution kernels
    */
    public SpatialDimensions getSpatialDimensions() {
        long[] size = cGetSD(cObject);
        return new SpatialDimensions(size[0], size[1], size[2]);
    }

    /**
     *  Sets the data structure representing the dimension for convolution kernels
     *  @param sd   The data structure representing the dimension for convolution kernels
     */
    public void setSpatialDimensions(SpatialDimensions sd) {
        long[] size = sd.getSize();
        cSetSD(cObject, size[0], size[1], size[2]);
    }

    private native long cInit(long index, long kernelSize, long stride, long padding, boolean predictionStage);
    private native boolean cGetPredictionStage(long cObject);
    private native void cSetPredictionStage(long cObject, boolean predictionStage);
    private native void cSetKernelSize(long cObject, long first, long second, long third);
    private native void cSetStride(long cObject, long first, long second, long third);
    private native void cSetPadding(long cObject, long first, long second, long third);
    private native void cSetSD(long cObject, long first, long second, long third);
    private native long[] cGetKernelSize(long cObject);
    private native long[] cGetStride(long cObject);
    private native long[] cGetPadding(long cObject);
    private native long[] cGetSD(long cObject);
}
