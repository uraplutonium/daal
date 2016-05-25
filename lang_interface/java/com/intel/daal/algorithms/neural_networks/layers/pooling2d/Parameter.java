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

package com.intel.daal.algorithms.neural_networks.layers.pooling2d;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING2D__PARAMETER"></a>
 * \brief Class that specifies parameters of the two-dimensional pooling layer
 */
public class Parameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {
    /** @private */
    public Parameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
    *  Gets the flag that specifies whether the layer is used at the prediction stage or not
    * @return  Flag that specifies whether the layer is used at the prediction stage or not
    */
    public boolean getPredictionStage() {
        return cGetPredictionStage(cObject);
    }

    /**
     *  Sets the flag that specifies whether the layer is used at the prediction stage or not
     *  @param predictionStage    Flag that specifies whether the layer is used at the prediction stage or not
     */
    public void setPredictionStage(boolean predictionStage) {
        cSetPredictionStage(cObject, predictionStage);
    }

    /**
    *  Gets the data structure representing the size of the 2D subtensor from which the element is computed
    * @return Data structure representing the size of the 2D subtensor from which the element is computed
    */
    public KernelSize getKernelSize() {
        long[] size = cGetKernelSize(cObject);
        return new KernelSize(size[0], size[1]);
    }

    /**
     *  Sets the data structure representing the size of the 2D subtensor from which the element is computed
     *  @param ks   The data structure representing the size of the 2D subtensor from which the element is computed
     */
    public void setKernelSize(KernelSize ks) {
        long[] size = ks.getSize();
        cSetKernelSize(cObject, size[0], size[1]);
    }

    /**
    *  Gets the data structure representing the intervals on which the subtensors for two-dimensional pooling are computed
    * @return Data structure representing the intervals on which the subtensors for two-dimensional pooling are selected
    */
    public Stride getStride() {
        long[] size = cGetStride(cObject);
        return new Stride(size[0], size[1]);
    }

    /**
     *  Sets the data structure representing the intervals on which the subtensors for two-dimensional pooling are selected
     *  @param str   The data structure representing the intervals on which the subtensors for two-dimensional pooling are selected
     */
    public void setStride(Stride str) {
        long[] size = str.getSize();
        cSetStride(cObject, size[0], size[1]);
    }

    /**
    *  Gets the structure representing the number of data elements to implicitly add
    *        to each side of the 2D subtensor on which two-dimensional pooling is performed
    * @return Data structure representing the number of data elements to implicitly add to each size
    *         of the two-dimensional subtensor on which two-dimensional pooling is performed
    */
    public Padding getPadding() {
        long[] size = cGetPadding(cObject);
        return new Padding(size[0], size[1]);
    }

    /**
    *  Sets the data structure representing the number of data elements to implicitly add to each size
    *  of the two-dimensional subtensor on which two-dimensional pooling is performed
    *  @param pad   The data structure representing the number of data elements to implicitly add to each size
    *                      of the two-dimensional subtensor on which two-dimensional pooling is performed
    */
    public void setPadding(Padding pad) {
        long[] size = pad.getSize();
        cSetPadding(cObject, size[0], size[1]);
    }

    /**
    *  Gets the data structure representing the indices of the dimension on which two-dimensional pooling is performed
    * @return Data structure representing the indices of the dimension on which two-dimensional pooling is performed
    */
    public SpatialDimensions getSpatialDimensions() {
        long[] size = cGetSD(cObject);
        return new SpatialDimensions(size[0], size[1]);
    }

    /**
     *  Sets the data structure representing the indices of the dimension on which two-dimensional pooling is performed
     *  @param sd   The data structure representing the indices of the dimension on which two-dimensional pooling is performed
     */
    public void setSpatialDimensions(SpatialDimensions sd) {
        long[] size = sd.getSize();
        cSetSD(cObject, size[0], size[1]);
    }

    private native long cInit(long index, long kernelSize, long stride, long padding, boolean predictionStage);
    private native boolean cGetPredictionStage(long cObject);
    private native void cSetPredictionStage(long cObject, boolean predictionStage);
    private native void cSetKernelSize(long cObject, long first, long second);
    private native void cSetStride(long cObject, long first, long second);
    private native void cSetPadding(long cObject, long first, long second);
    private native void cSetSD(long cObject, long first, long second);
    private native long[] cGetKernelSize(long cObject);
    private native long[] cGetStride(long cObject);
    private native long[] cGetPadding(long cObject);
    private native long[] cGetSD(long cObject);
}
