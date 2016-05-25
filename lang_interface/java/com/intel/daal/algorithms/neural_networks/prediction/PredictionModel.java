/* file: PredictionModel.java */
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

package com.intel.daal.algorithms.neural_networks.prediction;

import com.intel.daal.algorithms.neural_networks.ForwardLayers;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.BackwardLayers;
import com.intel.daal.algorithms.neural_networks.NextLayersCollection;
import com.intel.daal.algorithms.neural_networks.LayerDescriptors;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__MODEL"></a>
 * @brief Class Model object for the prediction stage of neural network algorithm
 */
public class PredictionModel extends com.intel.daal.algorithms.Model {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs model for the prediction stage of the neural network
     * @param context    Context to manage the model
     */
    public PredictionModel(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    /**
     * Constructs model for the prediction stage of the neural network by copying parameters of another model
     * @param context    Context to manage the model
     * @param other      A model to be used as the source to initialize the parameters of the model
     */
    public PredictionModel(DaalContext context, PredictionModel other) {
        super(context);
        cObject = cInit(other.cObject);
    }

    /**
     * Constructs model for the prediction stage of the neural network using collections of forward stage of layers and connections between layers
     * @param context              Context to manage the model
     * @param forwardLayers        List of forward stages of the layers
     * @param nextLayersCollection List of next layers for each layer
     */
    public PredictionModel(DaalContext context, ForwardLayers forwardLayers, NextLayersCollection nextLayersCollection) {
        super (context);
        cObject = cInit(forwardLayers.cObject, nextLayersCollection.cObject);
    }

    /**
     * Constructs model for the prediction stage of the neural network from the layer descriptors
     * @param context           Context to manage the model
     * @param layerDescriptors  Collection of layer descriptors of every inserted layer
     */
    public PredictionModel(DaalContext context, LayerDescriptors layerDescriptors) {
        super(context);
        cObject = cInitFromLayerDescriptors(layerDescriptors.cObject);
    }

    public PredictionModel(DaalContext context, long cModel) {
        super(context, cModel);
    }

    /**
     * Sets the list of forward layers and the list of connections between layers
     * @param forwardLayers  List of forward layers
     * @param nextLayersCollection  List of next layers for each layer with corresponding index
     */
    public void setLayers(ForwardLayers forwardLayers, NextLayersCollection nextLayersCollection) {
        cSetLayers(cObject, forwardLayers.cObject, nextLayersCollection.cObject);
    }

    /**
     * Returns list of forward layers
     * @return List of forward layers
     */
    public ForwardLayers getLayers() {
        return new ForwardLayers(getContext(), cGetForwardLayers(cObject));
    }

    /**
     * Returns the forward stage of a layer with certain index in the network
     * @param index  Index of the layer in the network
     * @return Forward stage of a layer with certain index in the network
     */
    public ForwardLayer getLayer(long index) {
        return new ForwardLayer(getContext(), cGetForwardLayer(cObject, index));
    }

    /**
     * Returns list of connections between layers
     * @return List of next layers for each layer with corresponding index
     */
    public NextLayersCollection getNextLayers() {
        return new NextLayersCollection(getContext(), cGetNextLayers(cObject));
    }

    private native long cInit();
    private native long cInit(long cModel);
    private native long cInit(long forwardLayersAddr, long nextLayersCollectionAddr);
    private native long cInitFromLayerDescriptors(long cLayerDescriptors);
    private native void cSetLayers(long cModel, long forwardLayersAddr, long nextLayersCollectionAddr);
    private native long cGetForwardLayers(long cModel);
    private native long cGetForwardLayer(long cModel, long index);
    private native long cGetNextLayers(long cModel);
}
