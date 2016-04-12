/* file: LayerDataId.java */
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

package com.intel.daal.algorithms.neural_networks.layers.prelu;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__PRELU__LAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward prelu layer and results for the forward prelu layer
 */
public final class LayerDataId {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    private int _value;

    /**
     * Constructs the result object identifier using the provided value
     * @param value     Value of the input object identifier
     */
    public LayerDataId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result identifier
     * @return Value corresponding to the result identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int auxDataId = 0;
    private static final int auxWeightsId = 1;

    public static final LayerDataId auxData    = new LayerDataId(auxDataId);    /*!< Data processed at the forward stage of the layer */
    public static final LayerDataId auxWeights = new LayerDataId(auxWeightsId); /*!< Weights of the prelu layer */
}
