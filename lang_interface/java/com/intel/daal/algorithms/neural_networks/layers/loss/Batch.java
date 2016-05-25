/* file: Batch.java */
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

/**
 * @brief Contains classes of the loss layer
 */
package com.intel.daal.algorithms.neural_networks.layers.loss;

import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__BATCH"></a>
 * @brief Provides methods for the loss layer in the batch processing mode
 *
 * @par References
 *      - <a href="DAAL-REF-LOSSFORWARD-ALGORITHM">Forward loss layer description and usage models</a>
 *      - @ref ForwardBatch class
 *      - <a href="DAAL-REF-LOSSBACKWARD-ALGORITHM">Backward loss layer description and usage models</a>
 *      - @ref BackwardBatch class
 */
public class Batch extends com.intel.daal.algorithms.neural_networks.layers.LayerIface {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the loss layer
     * @param context    Context to manage the loss layer
     * @param cObject    Address of C++ object
     */
    public Batch(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
    * Constructs the loss layer
    * @param context    Context to manage the loss layer
    */
    public Batch(DaalContext context) {
        super(context);
    }
}
