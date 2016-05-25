/* file: BackwardBatch.java */
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

package com.intel.daal.algorithms.neural_networks.layers.loss;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__BACKWARDBATCH"></a>
 * \brief Class that computes the results of the backward loss layer in the batch processing mode
 *
 * \par References
 *      - <a href="DAAL-REF-LOSSBACKWARD">Backward loss layer description and usage models</a>
 *      - @ref BackwardInput class
 *      - @ref BackwardResult class
 */
public class BackwardBatch extends com.intel.daal.algorithms.neural_networks.layers.BackwardLayer {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the backward loss layer by copying input objects of backward loss layer
     * @param context    Context to manage the backward loss layer
     * @param other      A backward loss layer to be used as the source to initialize the input objects of the backward loss layer
     */
    public BackwardBatch(DaalContext context, BackwardBatch other) {
        super(context);
    }

    public BackwardBatch(DaalContext context) {
        super(context);
    }

    public BackwardBatch(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Computes the result of the backward loss layer
     * @return  Backward loss layer result
     */
    @Override
    public BackwardResult compute() {
        super.compute();
        BackwardResult result = new BackwardResult(getContext(), cGetResult(cObject));
        return result;
    }

    /**
     * Returns the structure that contains result of the backward layer
     * @return Structure that contains result of the backward layer
     */
    @Override
    public BackwardResult getLayerResult() {
        return new BackwardResult(getContext(), cGetResult(cObject));
    }

    /**
     * Returns the structure that contains input object of the backward layer
     * \return Structure that contains input object of the backward layer
     */
    @Override
    public BackwardInput getLayerInput() {
        return new BackwardInput(getContext(), cGetInput(cObject));
    }
    /**
     * Returns the structure that contains parameters of the backward layer
     * @return Structure that contains parameters of the backward layer
     */
    @Override
    public Parameter getLayerParameter() {
        return null;
    }

    /**
     * Returns the newly allocated backward loss layer
     * with a copy of input objects of this backward loss layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated backward loss layer
     */
    @Override
    public BackwardBatch clone(DaalContext context) {
        return new BackwardBatch(context, this);
    }

    private native long cGetInput(long cAlgorithm);
    private native long cGetResult(long cAlgorithm);
}
