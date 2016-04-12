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

/**
 * @brief Contains classes for computing Stochastic gradient descent algorithm
 */
package com.intel.daal.algorithms.optimization_solver.sgd;

import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.algorithms.optimization_solver.sum_of_functions.Batch;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__BASEPARAMETER"></a>
 * @brief Base parameter of the Sum of functions algorithm
 */
public class BaseParameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the parameter for SGD algorithm
     * @param context       Context to manage the SGD algorithm
     */
    public BaseParameter(DaalContext context) {
        super(context);
    }

    /**
     * Constructs the parameter for SGD algorithm
     * @param context    Context to manage the SGD algorithm
     * @param cObject    Pointer to C++ implementation of the parameter
     */
    public BaseParameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
    * Sets objective function represented as sum of functions
    * @param function Objective function represented as sum of functions
    */
    public void setFunction(Batch function) {
        _function = function;
        cSetFunction(this.cObject, function.cBatchIface);
    }

    /**
     * Gets objective function represented as sum of functions
     * @return Objective function represented as sum of functions
     */
    public Batch getFunction() {
        return _function;
    }

    /**
    * Sets the maximal number of iterations of the algorithm
    * @param nIterations The maximal number of iterations of the algorithm
    */
    public void setNIterations(long nIterations) {
        cSetNIterations(this.cObject, nIterations);
    }

    /**
     * Gets the maximal number of iterations of the algorithm
     * @return The maximal number of iterations of the algorithm
     */
    public long getNIterations() {
        return cGetNIterations(this.cObject);
    }

    /**
    * Sets the accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
    * @param accuracyThreshold The accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
    */
    public void setAccuracyThreshold(double accuracyThreshold) {
        cSetAccuracyThreshold(this.cObject, accuracyThreshold);
    }

    /**
     * Gets the accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
     * @return The accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
     */
    public double getAccuracyThreshold() {
        return cGetAccuracyThreshold(this.cObject);
    }

    /**
     * The numeric table that represents 32 bit integer indices of terms in the objective function. If no indices are provided,
     * the implementation will generate random indices.
     * @param batchIndices The numeric table that represents 32 bit integer indices of terms in the objective function. If no indices are provided,
     * the implementation will generate random indices.
     */
    public void setBatchIndices(NumericTable batchIndices) {
        cSetBatchIndices(this.cObject, batchIndices.getCObject());
    }

    /**
     * Gets the numeric table that represents 32 bit integer indices of terms in the objective function. If no indices are provided,
     * the implementation will generate random indices.
     * @return The numeric table that represents 32 bit integer indices of terms in the objective function. If no indices are provided,
     * the implementation will generate random indices.
     */
    public NumericTable getBatchIndices() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetBatchIndices(this.cObject));
    }

    /**
     * Sets the numeric table that contains values of the learning rate sequence
     * @param learningRateSequence The numeric table that contains values of the learning rate sequence
     */
    public void setLearningRateSequence(NumericTable learningRateSequence) {
        cSetLearningRateSequence(this.cObject, learningRateSequence.getCObject());
    }

    /**
     * Gets the numeric table that contains values of the learning rate sequence
     * @return The numeric table that contains values of the learning rate sequence
     */
    public NumericTable getLearningRateSequence() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetLearningRateSequence(this.cObject));
    }

    /**
    * Sets the seed for random generation of 32 bit integer indices of terms in the objective function.
    * @param seed The seed for random generation of 32 bit integer indices of terms in the objective function.
    */
    public void setSeed(int seed) {
        cSetSeed(this.cObject, seed);
    }

    /**
     * Gets the seed for random generation of 32 bit integer indices of terms in the objective function.
     * @return The seed for random generation of 32 bit integer indices of terms in the objective function.
     */
    public int getSeed() {
        return cGetSeed(this.cObject);
    }

    private Batch _function;

    private native void cSetFunction(long parAddr, long function);

    private native void cSetNIterations(long parAddr, long nIterations);
    private native long cGetNIterations(long parAddr);

    private native void cSetAccuracyThreshold(long parAddr, double accuracyThreshold);
    private native double cGetAccuracyThreshold(long parAddr);

    private native void cSetBatchIndices(long parAddr, long batchIndices);
    private native long cGetBatchIndices(long parAddr);

    private native void cSetLearningRateSequence(long parAddr, long learningRateSequence);
    private native long cGetLearningRateSequence(long parAddr);

    private native void cSetSeed(long parAddr, int seed);
    private native int cGetSeed(long parAddr);

}
