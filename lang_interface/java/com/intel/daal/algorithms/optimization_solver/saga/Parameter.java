/* file: Parameter.java */
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
 * @ingroup saga
 * @{
 */
/**
 * @brief Contains classes for computing SAGA algorithm
 */
package com.intel.daal.algorithms.optimization_solver.saga;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.algorithms.optimization_solver.sum_of_functions.Batch;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SAGA__PARAMETER"></a>
 * @brief Parameter of the SAGA algorithm
 */
public class Parameter extends com.intel.daal.algorithms.optimization_solver.iterative_solver.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the parameter for the SAGA algorithm
     * @param context       Context to manage the parameter for the SAGA algorithm
     */
    public Parameter(DaalContext context) {
        super(context);
    }

    /**
     * Constructs the parameter for the SAGA algorithm
     * @param context    Context to manage the SAGA algorithm
     * @param cObject    Pointer to C++ implementation of the parameter
     */
    public Parameter(DaalContext context, long cObject) {
        super(context, cObject);
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
     * Sets the numeric table that contains value of the learning rate
     * @param learningRate The numeric table that contains value of the learning rate
     */
    public void setLearningRateSequence(NumericTable learningRate) {
        cSetLearningRateSequence(this.cObject, learningRate.getCObject());
    }

    /**
     * Gets the numeric table that contains value of the learning rate
     * @return The numeric table that contains value of the learning rate
     */
    public NumericTable getLearningRateSequence() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetLearningRateSequence(this.cObject));
    }

    /**
    * @DAAL_DEPRECATED
    * Sets the seed for random generation of 32 bit integer indices of terms in the objective function.
    * @param seed The seed for random generation of 32 bit integer indices of terms in the objective function.
    */
    public void setSeed(int seed) {
        cSetSeed(this.cObject, seed);
    }

    /**
     * @DAAL_DEPRECATED
     * Gets the seed for random generation of 32 bit integer indices of terms in the objective function.
     * @return The seed for random generation of 32 bit integer indices of terms in the objective function.
     */
    public int getSeed() {
        return cGetSeed(this.cObject);
    }

    /**
     * Sets the engine to be used by the algorithm
     * @param engine to be used by the algorithm
     */
    public void setEngine(com.intel.daal.algorithms.engines.BatchBase engine) {
        cSetEngine(cObject, engine.cObject);
    }

    private native void cSetBatchIndices(long parAddr, long batchIndicesAddr);
    private native long cGetBatchIndices(long parAddr);

    private native void cSetLearningRateSequence(long parAddr, long learningRateAddr);
    private native long cGetLearningRateSequence(long parAddr);

    private native void cSetSeed(long parAddr, int seed);
    private native int cGetSeed(long parAddr);
    private native void cSetEngine(long cObject, long cEngineObject);
}
/** @} */
