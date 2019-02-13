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
 * @ingroup decision_forest
 */
/**
 * @brief Contains classes of the decision forest regression algorithm
 */
package com.intel.daal.algorithms.decision_forest.regression.training;
import com.intel.daal.algorithms.decision_forest.VariableImportanceModeId;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__REGRESSION__TRAINING__PARAMETER"></a>
 * @brief Base class for parameters of the decision forest regression training algorithm
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {

    public Parameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Returns number of trees to be created by the decision forest regression training algorithm
     * @return Number of trees
     */
    public long getNTrees() {
        return cGetNTrees(this.cObject);
    }

    /**
     * Sets the number of trees to be created by the decision forest regression training algorithm
     * @param nTrees Number of trees
     */
    public void setNTrees(long nTrees) {
        cSetNTrees(this.cObject, nTrees);
    }

    /**
     * Returns fraction of observations used for a training of one tree, 0 to 1.
     * Default is 1 (sampling with replacement)
     * @return Fraction of observations
     */
    public double getObservationsPerTreeFraction() {
        return cGetObservationsPerTreeFraction(this.cObject);
    }

    /**
     * Sets fraction of observations used for a training of one tree, 0 to 1
     * @param value Fraction of observations
     */
    public void setObservationsPerTreeFraction(double value) {
        cSetObservationsPerTreeFraction(this.cObject, value);
    }

    /**
     * Returns number of features tried as possible splits by the decision forest regression training algorithm
     * If 0 then p/3 is used, where p is the total number of features.
     * @return Number of features
     */
    public long getFeaturesPerNode() {
        return cGetFeaturesPerNode(this.cObject);
    }

    /**
     * Sets the number of features tried as possible splits by decision forest regression training algorithm.
     * If 0 then p/3 is used, where p is the total number of features.
     * @param value Number of features
     */
    public void setFeaturesPerNode(long value) {
        cSetFeaturesPerNode(this.cObject, value);
    }

    /**
     * Returns maximal tree depth. Default is 0 (unlimited)
     * @return Maximal tree depth
     */
    public long getMaxTreeDepth() {
        return cGetMaxTreeDepth(this.cObject);
    }

    /**
     * Sets maximal tree depth. Default is 0 (unlimited)
     * @param value Maximal tree depth
     */
    public void setMaxTreeDepth(long value) {
        cSetMaxTreeDepth(this.cObject, value);
    }

    /**
     * Returns minimal number of samples per node. Default is 5
     * @return Minimal number of samples
     */
    public long getMinObservationsInLeafNode() {
        return cGetMinObservationsInLeafNode(this.cObject);
    }

    /**
     * Sets minimal number of samples per node. Default is 5
     * @param value Minimal number of samples
     */
    public void setMinObservationsInLeafNode(long value) {
        cSetMinObservationsInLeafNode(this.cObject, value);
    }

    /**
     * @DAAL_DEPRECATED
     * Returns the seed for the random numbers generator used by the algorithm
     * @return Seed for the seed for the random numbers generator used by the algorithm
     */
    public int getSeed() {
        return cGetSeed(this.cObject);
    }

    /**
     * @DAAL_DEPRECATED
     * Sets the seed for the random numbers generator used by the algorithm
     * @param seed   Seed for the random numbers generator used by the algorithm
     */
    public void setSeed(int seed) {
        cSetSeed(this.cObject, seed);
    }

    /**
     * Sets the engine to be used by the algorithm
     * @param engine to be used by the algorithm
     */
    public void setEngine(com.intel.daal.algorithms.engines.BatchBase engine) {
        cSetEngine(cObject, engine.cObject);
    }

    /**
     * Returns threshold value used as stopping criteria:
     * if the impurity value in the node is smaller than the threshold then the node is not split anymore
     * @return Impurity threshold
     */
    public double getImpurityThreshold() {
        return cGetImpurityThreshold(this.cObject);
    }

    /**
     * Sets threshold value used as stopping criteria
     * @param value Impurity threshold
     */
    public void setImpurityThreshold(double value) {
        cSetImpurityThreshold(this.cObject, value);
    }

    /**
     * Returns variable importance computation mode
     * @return Variable importance computation mode
     */
    public VariableImportanceModeId getVariableImportanceMode() {
        return new VariableImportanceModeId(cGetVariableImportanceMode(this.cObject));
    }

    /**
     * Sets the variable importance computation mode
     * @param value Variable importance computation mode
     */
    public void setVariableImportanceMode(VariableImportanceModeId value) {
        cSetVariableImportanceMode(this.cObject, value.getValue());
    }

    /**
     * Sets the 64 bit integer flag that indicates the results to compute
     * @param resultsToCompute The 64 bit integer flag that indicates the results to compute
     */
    public void setResultsToCompute(long resultsToCompute) {
        cSetResultsToCompute(this.cObject, resultsToCompute);
    }

    /**
     * Gets the 64 bit integer flag that indicates the results to compute
     * @return The 64 bit integer flag that indicates the results to compute
     */
    public long getResultsToCompute() {
        return cGetResultsToCompute(this.cObject);
    }

    private native long cGetNTrees(long parAddr);
    private native void cSetNTrees(long parAddr, long value);

    private native void   cSetObservationsPerTreeFraction(long parAddr, double value);
    private native double cGetObservationsPerTreeFraction(long parAddr);

    private native long cGetFeaturesPerNode(long parAddr);
    private native void cSetFeaturesPerNode(long parAddr, long value);

    private native long cGetMaxTreeDepth(long parAddr);
    private native void cSetMaxTreeDepth(long parAddr, long value);

    private native long cGetMinObservationsInLeafNode(long parAddr);
    private native void cSetMinObservationsInLeafNode(long parAddr, long value);

    private native int cGetSeed(long parAddr);
    private native void cSetSeed(long parAddr, int value);
    private native void cSetEngine(long cObject, long cEngineObject);

    private native void   cSetImpurityThreshold(long parAddr, double value);
    private native double cGetImpurityThreshold(long parAddr);

    private native void cSetResultsToCompute(long parAddr, long value);
    private native long cGetResultsToCompute(long parAddr);

    private native int cGetVariableImportanceMode(long parAddr);
    private native void cSetVariableImportanceMode(long parAddr, int value);

}
/** @} */
