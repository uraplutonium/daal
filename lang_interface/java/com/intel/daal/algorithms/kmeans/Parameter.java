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
 * @ingroup kmeans_compute
 * @{
 */
package com.intel.daal.algorithms.kmeans;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__PARAMETER"></a>
 * @brief Parameters of the K-Means computation method
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    DistanceType distanceType;

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /** @private */
    public Parameter(DaalContext context, long cParameter) {
        super(context);
        this.cObject = cParameter;
    }

    /**
     * Constructs a parameter
     * @param context               Context to manage the parameter of the K-Means algorithm
     * @param nClusters             Number of clusters
     * @param maxIterations         Number of iterations
     * @param accuracyThreshold     Threshold for the termination of the algorithm
     * @param gamma                 Weight used in distance calculation for categorical features
     * @param distanceType          Distance used in the algorithm
     * @param assignFlag            Flag to enable assignment of observations to clusters; assigns data points
     */
    public Parameter(DaalContext context, long nClusters, long maxIterations, double accuracyThreshold, double gamma,
            DistanceType distanceType, boolean assignFlag) {
        super(context);
        this.distanceType = distanceType;

        initialize(nClusters, maxIterations, accuracyThreshold, gamma, assignFlag);
    }

    /**
     * Constructs a parameter
     * @param context               Context to manage the parameter of the K-Means algorithm
     * @param nClusters             Number of clusters
     * @param maxIterations         Number of iterations
     * @param accuracyThreshold     Threshold for the termination of the algorithm
     * @param gamma                 Weight used in distance calculation for categorical features
     * @param distanceType          Distance used in the algorithm
     */
    public Parameter(DaalContext context, long nClusters, long maxIterations, double accuracyThreshold, double gamma,
            DistanceType distanceType) {
        super(context);
        this.distanceType = distanceType;

        boolean assignFlag = true;
        initialize(nClusters, maxIterations, accuracyThreshold, gamma, assignFlag);
    }

    /**
    * Constructs a parameter
    * @param context               Context to manage the parameter of the K-Means algorithm
    * @param nClusters             Number of clusters
    * @param maxIterations         Number of iterations
    * @param accuracyThreshold     Threshold for the termination of the algorithm
    * @param gamma                 Weight used in distance calculation for categorical features
    */
    public Parameter(DaalContext context, long nClusters, long maxIterations, double accuracyThreshold, double gamma) {
        super(context);
        boolean assignFlag = true;
        this.distanceType = DistanceType.euclidean;
        initialize(nClusters, maxIterations, accuracyThreshold, gamma, assignFlag);
    }

    /**
    * Constructs a parameter
    * @param context               Context to manage the parameter of the K-Means algorithm
    * @param nClusters             Number of clusters
    * @param maxIterations         Number of iterations
    * @param accuracyThreshold     Threshold for the termination of the algorithm
    */
    public Parameter(DaalContext context, long nClusters, long maxIterations, double accuracyThreshold) {
        super(context);

        boolean assignFlag = true;
        this.distanceType = DistanceType.euclidean;
        double gamma = 1.0;
        initialize(nClusters, maxIterations, accuracyThreshold, gamma, assignFlag);
    }

    /**
    * Constructs a parameter
    * @param context               Context to manage the parameter of the K-Means algorithm
    * @param nClusters             Number of clusters
    * @param maxIterations         Number of iterations
    */
    public Parameter(DaalContext context, long nClusters, long maxIterations) {
        super(context);
        DistanceType distanceType = DistanceType.euclidean;
        this.distanceType = distanceType;

        boolean assignFlag = true;
        double gamma = 1.0;
        double accuracyThreshold = 0.0;
        initialize(nClusters, maxIterations, accuracyThreshold, gamma, assignFlag);
    }

    private void initialize(long nClusters, long maxIterations, double accuracyThreshold, double gamma,
            boolean assignFlag) {
        if (this.distanceType == DistanceType.euclidean) {
            this.cObject = initEuclidean(nClusters, maxIterations);
        } else {
            throw new IllegalArgumentException("distanceType unsupported");
        }

        setAccuracyThreshold(accuracyThreshold);
        setGamma(gamma);
        setAssignFlag(assignFlag);
    }

    /**
     * Returns the distance type
     * @return Distance type
     */
    public DistanceType getDistanceType() {
        return distanceType;
    }

    /**
     * Retrieves the number of clusters
     * @return Number of clusters
     */
    public long getNClusters() {
        return cGetNClusters(this.cObject);
    }

    /**
     * Retrieves the number of iterations
     * @return Number of iterations
     */
    public long getMaxIterations() {
        return cGetMaxIterations(this.cObject);
    }

    /**
     * Retrieves the threshold for the termination of the algorithm
     * @return Threshold for the termination of the algorithm
     */
    public double getAccuracyThreshold() {
        return cGetAccuracyThreshold(this.cObject);
    }

    /**
     * Retrieves the weight used in distance calculation for categorical features
     * @return Weight used in distance calculation for categorical features
     */
    public double getGamma() {
        return cGetGamma(this.cObject);
    }

    /**
     * Retrieves the flag for the assignment of data points
     * @return Flag for the assignment of data points
     */
    public boolean getAssignFlag() {
        return cGetAssignFlag(this.cObject);
    }

    /**
    * Sets the number of clusters
    * @param nClusters Number of clusters
    */
    public void setNClusters(long nClusters) {
        cSetNClusters(this.cObject, nClusters);
    }

    /**
     * Sets the number of iterations
     * @param max Number of iterations.
     */
    public void setMaxIterations(long max) {
        cSetMaxIterations(this.cObject, max);
    }

    /**
     * Sets the threshold for the termination of the algorithm
     * @param accuracy Threshold for the termination of the algorithm
     */
    public void setAccuracyThreshold(double accuracy) {
        cSetAccuracyThreshold(this.cObject, accuracy);
    }

    /**
     * Sets the weight used in distance calculation for categorical features
     * @param gamma Weight used in distance calculation for categorical features
     */
    public void setGamma(double gamma) {
        cSetGamma(this.cObject, gamma);
    }

    /**
     * Sets the flag for the assignment of data points
     * @param assignFlag Flag to enable assignment of observations to clusters
     */
    public void setAssignFlag(boolean assignFlag) {
        cSetAssignFlag(this.cObject, assignFlag);
    }

    private native long initEuclidean(long nClusters, long maxIterations);

    private native long cGetNClusters(long parameterAddress);

    private native long cGetMaxIterations(long parameterAddress);

    private native double cGetAccuracyThreshold(long parameterAddress);

    private native double cGetGamma(long parameterAddress);

    private native boolean cGetAssignFlag(long parameterAddress);

    private native void cSetNClusters(long parameterAddress, long nClusters);

    private native void cSetMaxIterations(long parameterAddress, long maxIterations);

    private native void cSetAccuracyThreshold(long parameterAddress, double accuracyThreshold);

    private native void cSetGamma(long parameterAddress, double gamma);

    private native void cSetAssignFlag(long parameterAddress, boolean assignFlag);
}
/** @} */
