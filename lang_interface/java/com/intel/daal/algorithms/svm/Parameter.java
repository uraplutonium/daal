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
 * @brief Contains classes of the support vector machine (SVM) classification algorithm
 */
package com.intel.daal.algorithms.svm;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__PARAMETER"></a>
 * @brief Optional SVM algorithm parameters
 *
 * @ref opt_notice
 *
 */
public class Parameter extends com.intel.daal.algorithms.classifier.Parameter {

    public Parameter(DaalContext context, long cParameter) {
        super(context);
        this.cObject = cParameter;
    }

    /**
     * Sets an upper bound in constraints of the quadratic optimization problem
     * @param C     Upper bound in constraints of the quadratic optimization problem
     */
    public void setC(double C) {
        cSetC(this.cObject, C);
    }

    /**
     * Retrieves an upper bound in constraints of the quadratic optimization problem
     * @return Upper bound in constraints of the quadratic optimization problem
     */
    public double getC() {
        return cGetC(this.cObject);
    }

    /**
     * Sets the accuracy of the SVM training algorithm
     * @param Eps     Accuracy of the SVM training algorithm
     */
    public void setEps(double Eps) {
        cSetEps(this.cObject, Eps);
    }

    /**
     * Retrieves the accuracy of the SVM training algorithm
     * @return Accuracy of the SVM training algorithm
     */
    public double getEps() {
        return cGetEps(this.cObject);
    }

    /**
     * Sets the tau parameter of the working set selection scheme
     * @param Tau     Parameter of the working set selection scheme
     */
    public void setTau(double Tau) {
        cSetTau(this.cObject, Tau);
    }

    /**
     * Retrieves the tau parameter of the working set selection scheme
     * @return Parameter of the working set selection scheme
     */
    public double getTau() {
        return cGetTau(this.cObject);
    }

    /**
     * Sets the maximal number of iterations of the SVM training algorithm
     * @param Iter Maximal number of iterations of the SVM training algorithm
     */
    public void setIter(long Iter) {
        cSetIter(this.cObject, Iter);
    }

    /**
     * Retrieves the maximal number of iterations of the SVM training algorithm
     * @return Maximal number of iterations of the SVM training algorithm
     */
    public long getIter() {
        return cGetIter(this.cObject);
    }

    /**
     * Sets the size of the cache in bytes to store values of the kernel matrix.
     * A non-zero value enables use of a cache optimization technique
     * @param CacheSize Size of the cache in bytes
     */
    public void setCacheSize(long CacheSize) {
        cSetCacheSize(this.cObject, CacheSize);
    }

    /**
     * Retrieves the size of the cache in bytes to store values of the kernel matrix.
     * @return Size of the cache in bytes
     */
    public long getCacheSize() {
        return cGetCacheSize(this.cObject);
    }

    /**
     * Sets the flag that enables use of the shrinking optimization technique
     * @param DoShrinking   Flag that enables use of the shrinking optimization technique
     */
    public void setDoShrinking(boolean DoShrinking) {
        cSetDoShrinking(this.cObject, DoShrinking);
    }

    /**
     * Retrieves the flag that enables use of the shrinking optimization technique
     * @return   Flag that enables use of the shrinking optimization technique
     */
    public boolean getDoShrinking() {
        return cGetDoShrinking(this.cObject);
    }

    /**
     * Sets the kernel function
     * @param Kernel    Kernel function
     */
    public void setKernel(com.intel.daal.algorithms.kernel_function.Batch Kernel) {
        cSetKernel(this.cObject, Kernel.cObject);
    }

    private native void cSetC(long parAddr, double C);

    private native double cGetC(long parAddr);

    private native void cSetEps(long parAddr, double Eps);

    private native double cGetEps(long parAddr);

    private native void cSetTau(long parAddr, double Tau);

    private native double cGetTau(long parAddr);

    private native void cSetIter(long parAddr, long Iter);

    private native long cGetIter(long parAddr);

    private native void cSetCacheSize(long parAddr, long CacheSize);

    private native long cGetCacheSize(long parAddr);

    private native void cSetDoShrinking(long parAddr, boolean DoShrinking);

    private native boolean cGetDoShrinking(long parAddr);

    private native void cSetKernel(long parAddr, long kernelAddr);
}
