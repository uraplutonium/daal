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
 * @ingroup sum_of_functions
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.sum_of_functions;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SUM_OF_FUNCTIONS__PARAMETER"></a>
 * @brief Parameters of the Sum of functions algorithm
 */
public class Parameter extends com.intel.daal.algorithms.optimization_solver.objective_function.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    long cCreatedParameter;

    /**
     * Constructs the parameter for the sum of functions algorithm
     * @param context       Context to manage the sum of functions algorithm
     * @param cParameter    Pointer to C++ implementation of the parameter
     */
    public Parameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Constructs the parameter for the sum of functions algorithm
     * @param numberOfTerms The number of terms in the function that can be represented as sum
     * @param context       Context to manage the sum of functions algorithm
     */
    public Parameter(long numberOfTerms, DaalContext context) {
        super(context);
        this.cCreatedParameter = cCreateParameter(numberOfTerms);
        this.cObject = this.cCreatedParameter;
    }

    /**
     * Sets the numeric table of size 1 x m where m is batch size that represent
     * a batch of indices used to compute the function results, e.g.,
     * value of the sum of the functions. If no indices are provided,
     * all terms will be used in the computations.
     * @param batchIndices The numeric table of size 1 x m where m is batch size that represent
     * a batch of indices used to compute the function results, e.g.,
     * value of the sum of the functions. If no indices are provided,
     * all terms will be used in the computations.
     */
    public void setBatchIndices(NumericTable batchIndices) {
        cSetBatchIndices(this.cObject, batchIndices.getCObject());
    }

    /**
     * Gets the numeric table of size 1 x m where m is batch size that represent
     * a batch of indices used to compute the function results, e.g.,
     * value of the sum of the functions. If no indices are provided,
     * all terms will be used in the computations.
     * @return The numeric table of size 1 x m where m is batch size that represent
     * a batch of indices used to compute the function results, e.g.,
     * value of the sum of the functions. If no indices are provided,
     * all terms will be used in the computations.
     */
    public NumericTable getBatchIndices() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetBatchIndices(this.cObject));
    }

    /**
     * Sets the number of terms in the function
     * @param numberOfTerms The number of terms in the function
     */
    public void setNumberOfTerms(long numberOfTerms) {
        cSetNumberOfTerms(this.cObject, numberOfTerms);
    }

    /**
     * Gets the number of terms in the function
     * @return The number of terms in the function
     */
    public long getNumberOfTerms() {
        return cGetNumberOfTerms(this.cObject);
    }

    /**
     * Sets parameter pointer for algorithm in native side
     * @param cParameter The address of the native parameter object
     * @param cAlgorithm The address of the native algorithm object
     */
    public void setCParameter(long cParameter, long cAlgorithm) {
        this.cObject = cParameter;
        cSetCParameter(this.cObject, cAlgorithm);
    }

    /**
    * Releases memory allocated for the native parameter object
    */
    @Override
    public void dispose() {
        if(this.cCreatedParameter != 0) {
            cParameterDispose(this.cCreatedParameter);
            this.cCreatedParameter = 0;
        }
    }

    private native void cSetBatchIndices(long parAddr, long batchIndicesAddr);
    private native long cGetBatchIndices(long parAddr);
    private native void cSetNumberOfTerms(long parAddr, long numberOfTerms);
    private native long cGetNumberOfTerms(long parAddr);
    private native long cCreateParameter(long numberOfTerms);
    private native void cParameterDispose(long cCreatedParameter);
    private native void cSetCParameter(long cParameter, long cAlgorithm);
}
/** @} */
