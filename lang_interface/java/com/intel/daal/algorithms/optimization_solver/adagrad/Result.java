/* file: Result.java */
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
 * @ingroup adagrad
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.adagrad;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.OptionalArgument;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__RESULT"></a>
 * @brief Provides methods to access the results obtained with the compute() method of the
 *        Adagrad algorithm in the batch processing mode
 */
public class Result extends com.intel.daal.algorithms.optimization_solver.iterative_solver.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the result for the Adagrad algorithm
     * @param context Context to manage the result for the Adagrad algorithm
     */
    public Result(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    /**
    * Constructs the result for the Adagrad algorithm
    * @param context       Context to manage the Adagrad algorithm result
    * @param cResult       Pointer to C++ implementation of the result
    */
    public Result(DaalContext context, long cResult) {
        super(context, cResult);
    }

    /**
     * Returns the optional data of the Adagrad algorithm in the batch processing mode
     * @param id   Identifier of the optional data, @ref OptionalDataId
     * @return Optional data that corresponds to the given identifier
     */
    public NumericTable get(OptionalDataId id) {
        if (id != OptionalDataId.gradientSquareSum) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetOptionalData(cObject, id.getValue()));
    }

    /**
     * Sets the optional data of the Adagrad algorithm in the batch processing mode
     * @param id   Identifier of the optional data, @ref OptionalDataId
     * @param val Object to store the data
     */
    public void set(OptionalDataId id, NumericTable val) {
        if (id != OptionalDataId.gradientSquareSum) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetOptionalData(cObject, id.getValue(), val.getCObject());
    }

    private native long cNewResult();
    protected native long cGetOptionalData(long cObject, int id);
    protected native void cSetOptionalData(long cResult, int id, long cOptionalData);
}
/** @} */
