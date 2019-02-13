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
 * @ingroup kmeans_compute
 * @{
 */
package com.intel.daal.algorithms.kmeans;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__RESULT"></a>
 * @brief Results obtained with the compute() method of the K-Means algorithm in the batch processing mode
 */
public final class Result extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Default constructor. Constructs an empty Result object.
     * @param context   Context to manage the result of the K-Means algorithm
     */
    public Result(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public Result(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Returns the result of the K-Means algorithm
     * @param id   Identifier of the result, @ref ResultId
     * @return Result that corresponds to the given identifier
     */
    public NumericTable get(ResultId id) {
        if (id != ResultId.centroids && id != ResultId.assignments && id != ResultId.objectiveFunction
                && id != ResultId.nIterations) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetResultTable(cObject, id.getValue()));
    }

    /**
     * Sets the result of the K-Means algorithm
     * @param id    Identifier of the result
     * @param value Value of the result
     */
    public void set(ResultId id, NumericTable value) {
        if (id != ResultId.centroids && id != ResultId.assignments && id != ResultId.objectiveFunction
                && id != ResultId.nIterations) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetResultTable(cObject, id.getValue(), value.getCObject());
    }

    private native long cNewResult();

    private native long cGetResultTable(long cObject, int id);

    private native void cSetResultTable(long cObject, int id, long cNumericTable);
}
/** @} */
