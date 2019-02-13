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
 * @ingroup low_order_moments
 * @{
 */
package com.intel.daal.algorithms.low_order_moments;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__RESULT"></a>
 * @brief Provides methods to access final results obtained with the compute() method of the
 *        low order moments algorithm in the batch processing mode;
 *        or finalizeCompute() method of the algorithm in the online or distributed  processing mode
 */
public final class Result extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the result of the low order moments algorithm
     * @param context   Context to manage the result of the low order moments algorithm
     */
    public Result(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public Result(DaalContext context, long cResult) {
        super(context, cResult);
    }

    /**
     * Returns the final result of the low order moments algorithm
     * @param  id   Identifier of the result, @ref ResultId
     * @return Final result that corresponds to the given identifier
     */
    public NumericTable get(ResultId id) {
        int idValue = id.getValue();
        if (idValue != ResultId.minimum.getValue() && idValue != ResultId.maximum.getValue()
                && idValue != ResultId.sum.getValue() && idValue != ResultId.sumSquares.getValue()
                && idValue != ResultId.sumSquaresCentered.getValue() && idValue != ResultId.mean.getValue()
                && idValue != ResultId.secondOrderRawMoment.getValue() && idValue != ResultId.variance.getValue()
                && idValue != ResultId.standardDeviation.getValue() && idValue != ResultId.variation.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetResultTable(cObject, idValue));
    }

    /**
     * Sets the final result of the low order moments algorithm
     * @param id    Identifier of the final result
     * @param value Object to store the final result
     */
    public void set(ResultId id, NumericTable value) {
        int idValue = id.getValue();
        if (idValue != ResultId.minimum.getValue() && idValue != ResultId.maximum.getValue()
                && idValue != ResultId.sum.getValue() && idValue != ResultId.sumSquares.getValue()
                && idValue != ResultId.sumSquaresCentered.getValue() && idValue != ResultId.mean.getValue()
                && idValue != ResultId.secondOrderRawMoment.getValue() && idValue != ResultId.variance.getValue()
                && idValue != ResultId.standardDeviation.getValue() && idValue != ResultId.variation.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetResultTable(cObject, idValue, value.getCObject());
    }

    private native long cNewResult();

    private native long cGetResultTable(long cResult, int id);

    private native void cSetResultTable(long cResult, int id, long cNumericTable);
}
/** @} */
