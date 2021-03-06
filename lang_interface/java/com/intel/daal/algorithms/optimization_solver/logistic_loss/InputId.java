/* file: InputId.java */
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
 * @ingroup logistic_loss
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.logistic_loss;

import com.intel.daal.utils.*;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__LOGISTIC_LOSS__INPUTID"></a>
 * @brief Available identifiers of input objects for the logistic loss objective function algorithm
 */
public final class InputId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the input object identifier using the provided value
     * @param value     Value corresponding to the input object identifier
     */
    public InputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the input object identifier
     * @return Value corresponding to the input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int argumentId = 0;
    private static final int dataId = 1;
    private static final int dependentVariablesId = 2;

    public static final InputId argument           = new InputId(argumentId);           /*!< %Input argument table */
    public static final InputId data               = new InputId(dataId);               /*!< %Input data table */
    public static final InputId dependentVariables = new InputId(dependentVariablesId); /*!< %Input dependent variables table */
}
/** @} */
