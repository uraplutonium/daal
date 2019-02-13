/* file: ResultId.java */
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
 * @ingroup svd
 * @{
 */
package com.intel.daal.algorithms.svd;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__RESULTID"></a>
 * @brief Available types of results of the SVD algorithm
 */
public final class ResultId {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the result object identifier using the provided value
     * @param value     Value corresponding to the result object identifier
     */
    public ResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the result object identifier
     * @return Value corresponding to the result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int singularValuesId      = 0;
    private static final int leftSingularMatrixId  = 1;
    private static final int rightSingularMatrixId = 2;

    public static final ResultId singularValues      = new ResultId(singularValuesId); /*!< Singular values         */
    public static final ResultId leftSingularMatrix  = new ResultId(
            leftSingularMatrixId);                                                     /*!< Left orthogonal matrix  */
    public static final ResultId rightSingularMatrix = new ResultId(
            rightSingularMatrixId);                                                    /*!< Right orthogonal matrix */
}
/** @} */
