/* file: InitMethod.java */
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
 * @ingroup em_gmm_init
 * @{
 */
package com.intel.daal.algorithms.em_gmm.init;

import com.intel.daal.utils.*;
/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__INIT__INITMETHOD"></a>
 * @brief Available methods for computing initial values for the EM for GMM algorithm
 */
public final class InitMethod {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    private int _value;

    /**
     * Constructs the initialization method object using the provided value
     * @param value     Value corresponding to the initialization method object
     */
    public InitMethod(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the initialization method object
     * @return Value corresponding to the initialization method object
     */
    public int getValue() {
        return _value;
    }

    private static final int DefaultDenseValue = 0;

    public static final InitMethod defaultDense = new InitMethod(DefaultDenseValue); /*!< Default method */
}
/** @} */
