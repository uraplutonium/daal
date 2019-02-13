/* file: InitInputId.java */
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

/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__INIT__INITINPUTID"></a>
 * @brief Available identifiers of input objects for the default initialization of the EM for GMM algorithm
 */
public final class InitInputId {
    private int _value;

    /**
     * Constructs the default initialization input object identifier using the provided value
     * @param value     Value corresponding to the default initialization input object identifier
     */
    public InitInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the default initialization input object identifier
     * @return Value corresponding to the default initialization input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int Data = 0;

    public static final InitInputId data = new InitInputId(Data); /*!< %Input data table */
}
/** @} */
