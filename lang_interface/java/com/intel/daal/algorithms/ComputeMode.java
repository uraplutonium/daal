/* file: ComputeMode.java */
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
 * @ingroup base_algorithms
 * @{
 */
package com.intel.daal.algorithms;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COMPUTEMODE"></a>
 * Computation modes of the Intel(R) Data Analytics Acceleration Library algorithms
 */
public final class ComputeMode {
    private int _value;

    /**
     * Constructs the compute mode object using the provided value
     * @param value     Value corresponding to the compute mode object
     */
    public ComputeMode(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the compute mode object
     * @return Value corresponding to the compute mode object
     */
    public int getValue() {
        return _value;
    }

    private static final int batchValue       = 0;
    private static final int onlineValue      = 1;
    private static final int distributedValue = 2;

    /** Batch processing computation mode */
    public static final ComputeMode batch       = new ComputeMode(batchValue);
    /** Online mode - processing of data sets in blocks */
    public static final ComputeMode online      = new ComputeMode(onlineValue);
    /** Processing of data sets distributed across several devices */
    public static final ComputeMode distributed = new ComputeMode(distributedValue);
}
/** @} */
