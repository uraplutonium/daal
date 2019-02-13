/* file: PredictionMethod.java */
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
 * @ingroup multi_class_classifier_prediction
 * @{
 */
package com.intel.daal.algorithms.multi_class_classifier.prediction;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__PREDICTION__PREDICTIONMETHOD"></a>
 * @brief Available methods for the multi-class classifier prediction
 */
public final class PredictionMethod {

    private int _value;

    /**
     * Constructs the prediction method object using the provided value
     * @param value     Value corresponding to the prediction method object
     */
    public PredictionMethod(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the prediction method object
     * @return Value corresponding to the prediction method object
     */
    public int getValue() {
        return _value;
    }

    private static final int multiClassClassifierWuValue = 0;
    private static final int voteBasedValue = 1;

    public static final PredictionMethod multiClassClassifierWu = new PredictionMethod(
            multiClassClassifierWuValue); /*!< Prediction method for the Multi-class classifier proposed by Ting-Fan Wu et al */

    public static final PredictionMethod voteBased = new PredictionMethod(
            voteBasedValue); /*!< Prediction method that is based on votes returned by two-class classifiers */
}
/** @} */
