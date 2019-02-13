/* file: ModelInputId.java */
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
 * @defgroup prediction Prediction
 * @brief Contains classes for making prediction based on the classifier model
 * @ingroup classifier
 * @{
 */
/**
 * @brief Contains classes for making prediction based on the classification model
 */
package com.intel.daal.algorithms.classifier.prediction;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__PREDICTION__MODELINPUTID"></a>
 * @brief Available identifiers of input objects of the classification algorithms
 */
public final class ModelInputId {
    private int _value;

    /**
     * Constructs the model input object identifier using the provided value
     * @param value     Value corresponding to the model input object identifier
     */
    public ModelInputId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the model input object identifier
     * @return Value corresponding to the model input object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int Model = 1;

    public static final ModelInputId model = new ModelInputId(Model); /*!< Model to use in the prediction stage*/
}
/** @} */
