/* file: MultiClassConfusionMatrixResultId.java */
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
 * @ingroup quality_metric_multiclass
 * @{
 */
package com.intel.daal.algorithms.classifier.quality_metric.multi_class_confusion_matrix;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__QUALITY_METRIC__MULTI_CLASS_CONFUSION_MATRIX__MULTICLASSCONFUSIONMATRIXRESULTID"></a>
 * @brief Available identifiers of the results of the confusion matrix algorithm
 */
public final class MultiClassConfusionMatrixResultId {
    private int _value;

    /**
     * Constructs the confusion matrix result object identifier using the provided value
     * @param value     Value corresponding to the confusion matrix result object identifier
     */
    public MultiClassConfusionMatrixResultId(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the confusion matrix result object identifier
     * @return Value corresponding to the confusion matrix result object identifier
     */
    public int getValue() {
        return _value;
    }

    private static final int ConfusionMatrix   = 0;
    private static final int MultiClassMetrics = 1;

    /*!< Labels computed in the prediction stage of the classification algorithm */
    public static final MultiClassConfusionMatrixResultId confusionMatrix   = new MultiClassConfusionMatrixResultId(
            ConfusionMatrix);
    /*!< Table that contains quality metrics (i.e., precision, recall, etc.) for multi-class classifiers */
    public static final MultiClassConfusionMatrixResultId multiClassMetrics = new MultiClassConfusionMatrixResultId(
            MultiClassMetrics);
}
/** @} */
