/* file: PredictionInput.java */
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
 * @defgroup decision_tree_classification_prediction Prediction
 * @brief Contains classes for prediction based on decision tree classification models
 * @ingroup decision_tree_classification
 * @{
 */
package com.intel.daal.algorithms.decision_tree.classification.prediction;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.decision_tree.classification.Model;
import com.intel.daal.algorithms.classifier.prediction.NumericTableInputId;
import com.intel.daal.algorithms.classifier.prediction.ModelInputId;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__CLASSIFICATION__PREDICTION__PREDICTIONINPUT"></a>
 * @brief  %Input objects for the decision tree classification algorithm
 */
public class PredictionInput extends com.intel.daal.algorithms.classifier.prediction.PredictionInput {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public PredictionInput(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Returns the Model input object for the decision tree classification model-based prediction algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    @Override
    public Model get(ModelInputId id) {
        if (id == ModelInputId.model) {
            return new Model(getContext(), cGetInputModel(cObject, id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }
}
/** @} */
