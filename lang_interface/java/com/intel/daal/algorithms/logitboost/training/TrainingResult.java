/* file: TrainingResult.java */
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
 * @ingroup logitboost_training
 * @{
 */
package com.intel.daal.algorithms.logitboost.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.algorithms.logitboost.Model;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGITBOOST__TRAINING__TRAININGRESULT"></a>
 * @brief Provides methods to access final results obtained with the compute() method of the LogitBoost training algorithm
 *                                                                                                in the batch processing mode.
 */
public final class TrainingResult extends com.intel.daal.algorithms.boosting.training.TrainingResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public TrainingResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the final result of the LogitBoost training algorithms
     * @param id   Identifier of the result
     * @return         %Result that corresponds to the given identifier
     */
    @Override
    public Model get(TrainingResultId id) {
        if (id != TrainingResultId.model) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new Model(getContext(), cGetModel(cObject, TrainingResultId.model.getValue()));
    }
}
/** @} */
