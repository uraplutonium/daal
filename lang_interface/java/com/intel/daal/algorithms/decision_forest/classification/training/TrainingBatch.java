/* file: TrainingBatch.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation
* All Rights Reserved.
*
* If this  software was obtained  under the  Intel Simplified  Software License,
* the following terms apply:
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*
*
* If this  software  was obtained  under the  Apache License,  Version  2.0 (the
* "License"), the following terms apply:
*
* You may  not use this  file except  in compliance  with  the License.  You may
* obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
*
*
* Unless  required  by   applicable  law  or  agreed  to  in  writing,  software
* distributed under the License  is distributed  on an  "AS IS"  BASIS,  WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*
* See the   License  for the   specific  language   governing   permissions  and
* limitations under the License.
*******************************************************************************/

/**
 * @defgroup decision_forest_classification_training_batch Batch
 * @ingroup decision_forest_classification_training
 * @{
 */
/**
 * @brief Contains classes for training decision forest classification algorithm
 */
package com.intel.daal.algorithms.decision_forest.classification.training;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.classifier.training.TrainingInput;
import com.intel.daal.algorithms.decision_forest.classification.training.Parameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__TRAINING__TRAININGBATCH"></a>
 * @brief Trains a model of the decision forest classification algorithm in the batch processing mode
 * <!-- \n<a href="DAAL-REF-DECISION_FOREST__CLASSIFICATION-ALGORITHM">decision forest classification algorithm description and usage models</a> -->
 *
 * \par References
 *      - com.intel.daal.algorithms.classifier.training.InputId class
 *      - com.intel.daal.algorithms.classifier.training.TrainingResultId class
 *      - com.intel.daal.algorithms.decision_forest.classification.Model class
 *      - com.intel.daal.algorithms.classifier.training.TrainingInput class
 */
public class TrainingBatch extends com.intel.daal.algorithms.classifier.training.TrainingBatch {
    public Parameter      parameter; /*!< Parameters of the algorithm */
    public TrainingMethod method;   /*!< %Training method for the algorithm */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the decision forest classification training algorithm by copying input objects and parameters
     * of another decision forest classification training algorithm
     * @param context   Context to manage decision forest classification training
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public TrainingBatch(DaalContext context, TrainingBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());
        input = new TrainingInput(getContext(), cObject, ComputeMode.batch);
        parameter = new Parameter(getContext(),
                cInitParameter(this.cObject, prec.getValue(), method.getValue(), ComputeMode.batch.getValue()));
    }

    /**
     * Constructs the decision forest classification training algorithm
     * @param context   Context to manage decision forest classification training
     * @param cls       Data type to use in intermediate computations for decision forest classification training,
     *                  Double.class or Float.class
     * @param method    decision forest classification training method, @ref TrainingMethod
     * @param nClasses  Number of classes
     */
    public TrainingBatch(DaalContext context, Class<? extends Number> cls, TrainingMethod method, long nClasses) {
        super(context);

        this.method = method;

        if (this.method != TrainingMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue(), nClasses);
        input = new TrainingInput(getContext(), cObject, ComputeMode.batch);
        parameter = new Parameter(getContext(),
                cInitParameter(this.cObject, prec.getValue(), method.getValue(), ComputeMode.batch.getValue()));
    }

    /**
     * Trains a model of the decision forest classification algorithm
     * @return Structure that contains results of the decision forest classification training algorithm
     */
    @Override
    public TrainingResult compute() {
        super.compute();
        TrainingResult result = new TrainingResult(getContext(), cObject, ComputeMode.batch);
        return result;
    }

    /**
     * Returns the newly allocated decision forest classification training algorithm with a copy of input objects
     * and parameters of this decision forest classification training algorithm
     * @param context   Context to manage decision forest classification training
     *
     * @return The newly allocated algorithm
     */
    @Override
    public TrainingBatch clone(DaalContext context) {
        return new TrainingBatch(context, this);
    }

    private native long cInit(int prec, int method, long nClasses);

    private native long cInitParameter(long algAddr, int prec, int method, int cmode);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
