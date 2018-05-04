/* file: PredictionBatch.java */
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
 * @defgroup ridge_regression_prediction_batch Batch
 * @ingroup ridge_regression_prediction
 * @{
 */
package com.intel.daal.algorithms.ridge_regression.prediction;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__PREDICTION__PREDICTIONBATCH"></a>
 * @brief Provides methods for ridge regression model-based prediction
 * <!-- \n<a href="DAAL-REF-RIDGEREGRESSION-ALGORITHM">Ridge regression algorithm description and usage models</a> -->
 *
 * @par References
 *      - Model class
 *      - ModelNormEq class
 *      - PredictionInputId class
 *      - PredictionResultId class
 */
public class PredictionBatch extends com.intel.daal.algorithms.Prediction {
    public Input            input;     /*!< %Input data */
    public PredictionMethod method; /*!< %Prediction method for the algorithm */

    private Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs a ridge regression prediction algorithm by copying
     * input objects and parameters of another ridge regression prediction algorithm
     * @param context   Context to manage ridge regression model-based prediction
     * @param other     %Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public PredictionBatch(DaalContext context, PredictionBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the ridge regression prediction algorithm in the batch processing mode
     * @param context   Context to manage ridge regression model-based prediction
     * @param cls       Data type to use in intermediate computations of ridge regression,
     *                  Double.class or Float.class
     * @param method    %Algorithm prediction method, @ref PredictionMethod
     */
    public PredictionBatch(DaalContext context, Class<? extends Number> cls, PredictionMethod method) {
        super(context);

        this.method = method;
        if (this.method != PredictionMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), method.getValue());
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of ridge regression model-based prediction in the batch processing mode
     * @return Result of ridge regression model-based prediction
     */
    @Override
    public PredictionResult compute() {
        super.compute();
        return new PredictionResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Registers user-allocated memory to store the result of ridge regression model-based prediction
     * @param result Object to store the result of ridge regression model-based prediction
     */
    public void setResult(PredictionResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns a newly allocated ridge regression prediction algorithm
     * with a copy of the input objects of this ridge regression prediction algorithm
     * in the batch processing mode
     * @param context   Context to manage ridge regression model-based prediction
     *
     * @return Newly allocated algorithm
     */
    @Override
    public PredictionBatch clone(DaalContext context) {
        return new PredictionBatch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);

    private native long cGetResult(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
