/* file: Model.java */
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
 * @defgroup svm Support Vector Machine Classifier
 * @brief Contains classes to work with the support vector machine classifier
 * @ingroup classification
 * @{
 */
package com.intel.daal.algorithms.svm;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__MODEL"></a>
 * @brief %Model of the classifier trained by the svm.training.TrainingBatch algorithm.
 */
public class Model extends com.intel.daal.algorithms.classifier.Model {
    private Precision                 prec; /*!< Precision of intermediate computations */
    private double    _bias; /*!< Bias of the model */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public Model(DaalContext context, long cModel) {
        super(context, cModel);
        _bias = cGetBias(cModel);
    }

    /**
     * Returns support vectors constructed during the training of the SVM model
     * @return Array of support vectors
     */
    public NumericTable getSupportVectors() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetSupportVectors(this.getCObject()));
    }

    /**
     * Returns classification coefficients constructed during the training of the SVM model
     * @return Array of classification coefficients
     */
    public NumericTable getClassificationCoefficients() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetClassificationCoefficients(this.getCObject()));
    }

    /**
     * Returns the bias constructed during the training of the SVM model
     * @return Bias
     */
    public double getBias() {
        return _bias;
    }

    private native long cGetSupportVectors(long modelAddr);

    private native long cGetClassificationCoefficients(long modelAddr);

    private native double cGetBias(long modelAddr);
}
/** @} */
