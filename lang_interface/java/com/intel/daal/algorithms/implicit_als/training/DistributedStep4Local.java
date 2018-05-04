/* file: DistributedStep4Local.java */
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
 * @ingroup implicit_als_training_distributed
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.TrainingDistributed;
import com.intel.daal.algorithms.implicit_als.Parameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDSTEP4LOCAL"></a>
 * @brief Runs the implicit ALS training algorithm in the fourth step of the distributed processing mode
 */
public class DistributedStep4Local extends TrainingDistributed {
    public DistributedStep4LocalInput input;        /*!< %Input data */
    public Parameter  parameter;     /*!< Parameters of the algorithm */
    public TrainingMethod method;   /*!< %Training method for the algorithm */
    private Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the implicit ALS training algorithm in the fourth step of the distributed processing mode
     * by copying input objects and parameters of another implicit ALS training algorithm
     * @param context   Context to manage the implicit ALS training algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public DistributedStep4Local(DaalContext context, DistributedStep4Local other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());
        input = new DistributedStep4LocalInput(getContext(), cObject, prec, method);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the implicit ALS training algorithm in the fourth step of the distributed processing mode
     * @param context   Context to manage the implicit ALS training algorithm
     * @param cls       Data type to use in intermediate computations for the implicit ALS training algorithm,
     *                  Double.class or Float.class
     * @param method    Implicit ALS computation method, @ref TrainingMethod
     */
    public DistributedStep4Local(DaalContext context, Class<? extends Number> cls, TrainingMethod method) {
        super(context);

        this.method = method;
        if (this.method != TrainingMethod.fastCSR && this.method != TrainingMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else if (cls == Float.class) {
            prec = Precision.singlePrecision;
        } else {
            throw new IllegalArgumentException("type unsupported");
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue());

        input = new DistributedStep4LocalInput(getContext(), cObject, prec, method);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes partial results of the implicit ALS training algorithm obtained in the fourth step of the distributed processing mode
     * @return Partial results of the implicit ALS training algorithm obtained in the fourth step of the distributed processing mode
     */
    @Override
    public DistributedPartialResultStep4 compute() {
        super.compute();
        return new DistributedPartialResultStep4(getContext(), cGetPartialResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Registers user-allocated memory to store partial results of the implicit ALS training algorithm
     * obtained in the fourth step of the distributed processing mode
     * @param partialResult         Structure to store partial results of the implicit ALS training algorithm
     * obtained in the fourth step of the distributed processing mode
     */
    public void setPartialResult(DistributedPartialResultStep4 partialResult) {
        cSetPartialResult(this.cObject, prec.getValue(), method.getValue(), partialResult.getCObject());
    }

    /**
     * Returns the newly allocated ALS training algorithm in the fourth step of the distributed processing mode
     * with a copy of input objects and parameters of this ALS training algorithm
     * @param context   Context to manage the implicit ALS training algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public DistributedStep4Local clone(DaalContext context) {
        return new DistributedStep4Local(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native long cGetPartialResult(long cObject, int prec, int method);

    private native void cSetPartialResult(long cObject, int prec, int method, long cPartialResult);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
