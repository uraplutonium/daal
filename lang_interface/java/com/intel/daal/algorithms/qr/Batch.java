/* file: Batch.java */
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
* @defgroup qr QR Decomposition
* @brief Contains classes for computing the results of the QR decomposition algorithm
* @ingroup analysis
* @{
*/
/**
* @defgroup qr_without_pivoting QR Decomposition without Pivoting
* @brief Contains classes for computing the results of the QR decomposition algorithm without Pivoting
* @ingroup qr
* @{
*/
/**
 * @defgroup qr_batch Batch
 * @ingroup qr_without_pivoting
 * @{
 */
/**
 * @brief Contains classes for computing the QR decomposition
 */
package com.intel.daal.algorithms.qr;

import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QR__BATCH"></a>
 * @brief Computes the results of the QR decomposition algorithm in the batch processing mode
 * <!-- \n<a href="DAAL-REF-QR-ALGORITHM">QR algorithm description and usage models</a> -->
 *
 * @par References
 *      - InputId class. Identifiers of input objects for the QR decomposition algorithm
 *      - ResultId class. Identifiers of the results of the algorithm
 */
public class Batch extends AnalysisBatch {
    public Input      input; /*!< %Input data */
    public Method     method; /*!< Computation method for the algorithm */
    private Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the QR decomposition algorithm by copying input objects and parameters
     * of another QR decomposition algorithm
     * @param context   Context to manage the QR decomposition algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the QR decomposition algorithm
     * @param context   Context to manage the QR decomposition algorithm
     * @param cls       Data type to use in intermediate computations for the QR decomposition algorithm,
     *                  Double.class or Float.class
     * @param method    Computation method, @ref Method
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);

        this.method = method;

        if (method != Method.defaultDense) {
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
     * Runs the QR decomposition algorithm
     * @return  Results of the QR decomposition algorithm
     */
    @Override
    public Result compute() {
        super.compute();
        return new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Registers user-allocated memory for storing the results of the QR decomposition algorithm
     * @param result Structure for storing the results of the QR decomposition algorithm
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated QR decomposition algorithm
     * with a copy of input objects and parameters of this QR decomposition algorithm
     * @param context   Context to manage the QR decomposition algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    private native void cSetResult(long cAlgorithm, int prec, int method, long cResult);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
/** @} */
/** @} */
