/* file: Online.java */
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
 * @defgroup covariance_online Online
 * @ingroup covariance
 * @{
 */
package com.intel.daal.algorithms.covariance;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisOnline;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__ONLINE"></a>
 * @brief Computes the correlation or variance-covariance matrix in the online processing mode
 * <!-- \n<a href="DAAL-REF-COVARIANCE-ALGORITHM">Correlation or variance-covariance matrix algorithm description and usage models</a> -->
 *
 * @par References
 *      - InputId class. Identifiers of input objects
 *      - PartialResultId class. Identifiers of partial results
 *      - ResultId class. Identifiers of the results
 *      - Input class
 *      - OnlineParameter class
 *      - PartialResult class
 *      - Result class
 */
public class Online extends OnlineImpl {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the correlation or variance-covariance matrix algorithm in the online processing mode
     * by copying input objects and parameters of another algorithm for correlation or variance-covariance
     * matrix computation
     * @param context   Context to manage the correlation or variance-covariance matrix algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public Online(DaalContext context, Online other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());

        input = new Input(getContext(), cObject, prec, method, ComputeMode.online);
        parameter = new OnlineParameter(getContext(),  cInitParameter(cObject, prec.getValue(), method.getValue()), cObject);
    }

    /**
     * Constructs the correlation or variance-covariance matrix algorithm in the online processing mode
     * @param context   Context to manage the correlation or variance-covariance matrix algorithm
     * @param cls       Data type to use in intermediate computations of the correlation or variance-covariance matrix,
     *                  Double.class or Float.class
     * @param method    Computation method, @ref Method
     */
    public Online(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);
        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != Method.defaultDense && this.method != Method.singlePassDense
            && this.method != Method.sumDense && this.method != Method.fastCSR
            && this.method != Method.singlePassCSR && this.method != Method.sumCSR) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue());

        input = new Input(getContext(), cObject, prec, method, ComputeMode.online);
        parameter = new OnlineParameter(getContext(),  cInitParameter(cObject, prec.getValue(), method.getValue()), cObject);
    }

    /**
     * Returns the newly allocated correlation or variance-covariance matrix algorithm in the online processing mode
     * with a copy of input objects and parameters of this correlation or variance-covariance matrix algorithm
     * @param context   Context to manage the correlation or variance-covariance matrix algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Online clone(DaalContext context) {
        return new Online(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cClone(long cAlgorithm, int prec, int method);

    private native long cInitParameter(long cAlgorithm, int prec, int method);

}
/** @} */
