/* file: DistributedStep3Local.java */
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
 * @ingroup svd_distributed
 * @{
 */
package com.intel.daal.algorithms.svd;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisDistributed;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTEDSTEP3LOCAL"></a>
 * @brief Runs the third step of the SVD algorithm in the distributed processing mode
 * <!-- \n<a href="DAAL-REF-SVD-ALGORITHM">SVD algorithm description and usage models</a> -->
 *
 * @par References
 *      - DistributedStep3LocalInputId class. Identifiers of SVD input objects
 *      - ResultId class. Identifiers of SVD results
 *      - ResultFormat class. Options to return SVD output matrices
 */
public class DistributedStep3Local extends AnalysisDistributed {
    public DistributedStep3LocalInput          input;        /*!< %Input data */
    public Parameter  parameter;     /*!< Parameters of the algorithm */
    public Method                               method;  /*!< Computation method for the algorithm */
    private Result    result;      /*!< %Result of the algorithm */
    private DistributedStep3LocalPartialResult partialResult;   /*!< Partial result of the algorithm */
    private Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the SVD algorithm by copying input objects and parameters
     * of another SVD algorithm
     * @param context   Context to manage created SVD algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public DistributedStep3Local(DaalContext context, DistributedStep3Local other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new DistributedStep3LocalInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the SVD algorithm
     * @param context   Context to manage created SVD algorithm
     * @param cls       Data type to use in intermediate computations for the SVD algorithm,
     *                  Double.class or Float.class
     * @param method    SVD computation method, @ref Method
     */
    public DistributedStep3Local(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);

        this.method = method;
        if (this.method != Method.defaultDense) {
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

        this.cObject = cInitDistributed(prec.getValue(), method.getValue());
        input = new DistributedStep3LocalInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Runs the SVD algorithm
     * @return  Partial results of the SVD algorithm obtained in the third step in the distributed processing mode
     */
    @Override
    public DistributedStep3LocalPartialResult compute() {
        super.compute();
        partialResult = new DistributedStep3LocalPartialResult(getContext(), cGetPartialResult(cObject, prec.getValue(), method.getValue()));
        result = new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return partialResult;
    }

    /**
     * Computes final results of the SVD algorithm
     * @return  Final results of the SVD algorithm
     */
    @Override
    public Result finalizeCompute() {
        return result;
    }

    /**
     * Returns the newly allocated SVD algorithm
     * with a copy of input objects and parameters of this SVD algorithm
     * @param context   Context to manage created SVD algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public DistributedStep3Local clone(DaalContext context) {
        return new DistributedStep3Local(context, this);
    }

    private native long cInitDistributed(int prec, int method);

    private native long cInitParameter(long addr, int prec, int method);

    private native long cGetInput(long addr, int prec, int method);

    private native long cGetResult(long addr, int prec, int method);

    private native long cGetPartialResult(long addr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
