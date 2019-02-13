/* file: Batch.java */
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
 * @defgroup iterative_solver Iterative Solver
 * @brief Contains classes for computing the iterative solver
 * @ingroup optimization_solver
 * @{
 */
/**
 * @defgroup iterative_solver_batch Batch
 * @ingroup iterative_solver
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.iterative_solver;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__BATCH"></a>
 * @brief %Base interface for the iterative solver algorithm in the batch processing mode
 * <!-- \n<a href="DAAL-REF-ITERATIVE_SOLVER-ALGORITHM">iterative solver algorithm description and usage models</a> -->
 *
 * @par References
 *      - Parameter class
 *      - com.intel.daal.algorithms.optimization_solver.iterative_solver.InputId class
 *      - com.intel.daal.algorithms.optimization_solver.iterative_solver.ResultId class
 *
 */
public class Batch extends com.intel.daal.algorithms.optimization_solver.BatchIface {

    public Input input;     /*!< %Input data */
    public Parameter parameter; /*!< Parameters of the algorithm */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the iterative solver algorithm by copying input objects and parameters of another iterative solver algorithm
     * @param context    Context to manage the iterative solver algorithm
     * @param other      An algorithm to be used as the source to initialize the input objects
     *                   and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);

        this.cObject = cClone(other.cObject);
        input = new Input(context, cGetInput(cObject));
        parameter = new Parameter(getContext(), cGetParameter(this.cObject));
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__BATCH__BATCH"></a>
     * Constructs the iterative solver algorithm
     *
     * @param context      Context to manage the iterative solver algorithm
     */
    public Batch(DaalContext context) {
        super(context);
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__BATCH__BATCH"></a>
     * Constructs the iterative solver algorithm
     *
     * @param context      Context to manage the iterative solver algorithm
     * @param cAlgorithm   Pointer to the C++ implememntation
     */
    public Batch(DaalContext context, long cAlgorithm) {
        super(context);

        this.cObject = cAlgorithm;
        input = new Input(context, cGetInput(cObject));
        parameter = new Parameter(getContext(), cGetParameter(this.cObject));
    }

    /**
     * Computes the iterative solver in the batch processing mode
     * @return  Results of the computation
     */
    @Override
    public Result compute() {
        super.compute();
        Result result = new Result(getContext(), cGetResult(cObject));
        return result;
    }

    /**
     * Returns the newly allocated iterative solver algorithm
     * with a copy of input objects and parameters of this iterative solver algorithm
     * @param context    Context to manage the iterative solver algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit();
    private native long cClone(long algAddr);
    private native long cGetInput(long cAlgorithm);
    private native long cGetParameter(long cAlgorithm);
    protected native long cGetResult(long cAlgorithm);
}
/** @} */
/** @} */
