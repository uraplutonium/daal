/* file: TrainingParameter.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

package com.intel.daal.algorithms.neural_networks.training;

import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.Precision;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__TRAININGPARAMETER"></a>
 * \brief Class representing the parameters of neural network
 */
public class TrainingParameter extends com.intel.daal.algorithms.Parameter {
    Precision prec;

    /**
     * Constructs the parameters of neural network algorithm
     * @param context   Context to manage the parameter object
     * @param cls       Data type to use in intermediate computations for the neural network,
     *                  Double.class or Float.class
     */
    public TrainingParameter(DaalContext context, Class <? extends Number> cls) {
        super(context);

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        cObject = cInit(prec.getValue());
    }

    public TrainingParameter(DaalContext context, Precision prec, long cParameter) {
        super(context, cParameter);
        this.prec = prec;
    }

    /**
     *  Gets the size of the batch to be processed by the neural network
     */
    public long getBatchSize() {
        return cGetBatchSize(cObject, prec.getValue());
    }

    /**
     *  Sets the size of the batch to be processed by the neural network
     *  @param batchSize Size of the batch to be processed by the neural network
     */
    public void setBatchSize(long batchSize) {
        cSetBatchSize(cObject, prec.getValue(), batchSize);
    }

    /**
     *  Gets the maximal number of iterations of the algorithm
     */
    public long getNIterations() {
        return cGetNIterations(cObject, prec.getValue());
    }

    /**
     *  Sets the maximal number of iterations of the algorithm
     *  @param nIterations Maximal number of iterations of the algorithm
     */
    public void setNIterations(long nIterations) {
        cSetNIterations(cObject, prec.getValue(), nIterations);
    }

    /**
     *  Gets the optimization solver used in the neural network
     */
    public com.intel.daal.algorithms.optimization_solver.sgd.Batch getOptimizationSolver() {
        if (prec == Precision.singlePrecision) {
            return new com.intel.daal.algorithms.optimization_solver.sgd.Batch(getContext(),
                                                                               Float.class,
                                                                               com.intel.daal.algorithms.optimization_solver.sgd.Method.defaultDense,
                                                                               cGetOptimizationSolver(cObject, prec.getValue()));
        } else {
            return new com.intel.daal.algorithms.optimization_solver.sgd.Batch(getContext(),
                                                                               Double.class,
                                                                               com.intel.daal.algorithms.optimization_solver.sgd.Method.defaultDense,
                                                                               cGetOptimizationSolver(cObject, prec.getValue()));
        }
    }

    /**
     *  Sets the optimization solver used in the neural network
     *  @param optimizationSolver Optimization solver used in the neural network
     */
    public void setOptimizationSolver(com.intel.daal.algorithms.optimization_solver.Batch optimizationSolver) {
       cSetOptimizationSolver(cObject, prec.getValue(), optimizationSolver.cObject);
    }

    /**
     *  Gets the objective function used in the neural network
     */
    public com.intel.daal.algorithms.optimization_solver.mse.Batch getObjectiveFunction() {
        return null;
//TODO THROW EXCEPTION
//        return new com.intel.daal.algorithms.optimization_solver.mse.Batch(getContext(), cGetObjectiveFunction(cObject, prec.getValue()));
    }

    /**
     *  Sets the objective function used in the neural network
     *  @param objectiveFunction objective function used in the neural network
     */
    public void setObjectiveFunction(com.intel.daal.algorithms.optimization_solver.mse.Batch objectiveFunction) {
       cSetObjectiveFunction(cObject, prec.getValue(), objectiveFunction.cObject);
    }

    private native long cInit(long prec);
    private native long cGetBatchSize(long cParameter, long prec);
    private native void cSetBatchSize(long cParameter, long prec, long batchSize);
    private native long cGetNIterations(long cParameter, long prec);
    private native void cSetNIterations(long cParameter, long prec, long nIterations);
    private native long cGetOptimizationSolver(long cParameter, long prec);
    private native void cSetOptimizationSolver(long cParameter, long prec, long optAddr);
//    private native long cGetObjectiveFunction(long cParameter);
    private native void cSetObjectiveFunction(long cParameter, long prec, long objAddr);
}
