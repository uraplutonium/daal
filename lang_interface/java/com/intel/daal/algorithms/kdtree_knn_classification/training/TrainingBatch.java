/* file: TrainingBatch.java */
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
 * @defgroup kdtree_knn_classification_batch Batch
 * @ingroup kdtree_knn_classification_training
 * @{
 */
package com.intel.daal.algorithms.kdtree_knn_classification.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.kdtree_knn_classification.Parameter;
import com.intel.daal.algorithms.classifier.training.TrainingInput;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__TRAININGBATCH"></a>
 * @brief Provides methods for k nearest neighbors model-based training in the batch processing mode
 * <!-- \n<a href="DAAL-REF-KNN-ALGORITHM">k nearest neighbors algorithm description and usage models</a> -->
 */
public class TrainingBatch extends com.intel.daal.algorithms.classifier.training.TrainingBatch {
    public  Parameter      parameter; /*!< Parameters of the algorithm */
    public  TrainingMethod method;    /*!< %Training method for the algorithm */
    private Precision      prec;      /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs a k nearest neighbors training algorithm by copying input objects
     * and parameters of another k nearest neighbors training algorithm
     * in the batch processing mode
     * @param context   Context to manage k nearest neighbors model-based training
     * @param other     Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public TrainingBatch(DaalContext context, TrainingBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new TrainingInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the k nearest neighbors training algorithm in the batch processing mode
     * @param context   Context to manage k nearest neighbors model-based training
     * @param cls       Data type to use in intermediate computations of k nearest neighbors,
     *                  Double.class or Float.class
     * @param method    Algorithm computation method, @ref TrainingMethod
     */
    public TrainingBatch(DaalContext context, Class<? extends Number> cls, TrainingMethod method) {
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

        this.cObject = cInit(prec.getValue(), method.getValue());
        input = new TrainingInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Computes the result of k nearest neighbors model-based training
     * in the batch processing mode
     * @return Result of k nearest neighbors model-based training
     */
    @Override
    public TrainingResult compute() {
        super.compute();
        TrainingResult result = new TrainingResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Returns a newly allocated k nearest neighbors training algorithm
     * with a copy of the input objects and parameters of this k nearest neighbors
     * training algorithm in the batch processing mode
     * @param context   Context to manage k nearest neighbors model-based training
     *
     * @return Newly allocated algorithm
     */
    @Override
    public TrainingBatch clone(DaalContext context) {
        return new TrainingBatch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cInitParameter(long algAddr, int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
