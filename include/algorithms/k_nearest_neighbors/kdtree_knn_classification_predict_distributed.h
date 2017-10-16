/* file: kdtree_knn_classification_predict_distributed.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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

/*
//++
//  Implementation of the interface for K-Nearest Neighbors (kNN) model-based prediction
//--
*/

#ifndef __KDTREE_KNN_CLASSIFICATION_PREDICT_DISTRIBUTED_H__
#define __KDTREE_KNN_CLASSIFICATION_PREDICT_DISTRIBUTED_H__

#include "algorithms/algorithm.h"
#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_predict_types.h"
#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_model.h"
#include "algorithms/classifier/classifier_predict.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace prediction
{
namespace interface1
{
/**
 * @defgroup kdtree_knn_classification_prediction_distributed Distributed
 * @ingroup kdtree_knn_classification_prediction
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__DISTRIBUTEDCONTAINER"></a>
 *  \brief Class containing computation methods for KD-tree based kNN model-based prediction in the distributed processing mode
 */
template<ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__DISTRIBUTEDCONTAINER_STEP1LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for KD-tree based kNN model-based prediction
 *        in the first step of the distributed processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step1Local, algorithmFPType, method, cpu> : public DistributedPredictionContainerIface
{
public:
    /**
     * Constructs a container for KD-tree based kNN model-based prediction with a specified environment
     * in the first step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);

    ~DistributedContainer();

    /**
     * Computes the result of KD-tree based kNN model-based prediction
     * in the first step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;

    /**
     * Computes the result of KD-tree based kNN model-based prediction
     * in the first step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;

};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__DISTRIBUTEDCONTAINER_STEP2LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing computation methods for KD-tree based kNN model-based prediction
 *        in the second step of the distributed processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step2Local, algorithmFPType, method, cpu> : public DistributedPredictionContainerIface
{
public:
    /**
     * Constructs a container for KD-tree based kNN model-based prediction with a specified environment
     * in the second step of the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);

    ~DistributedContainer();

    /**
     * Computes the result of KD-tree based kNN model-based prediction
     * in the second step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;

    /**
     * Computes the result of KD-tree based kNN model-based prediction
     * in the second step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;

};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__DISTRIBUTED"></a>
 * \brief Provides methods to run implementations of the KD-tree based kNN model-based prediction
 * <!-- \n<a href="DAAL-REF-KNN-ALGORITHM">kNN algorithm description and usage models</a> -->
 *
 * \tparam step             Step of the distributed processing mode, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations for KD-tree based kNN model-based prediction
 *                          in the distributed processing mode, double or float
 * \tparam method           Computation method in the distributed processing mode, \ref Method
 *
 * \par Enumerations
 *      - \ref ComputeStep  Computation steps
 *      - \ref Method       Computation methods for KD-tree based kNN model-based prediction
 *
 * \par References
 *      - \ref Distributed class
  */
template<ComputeStep step, typename algorithmFPType = double, Method method = defaultDense>
class Distributed : public daal::algorithms::DistributedPrediction {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__DISTRIBUTED_STEP1LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Provides methods to run implementations of the KD-tree based kNN model-based prediction
 *        in the first step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-KNN-ALGORITHM">kNN algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for KD-tree based kNN model-based prediction
 *                          in the distributed processing mode, double or float
 * \tparam method           Computation method in the distributed processing mode, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods for KD-tree based kNN model-based prediction
 *
 * \par References
 *      - \ref DistributedInput<step1Local> class
 */
template<typename algorithmFPType, Method method>
class Distributed<step1Local, algorithmFPType, method> : public daal::algorithms::DistributedPrediction
{
public:
    DistributedInput<step1Local> input; /*!< %Input data structure */
    Parameter parameter;                /*!< \ref kdtree_knn_classification::interface1::Parameter "Parameters" of prediction */

    /** Default constructor */
    Distributed() : daal::algorithms::DistributedPrediction(), input(), parameter()
    {
        initialize();
    }

    /**
     * Constructs a KD-tree based kNN prediction algorithm by copying input objects and parameters
     * of another KD-tree based kNN prediction algorithm
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step1Local, algorithmFPType, method> & other) : daal::algorithms::DistributedPrediction(other),
                                                                                  input(),
                                                                                  parameter(other.parameter)
    {
        initialize();
        this->input.set(kdtree_knn_classification::prediction::data,  other.input.get(kdtree_knn_classification::prediction::data));
        this->input.set(kdtree_knn_classification::prediction::partialModel, other.input.get(kdtree_knn_classification::prediction::partialModel));
    }

    virtual ~Distributed() {}

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Registers user-allocated memory to store the partial result of KD-tree based kNN model-based prediction
     * \param[in] partialResult    Structure to store the partial result of KD-tree based kNN model-based prediction
     */
    services::Status setPartialResult(const services::SharedPtr<DistributedPartialResultStep1> & partialResult)
    {
        DAAL_CHECK(partialResult, ErrorNullPartialResult)
        _partialResult = partialResult;
        _pres = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    services::SharedPtr<DistributedPartialResultStep1> getPartialResult() { return _partialResult; }

    /**
     * Returns a pointer to a newly allocated KD-tree based kNN prediction algorithm
     * with a copy of the input objects and parameters for this KD-tree based kNN prediction algorithm
     * in the distributed processing mode
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step1Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step1Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    services::SharedPtr<DistributedPartialResultStep1> _partialResult;

    virtual Distributed<step1Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step1Local, algorithmFPType, method>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE { return services::Status(); }

    services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(_in, &parameter, (int)method);
        _pres = _partialResult.get();
        return s;
    }

    services::Status initializePartialResult() DAAL_C11_OVERRIDE
    {
        _pres = _partialResult.get();
        return services::Status();
    }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step1Local, algorithmFPType, method)(&_env);
        _par = &parameter;
        _in = &input;
        _partialResult = services::SharedPtr<DistributedPartialResultStep1>(new DistributedPartialResultStep1());
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__DISTRIBUTED_STEP2LOCAL_ALGORITHMFPTYPE_METHOD"></a>
 * \brief Provides methods to run implementations of the KD-tree based kNN model-based prediction
 *        in the second step of the distributed processing mode
 * <!-- \n<a href="DAAL-REF-KNN-ALGORITHM">kNN algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for KD-tree based kNN model-based prediction
 *                          in the distributed processing mode, double or float
 * \tparam method           Computation method in the distributed processing mode, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods for KD-tree based kNN model-based prediction
 *
 * \par References
 *      - \ref DistributedInput<step1Local> class
 */
template<typename algorithmFPType, Method method>
class Distributed<step2Local, algorithmFPType, method> : public daal::algorithms::DistributedPrediction
{
public:
    DistributedInput<step2Local> input; /*!< %Input data structure */
    Parameter parameter;                /*!< \ref kdtree_knn_classification::interface1::Parameter "Parameters" of prediction */

    /** Default constructor */
    Distributed() : daal::algorithms::DistributedPrediction(), input(), parameter()
    {
        initialize();
    }

    /**
     * Constructs a KD-tree based kNN prediction algorithm by copying input objects and parameters
     * of another KD-tree based kNN prediction algorithm
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed<step2Local, algorithmFPType, method> & other) : daal::algorithms::DistributedPrediction(other),
                                                                                  input(),
                                                                                  parameter(other.parameter)
    {
        initialize();
        this->input.set(kdtree_knn_classification::prediction::arrangedData, other.input.get(kdtree_knn_classification::prediction::arrangedData));
        this->input.set(kdtree_knn_classification::prediction::intermediatePrediction,
                        other.input.get(kdtree_knn_classification::prediction::intermediatePrediction));
        this->input.set(kdtree_knn_classification::prediction::partialModel, other.input.get(kdtree_knn_classification::prediction::partialModel));
        this->input.set(kdtree_knn_classification::prediction::communicationResponses,
                        other.input.get(kdtree_knn_classification::prediction::communicationResponses));
        this->input.set(kdtree_knn_classification::prediction::communicationInputQueries,
                        other.input.get(kdtree_knn_classification::prediction::communicationInputQueries));
        this->input.set(kdtree_knn_classification::prediction::key, other.input.get(kdtree_knn_classification::prediction::key));
        this->input.set(kdtree_knn_classification::prediction::round, other.input.get(kdtree_knn_classification::prediction::round));
    }

    virtual ~Distributed() {}

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Registers user-allocated memory to store the partial result of KD-tree based kNN model-based prediction
     * \param[in] partialResult    Structure to store the partial result of KD-tree based kNN model-based prediction
     */
    services::Status setPartialResult(const services::SharedPtr<DistributedPartialResultStep2> & partialResult)
    {
        DAAL_CHECK(partialResult, ErrorNullPartialResult)
        _partialResult = partialResult;
        _pres = _partialResult.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains computed partial results
     * \return Structure that contains computed partial results
     */
    services::SharedPtr<DistributedPartialResultStep2> getPartialResult() { return _partialResult; }

    /**
     * Returns a pointer to a newly allocated KD-tree based kNN prediction algorithm
     * with a copy of the input objects and parameters for this KD-tree based kNN prediction algorithm
     * in the distributed processing mode
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step2Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step2Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    services::SharedPtr<DistributedPartialResultStep2> _partialResult;

    virtual Distributed<step2Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step2Local, algorithmFPType, method>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE { return services::Status(); }

    services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _partialResult->allocate<algorithmFPType>(_in, &parameter, (int)method);
        _pres = _partialResult.get();
        return s;
    }

    services::Status initializePartialResult() DAAL_C11_OVERRIDE
    {
        _pres = _partialResult.get();
        return services::Status();
    }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Local, algorithmFPType, method)(&_env);
        _par = &parameter;
        _in = &input;
        _partialResult = services::SharedPtr<DistributedPartialResultStep2>(new DistributedPartialResultStep2());
    }
};

/** @} */
} // namespace interface1

using interface1::DistributedContainer;
using interface1::Distributed;

} // namespace prediction
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
