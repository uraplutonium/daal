/* file: kdtree_knn_classification_training_distributed.h */
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
//  Implementation of the interface for k-Nearest Neighbor (kNN) model-based training in the distributed processing mode
//--
*/

#ifndef __KDTREE_KNN_CLASSIFICATION_TRAINING_DISTRIBUTED_H__
#define __KDTREE_KNN_CLASSIFICATION_TRAINING_DISTRIBUTED_H__

#include "algorithms/algorithm.h"
#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_training_types.h"
#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_model.h"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace training
{
namespace interface1
{
/**
 * @defgroup kdtree_knn_classification_distributed Distributed
 * @ingroup kdtree_knn_classification
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDCONTAINER"></a>
 * \brief Class containing methods for KD-tree based kNN model-based training in the distributed processing mode
 */
template<ComputeStep step, typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDCONTAINER_STEP1LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods for KD-tree based kNN model-based training in the first step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step1Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for KD-tree based kNN model-based training with a specified environment in the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);

    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes the result of KD-tree based kNN model-based training in the first step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;

    /**
     * Computes the result of KD-tree based kNN model-based training in the first step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDCONTAINER_STEP2MASTER_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods for KD-tree based kNN model-based training in the second step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step2Master, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for KD-tree based kNN model-based training with a specified environment in the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);

    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes the result of KD-tree based kNN model-based training in the second step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;

    /**
     * Computes the result of KD-tree based kNN model-based training in the second step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDCONTAINER_STEP3LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods for KD-tree based kNN model-based training in the third step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step3Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for KD-tree based kNN model-based training with a specified environment in the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);

    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes the result of KD-tree based kNN model-based training in the third step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;

    /**
     * Computes the result of KD-tree based kNN model-based training in the third step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDCONTAINER_STEP4LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods for KD-tree based kNN model-based training in the fourth step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step4Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for KD-tree based kNN model-based training with a specified environment in the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);

    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes the result of KD-tree based kNN model-based training in the fourth step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;

    /**
     * Computes the result of KD-tree based kNN model-based training in the fourth step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDCONTAINER_STEP5LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods for KD-tree based kNN model-based training in the fifth step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step5Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for KD-tree based kNN model-based training with a specified environment in the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);

    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes the result of KD-tree based kNN model-based training in the fifth step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;

    /**
     * Computes the result of KD-tree based kNN model-based training in the fifth step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDCONTAINER_STEP5LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods for KD-tree based kNN model-based training in the sixth step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step6Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for KD-tree based kNN model-based training with a specified environment in the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);

    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes the result of KD-tree based kNN model-based training in the sixth step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;

    /**
     * Computes the result of KD-tree based kNN model-based training in the sixth step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDCONTAINER_STEP5LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods for KD-tree based kNN model-based training in the seventh step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step7Master, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for KD-tree based kNN model-based training with a specified environment in the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);

    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes the result of KD-tree based kNN model-based training in the seventh step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;

    /**
     * Computes the result of KD-tree based kNN model-based training in the seventh step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDCONTAINER_STEP5LOCAL_ALGORITHMFPTYPE_METHOD_CPU"></a>
 * \brief Class containing methods for KD-tree based kNN model-based training in the eighth step of the distributed processing mode
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT DistributedContainer<step8Local, algorithmFPType, method, cpu> : public TrainingContainerIface<distributed>
{
public:
    /**
     * Constructs a container for KD-tree based kNN model-based training with a specified environment in the distributed processing mode
     * \param[in] daalEnv   Environment object
     */
    DistributedContainer(daal::services::Environment::env * daalEnv);

    /** Default destructor */
    ~DistributedContainer();

    /**
     * Computes the result of KD-tree based kNN model-based training in the eighth step of the distributed processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;

    /**
     * Computes the result of KD-tree based kNN model-based training in the eighth step of the distributed processing mode
     */
    services::Status finalizeCompute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTED"></a>
 * \brief Provides methods for KD-tree based kNN model-based training in the distributed processing mode
 *
 * \tparam step             Step of the algorithm in the distributed processing mode, \ref ComputeStep
 * \tparam algorithmFPType  Data type to use in intermediate computations for KD-tree based kNN model-based training, double or float
 * \tparam method           KD-tree based kNN model-based training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref interface1::Parameter "Parameter" class
 *      - \ref kdtree_knn_classification::interface1::Model "kdtree_knn_classification::Model" class
 *      - \ref prediction::interface1::Distributed "prediction::Distributed" class
 */
template<ComputeStep step, typename algorithmFPType = double, Method method = defaultDense>
class DAAL_EXPORT Distributed : public Training<distributed>
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTED"></a>
 * \brief Provides methods for KD-tree based kNN model-based training in the distributed processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for KD-tree based kNN model-based training, double or float
 * \tparam method           KD-tree based kNN training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref kdtree_knn_classification::interface1::Model "kdtree_knn_classification::Model" class
 *      - \ref prediction::interface1::Distributed "prediction::Distributed" class
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step1Local, algorithmFPType, method> : public Training<distributed>
{
public:
    DistributedInput<step1Local> input; /*!< %Input objects for the KD-tree based kNN model training algorithm in the distributed processing mode */
    Parameter parameter; /*!< \ref interface1::Parameter "Parameters" of the algorithm */

    /** Default constructor */
    Distributed()
    {
        initialize();
    }

    /**
     * Constructs a KD-tree based kNN training algorithm by copying input objects
     * and parameters of another KD-tree based kNN training algorithm in the distributed processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed & other)
    {
        initialize();
        input.set(classifier::training::data, other.input.get(classifier::training::data));
        input.set(classifier::training::labels, other.input.get(classifier::training::labels));
        parameter = other.parameter;
    }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Registers user-allocated memory to store the partial result of KD-tree based kNN model-based training
     * \param[in] partialResult    Structure to store the partial result of KD-tree based kNN model-based training
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
     * Returns a pointer to a newly allocated KD-tree based kNN training algorithm
     * with a copy of the input objects and parameters for this KD-tree based kNN training algorithm
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
        _partialResult = services::SharedPtr<DistributedPartialResultStep1>(new DistributedPartialResultStep1());
        services::Status s = _partialResult->allocate<algorithmFPType>(_in, 0, 0);
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
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTED"></a>
 * \brief Provides methods for KD-tree based kNN model-based training in the distributed processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for KD-tree based kNN model-based training, double or float
 * \tparam method           KD-tree based kNN training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref kdtree_knn_classification::interface1::Model "kdtree_knn_classification::Model" class
 *      - \ref prediction::interface1::Distributed "prediction::Distributed" class
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step2Master, algorithmFPType, method> : public Training<distributed>
{
public:
    DistributedInput<step2Master> input; /*!< %Input objects for the KD-tree based kNN model training algorithm in the distributed processing mode */
    Parameter parameter; /*!< \ref interface1::Parameter "Parameters" of the algorithm */

    /** Default constructor */
    Distributed()
    {
        initialize();
    }

    /**
     * Constructs a KD-tree based kNN training algorithm by copying input objects
     * and parameters of another KD-tree based kNN training algorithm in the distributed processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed & other)
    {
        initialize();
        input.set(inputOfStep2, other.input.get(inputOfStep2));
        parameter = other.parameter;
    }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Registers user-allocated memory to store the partial result of KD-tree based kNN model-based training
     * \param[in] partialResult    Structure to store the partial result of KD-tree based kNN model-based training
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
     * Returns a pointer to a newly allocated KD-tree based kNN training algorithm
     * with a copy of the input objects and parameters for this KD-tree based kNN training algorithm
     * in the distributed processing mode
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step2Master, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step2Master, algorithmFPType, method> >(cloneImpl());
    }

protected:
    services::SharedPtr<DistributedPartialResultStep2> _partialResult;

    virtual Distributed<step2Master, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step2Master, algorithmFPType, method>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE { return services::Status(); }

    services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult = services::SharedPtr<DistributedPartialResultStep2>(new DistributedPartialResultStep2());
        services::Status s = _partialResult->allocate<algorithmFPType>(_in, 0, 0);
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
        _ac  = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Master, algorithmFPType, method)(&_env);
        _par = &parameter;
        _in = &input;
        _partialResult = services::SharedPtr<DistributedPartialResultStep2>(new DistributedPartialResultStep2());
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTED"></a>
 * \brief Provides methods for KD-tree based kNN model-based training in the distributed processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for KD-tree based kNN model-based training, double or float
 * \tparam method           KD-tree based kNN training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref kdtree_knn_classification::interface1::Model "kdtree_knn_classification::Model" class
 *      - \ref prediction::interface1::Distributed "prediction::Distributed" class
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step3Local, algorithmFPType, method> : public Training<distributed>
{
public:
    DistributedInput<step3Local> input; /*!< %Input objects for the KD-tree based kNN model training algorithm in the distributed processing mode */
    Parameter parameter; /*!< \ref interface1::Parameter "Parameters" of the algorithm */

    /** Default constructor */
    Distributed()
    {
        initialize();
    }

    /**
     * Constructs a KD-tree based kNN training algorithm by copying input objects
     * and parameters of another KD-tree based kNN training algorithm in the distributed processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed & other)
    {
        initialize();
        input.set(dataForStep3, other.input.get(dataForStep3));
        input.set(labelsForStep3, other.input.get(labelsForStep3));
        input.set(boundingBoxesForStep3, other.input.get(boundingBoxesForStep3));
        parameter = other.parameter;
    }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Registers user-allocated memory to store the partial result of KD-tree based kNN model-based training
     * \param[in] partialResult    Structure to store the partial result of KD-tree based kNN model-based training
     */
    services::Status setPartialResult(const services::SharedPtr<DistributedPartialResultStep3> & partialResult)
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
    services::SharedPtr<DistributedPartialResultStep3> getPartialResult() { return _partialResult; }

    /**
     * Returns a pointer to a newly allocated KD-tree based kNN training algorithm
     * with a copy of the input objects and parameters for this KD-tree based kNN training algorithm
     * in the distributed processing mode
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step3Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step3Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    services::SharedPtr<DistributedPartialResultStep3> _partialResult;

    virtual Distributed<step3Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step3Local, algorithmFPType, method>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE { return services::Status(); }

    services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult = services::SharedPtr<DistributedPartialResultStep3>(new DistributedPartialResultStep3());
        services::Status s= _partialResult->allocate<algorithmFPType>(_in, 0, 0);
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
        _ac  = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step3Local, algorithmFPType, method)(&_env);
        _par = &parameter;
        _in = &input;
        _partialResult = services::SharedPtr<DistributedPartialResultStep3>(new DistributedPartialResultStep3());
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTED"></a>
 * \brief Provides methods for KD-tree based kNN model-based training in the distributed processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for KD-tree based kNN model-based training, double or float
 * \tparam method           KD-tree based kNN training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref kdtree_knn_classification::interface1::Model "kdtree_knn_classification::Model" class
 *      - \ref prediction::interface1::Distributed "prediction::Distributed" class
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step4Local, algorithmFPType, method> : public Training<distributed>
{
public:
    DistributedInput<step4Local> input; /*!< %Input objects for the KD-tree based kNN model training algorithm in the distributed processing mode */
    Parameter parameter; /*!< \ref interface1::Parameter "Parameters" of the algorithm */

    /** Default constructor */
    Distributed()
    {
        initialize();
    }

    /**
     * Constructs a KD-tree based kNN training algorithm by copying input objects
     * and parameters of another KD-tree based kNN training algorithm in the distributed processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed & other)
    {
        initialize();
        input.set(dataForStep4, other.input.get(dataForStep4));
        input.set(labelsForStep4, other.input.get(labelsForStep4));
        input.set(dimensionForStep4, other.input.get(dimensionForStep4));
        input.set(boundingBoxesForStep4, other.input.get(boundingBoxesForStep4));
        input.set(samplesForStep4, other.input.get(samplesForStep4));
        parameter = other.parameter;
    }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Registers user-allocated memory to store the partial result of KD-tree based kNN model-based training
     * \param[in] partialResult    Structure to store the partial result of KD-tree based kNN model-based training
     */
    services::Status setPartialResult(const services::SharedPtr<DistributedPartialResultStep4> & partialResult)
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
    services::SharedPtr<DistributedPartialResultStep4> getPartialResult() { return _partialResult; }

    /**
     * Returns a pointer to a newly allocated KD-tree based kNN training algorithm
     * with a copy of the input objects and parameters for this KD-tree based kNN training algorithm
     * in the distributed processing mode
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step4Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step4Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    services::SharedPtr<DistributedPartialResultStep4> _partialResult;

    virtual Distributed<step4Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step4Local, algorithmFPType, method>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE { return services::Status(); }

    services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult = services::SharedPtr<DistributedPartialResultStep4>(new DistributedPartialResultStep4());
        services::Status s = _partialResult->allocate<algorithmFPType>(_in, 0, 0);
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
        _ac  = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step4Local, algorithmFPType, method)(&_env);
        _par = &parameter;
        _in = &input;
        _partialResult = services::SharedPtr<DistributedPartialResultStep4>(new DistributedPartialResultStep4());
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTED"></a>
 * \brief Provides methods for KD-tree based kNN model-based training in the distributed processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for KD-tree based kNN model-based training, double or float
 * \tparam method           KD-tree based kNN training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref kdtree_knn_classification::interface1::Model "kdtree_knn_classification::Model" class
 *      - \ref prediction::interface1::Distributed "prediction::Distributed" class
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step5Local, algorithmFPType, method> : public Training<distributed>
{
public:
    DistributedInput<step5Local> input; /*!< %Input objects for the KD-tree based kNN model training algorithm in the distributed processing mode */
    Parameter parameter; /*!< \ref interface1::Parameter "Parameters" of the algorithm */

    /** Default constructor */
    Distributed()
    {
        initialize();
    }

    /**
     * Constructs a KD-tree based kNN training algorithm by copying input objects
     * and parameters of another KD-tree based kNN training algorithm in the distributed processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed & other)
    {
        initialize();
        input.set(dataForStep5, other.input.get(dataForStep5));
        input.set(labelsForStep5, other.input.get(labelsForStep5));
        input.set(dimensionForStep5, other.input.get(dimensionForStep5));
        input.set(isPartnerGreaterForStep5, other.input.get(isPartnerGreaterForStep5));
        input.set(histogramForStep5, other.input.get(histogramForStep5));
        parameter = other.parameter;
    }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Registers user-allocated memory to store the partial result of KD-tree based kNN model-based training
     * \param[in] partialResult    Structure to store the partial result of KD-tree based kNN model-based training
     */
    services::Status setPartialResult(const services::SharedPtr<DistributedPartialResultStep5> & partialResult)
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
    services::SharedPtr<DistributedPartialResultStep5> getPartialResult() { return _partialResult; }

    /**
     * Returns a pointer to a newly allocated KD-tree based kNN training algorithm
     * with a copy of the input objects and parameters for this KD-tree based kNN training algorithm
     * in the distributed processing mode
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step5Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step5Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    services::SharedPtr<DistributedPartialResultStep5> _partialResult;

    virtual Distributed<step5Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step5Local, algorithmFPType, method>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE { return services::Status(); }

    services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult = services::SharedPtr<DistributedPartialResultStep5>(new DistributedPartialResultStep5());
        services::Status s = _partialResult->allocate<algorithmFPType>(_in, 0, 0);
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
        _ac  = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step5Local, algorithmFPType, method)(&_env);
        _par = &parameter;
        _in = &input;
        _partialResult = services::SharedPtr<DistributedPartialResultStep5>(new DistributedPartialResultStep5());
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTED"></a>
 * \brief Provides methods for KD-tree based kNN model-based training in the distributed processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for KD-tree based kNN model-based training, double or float
 * \tparam method           KD-tree based kNN training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref kdtree_knn_classification::interface1::Model "kdtree_knn_classification::Model" class
 *      - \ref prediction::interface1::Distributed "prediction::Distributed" class
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step6Local, algorithmFPType, method> : public Training<distributed>
{
public:
    DistributedInput<step6Local> input; /*!< %Input objects for the KD-tree based kNN model training algorithm in the distributed processing mode */
    Parameter parameter; /*!< \ref interface1::Parameter "Parameters" of the algorithm */

    /** Default constructor */
    Distributed()
    {
        initialize();
    }

    /**
     * Constructs a KD-tree based kNN training algorithm by copying input objects
     * and parameters of another KD-tree based kNN training algorithm in the distributed processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed & other)
    {
        initialize();
        input.set(dataForStep6, other.input.get(dataForStep6));
        input.set(labelsForStep6, other.input.get(labelsForStep6));
        input.set(dataFromPartnerForStep6, other.input.get(dataFromPartnerForStep6));
        input.set(labelsFromPartnerForStep6, other.input.get(labelsFromPartnerForStep6));
        input.set(markersForStep6, other.input.get(markersForStep6));
        parameter = other.parameter;
    }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Registers user-allocated memory to store the partial result of KD-tree based kNN model-based training
     * \param[in] partialResult    Structure to store the partial result of KD-tree based kNN model-based training
     */
    services::Status setPartialResult(const services::SharedPtr<DistributedPartialResultStep6> & partialResult)
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
    services::SharedPtr<DistributedPartialResultStep6> getPartialResult() { return _partialResult; }

    /**
     * Returns a pointer to a newly allocated KD-tree based kNN training algorithm
     * with a copy of the input objects and parameters for this KD-tree based kNN training algorithm
     * in the distributed processing mode
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step6Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step6Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    services::SharedPtr<DistributedPartialResultStep6> _partialResult;

    virtual Distributed<step6Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step6Local, algorithmFPType, method>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE { return services::Status(); }

    services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult = services::SharedPtr<DistributedPartialResultStep6>(new DistributedPartialResultStep6());
        services::Status s = _partialResult->allocate<algorithmFPType>(_in, 0, 0);
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
        _ac  = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step6Local, algorithmFPType, method)(&_env);
        _par = &parameter;
        _in = &input;
        _partialResult = services::SharedPtr<DistributedPartialResultStep6>(new DistributedPartialResultStep6());
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTED"></a>
 * \brief Provides methods for KD-tree based kNN model-based training in the distributed processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for KD-tree based kNN model-based training, double or float
 * \tparam method           KD-tree based kNN training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref kdtree_knn_classification::interface1::Model "kdtree_knn_classification::Model" class
 *      - \ref prediction::interface1::Distributed "prediction::Distributed" class
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step7Master, algorithmFPType, method> : public Training<distributed>
{
public:
    DistributedInput<step7Master> input; /*!< %Input objects for the KD-tree based kNN model training algorithm in the distributed processing mode */
    Parameter parameter; /*!< \ref interface1::Parameter "Parameters" of the algorithm */

    /** Default constructor */
    Distributed()
    {
        initialize();
    }

    /**
     * Constructs a KD-tree based kNN training algorithm by copying input objects
     * and parameters of another KD-tree based kNN training algorithm in the distributed processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed & other)
    {
        initialize();
        input.set(boundingBoxesForStep7, other.input.get(boundingBoxesForStep7));
        input.set(numberOfLoopsForStep7, other.input.get(numberOfLoopsForStep7));
        input.set(loopNumberForStep7, other.input.get(loopNumberForStep7));
        input.set(partialModelForStep7, other.input.get(partialModelForStep7));
        input.set(dimensionForStep7, other.input.get(dimensionForStep7));
        input.set(medianForStep7, other.input.get(medianForStep7));
        parameter = other.parameter;
    }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Registers user-allocated memory to store the partial result of KD-tree based kNN model-based training
     * \param[in] partialResult    Structure to store the partial result of KD-tree based kNN model-based training
     */
    services::Status setPartialResult(const services::SharedPtr<DistributedPartialResultStep7> & partialResult)
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
    services::SharedPtr<DistributedPartialResultStep7> getPartialResult() { return _partialResult; }

    /**
     * Returns a pointer to a newly allocated KD-tree based kNN training algorithm
     * with a copy of the input objects and parameters for this KD-tree based kNN training algorithm
     * in the distributed processing mode
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step7Master, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step7Master, algorithmFPType, method> >(cloneImpl());
    }

protected:
    services::SharedPtr<DistributedPartialResultStep7> _partialResult;

    virtual Distributed<step7Master, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step7Master, algorithmFPType, method>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE { return services::Status(); }

    services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult = services::SharedPtr<DistributedPartialResultStep7>(new DistributedPartialResultStep7());
        services::Status s = _partialResult->allocate<algorithmFPType>(_in, 0, 0);
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
        _ac  = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step7Master, algorithmFPType, method)(&_env);
        _par = &parameter;
        _in = &input;
        _partialResult = services::SharedPtr<DistributedPartialResultStep7>(new DistributedPartialResultStep7());
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTED"></a>
 * \brief Provides methods for KD-tree based kNN model-based training in the distributed processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for KD-tree based kNN model-based training, double or float
 * \tparam method           KD-tree based kNN training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref kdtree_knn_classification::interface1::Model "kdtree_knn_classification::Model" class
 *      - \ref prediction::interface1::Distributed "prediction::Distributed" class
 */
template<typename algorithmFPType, Method method>
class DAAL_EXPORT Distributed<step8Local, algorithmFPType, method> : public Training<distributed>
{
public:
    DistributedInput<step8Local> input; /*!< %Input objects for the KD-tree based kNN model training algorithm in the distributed processing mode */
    Parameter parameter; /*!< \ref interface1::Parameter "Parameters" of the algorithm */

    /** Default constructor */
    Distributed()
    {
        initialize();
    }

    /**
     * Constructs a KD-tree based kNN training algorithm by copying input objects
     * and parameters of another KD-tree based kNN training algorithm in the distributed processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Distributed(const Distributed & other)
    {
        initialize();
        input.set(dataForStep8, other.input.get(dataForStep8));
        input.set(labelsForStep8, other.input.get(labelsForStep8));
        input.set(partialModelForStep8, other.input.get(partialModelForStep8));
        parameter = other.parameter;
    }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Registers user-allocated memory to store the partial result of KD-tree based kNN model-based training
     * \param[in] partialResult    Structure to store the partial result of KD-tree based kNN model-based training
     */
    services::Status setPartialResult(const services::SharedPtr<DistributedPartialResultStep8> & partialResult)
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
    services::SharedPtr<DistributedPartialResultStep8> getPartialResult() { return _partialResult; }

    /**
     * Returns a pointer to a newly allocated KD-tree based kNN training algorithm
     * with a copy of the input objects and parameters for this KD-tree based kNN training algorithm
     * in the distributed processing mode
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Distributed<step8Local, algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Distributed<step8Local, algorithmFPType, method> >(cloneImpl());
    }

protected:
    services::SharedPtr<DistributedPartialResultStep8> _partialResult;

    virtual Distributed<step8Local, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Distributed<step8Local, algorithmFPType, method>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE { return services::Status(); }

    services::Status allocatePartialResult() DAAL_C11_OVERRIDE
    {
        _partialResult = services::SharedPtr<DistributedPartialResultStep8>(new DistributedPartialResultStep8());
        services::Status s = _partialResult->allocate<algorithmFPType>(_in, 0, 0);
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
        _ac  = new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step8Local, algorithmFPType, method)(&_env);
        _par = &parameter;
        _in = &input;
        _partialResult = services::SharedPtr<DistributedPartialResultStep8>(new DistributedPartialResultStep8());
    }
};

/** @} */
} // namespace interface1

using interface1::DistributedContainer;
using interface1::Distributed;

} // namespace training
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
