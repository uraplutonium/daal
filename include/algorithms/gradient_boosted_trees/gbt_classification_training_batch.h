/* file: gbt_classification_training_batch.h */
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
//  Implementation of the interface for Gradient Boosted Trees model-based training
//--
*/

#ifndef __GBT_CLASSIFICATION_TRAINING_BATCH_H__
#define __GBT_CLASSIFICATION_TRAINING_BATCH_H__

#include "algorithms/classifier/classifier_training_batch.h"
#include "algorithms/gradient_boosted_trees/gbt_classification_training_types.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace classification
{
namespace training
{
namespace interface1
{
/**
 * @defgroup gbt_classification_training_batch Batch
 * @ingroup gbt_classification_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__CLASSIFICATION__TRAINING__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of Gradient Boosted Trees model-based training.
 *        This class is associated with daal::algorithms::gbt::classification::training::Batch class
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations, double or float
 * \tparam method           Gradient Boosted Trees model training method, \ref Method
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for Gradient Boosted Trees model-based training with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of Gradient Boosted Trees model-based training in the batch processing mode
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__CLASSIFICATION__TRAINING__BATCH"></a>
 * \brief Trains model of the Gradient Boosted Trees algorithms in the batch processing mode
 * <!-- \n<a href="DAAL-REF-GBT__CLASSIFICATION-ALGORITHM">Gradient Boosted Trees algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for Gradient Boosted Trees, double or float
 * \tparam method           Gradient Boosted Trees computation method, \ref daal::algorithms::gbt::classification::training::Method
 *
 * \par Enumerations
 *      - \ref Method                         Gradient Boosted Trees training methods
 *      - \ref classifier::training::InputId  Identifiers of input objects for the Gradient Boosted Trees training algorithm
 *      - \ref classifier::training::ResultId Identifiers of Gradient Boosted Trees training results
 *
 * \par References
 *      - \ref gbt::classification::interface1::Model "Model" class
 *      - \ref classifier::training::interface1::Input "classifier::training::Input" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public classifier::training::Batch
{
public:
    classifier::training::Input input;  /*!< %Input data structure */

    /**
     * Constructs the Gradient Boosted Trees training algorithm
     * \param[in] nClasses  Number of classes
     */
    Batch(size_t nClasses)
    {
        _par = new Parameter(nClasses);
        initialize();
    }

    /**
     * Constructs a Gradient Boosted Trees training algorithm by copying input objects and parameters
     * of another Gradient Boosted Trees training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) : classifier::training::Batch(other), input(other.input)
    {
        _par = new Parameter(other.parameter());
        initialize();
    }

    /** Destructor */
    ~Batch()
    {
        delete _par;
    }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    Parameter& parameter() { return *static_cast<Parameter*>(_par); }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    const Parameter& parameter() const { return *static_cast<const Parameter*>(_par); }

    /**
     * Get input objects for the Gradient Boosted Trees training algorithm
     * \return %Input objects for the Gradient Boosted Trees training algorithm
     */
    classifier::training::Input * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Returns the structure that contains results of Gradient Boosted Trees training
     * \return Structure that contains results of Gradient Boosted Trees training
     */
    ResultPtr getResult()
    {
        return Result::cast(_result);
    }

    /**
     * Resets the training results of the classification algorithm
     */
    services::Status resetResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new Result());
        DAAL_CHECK(_result, services::ErrorNullResult);
        _res = NULL;
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated Gradient Boosted Trees training algorithm with a copy of input objects
     * and parameters of this Gradient Boosted Trees training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    virtual services::Status checkComputeParams() DAAL_C11_OVERRIDE;

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        ResultPtr res = getResult();
        DAAL_CHECK(res, services::ErrorNullResult);
        services::Status s = res->template allocate<algorithmFPType>(&input, &parameter(), method);
        _res = _result.get();
        return s;
    }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in  = &input;
        _result.reset(new Result());
    }
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace daal::algorithms::gbt::classification::training
}
}
}
} // namespace daal
#endif // __LOGIT_BOOST_TRAINING_BATCH_H__