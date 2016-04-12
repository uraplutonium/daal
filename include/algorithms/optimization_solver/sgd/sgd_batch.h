/* file: sgd_batch.h */
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

/*
//++
//  Implementation of Stochastic gradient descent algorithm interface.
//--
*/

#ifndef __SGD_BATCH_H__
#define __SGD_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/optimization_solver/optimization_solver_batch.h"
#include "sgd_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sgd
{
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the Stochastic gradient descent algorithm.
 *        This class is associated with daal::algorithms::optimization_solver::sgd::BatchContainer class.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the Stochastic gradient descent algorithm, double or float
 * \tparam method           Stochastic gradient descent computation method, daal::algorithms::optimization_solver::sgd::Method
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    BatchContainer(daal::services::Environment::env *daalEnv);
    ~BatchContainer();
    /**
     * Runs implementation of the Stochastic gradient descent algorithm in the batch processing mode
     */
    virtual void compute();
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__BATCH"></a>
 * \brief Computes Stochastic gradient descent in the batch processing mode.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the Stochastic gradient descent algorithm,
 *                          double or float
 * \tparam method           Stochastic gradient descent computation method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for Stochastic gradient descent
 *      - \ref InputId  Identifiers of input objects for Stochastic gradient descent
 *      - \ref ResultId Result identifiers for the Stochastic gradient descent
 *
 * \par References
 *      - <a href="DAAL-REF-SGD-ALGORITHM">Stochastic gradient descent algorithm description and usage models</a>
 *      - Input class
 *      - Result class
 */
template<typename algorithmFPType = double, Method method = defaultDense>
class DAAL_EXPORT Batch : public optimization_solver::Batch
{
public:
    /** Default constructor */
    Batch(services::SharedPtr<sum_of_functions::Batch> objectiveFunction) : optimization_solver::Batch(), parameter(objectiveFunction)
    {
        initialize();
    }

    /**
     * Constructs a Stochastic gradient descent algorithm by copying input objects
     * of another Stochastic gradient descent algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) : optimization_solver::Batch(), parameter(other.parameter)
    {
        initialize();
        input.set(inputArgument, other.input.get(inputArgument));
    }

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains results of the Stochastic gradient descent algorithm
     * \return Structure that contains results of the Stochastic gradient descent algorithm
     */
    services::SharedPtr<Result> getResult()
    {
        return services::staticPointerCast<Result, Result>(_result);
    }

    /*
     * Registers user-allocated memory to store results of the Stochastic gradient descent algorithm
     * \param[in] result  Structure to store  results of the Stochastic gradient descent algorithm
     */
    void setResult(services::SharedPtr<Result> result)
    {
        _result = result;
        _res = _result.get();
    }

    /**
     * Returns a pointer to the newly allocated Stochastic gradient descent algorithm with a copy of input objects
     * of this Stochastic gradient descent algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Batch<algorithmFPType, method> *cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    virtual void allocateResult()
    {
        _result->allocate<algorithmFPType>(&input, &parameter, (int) method);
        _res = _result.get();
    }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _par = &parameter;
        _in  = &input;
        _result = services::SharedPtr<Result>(new Result());
    }

    services::SharedPtr<Result> _result;

public:
    Input input; /*!< %Input data structure */
    Parameter<method> parameter; /*!< %Parameter data structure */
};
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace sgd
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif
