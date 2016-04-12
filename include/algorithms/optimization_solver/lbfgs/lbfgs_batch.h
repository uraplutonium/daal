/* file: lbfgs_batch.h */
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
//  Implementation of limited memory Broyden-Fletcher-Goldfarb-Shanno algorithm interface.
//--
*/

#ifndef __LBFGS_BATCH_H__
#define __LBFGS_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/optimization_solver/optimization_solver_batch.h"
#include "lbfgs_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace lbfgs
{
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__LBFGS__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the LBFGS algorithm.
 *        This class is associated with daal::algorithms::optimization_solver::lbfgs::Batch class.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the LBFGS algorithm, double or float
 * \tparam method           Stochastic gradient descent computation method, daal::algorithms::optimization_solver::lbfgs::Method
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    BatchContainer(daal::services::Environment::env *daalEnv);
    ~BatchContainer();
    /**
     * Runs implementation of the LBFGS algorithm in the batch processing mode
     */
    virtual void compute();
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__LBFGS__BATCH"></a>
 * \brief Computes LBFGS in the batch processing mode.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the LBFGS algorithm,
 *                          double or float
 * \tparam method           LBFGS computation method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for LBFGS
 *      - \ref InputId  Identifiers of input objects for LBFGS
 *      - \ref ResultId optimization_solver::Result identifiers for the LBFGS
 *
 * \par References
 *      - <a href="DAAL-REF-LBFGS-ALGORITHM">Limited memory BFGS algorithm description and usage models</a>
 *      - Input class
 *      - Result class
 */
template<typename algorithmFPType = double, Method method = defaultDense>
class DAAL_EXPORT Batch : public optimization_solver::Batch
{
public:
    Input input;           /*!< %Input data structure */
    Parameter parameter;   /*!< %Parameters of the algorithm */

    /**
     * Constructs the LBFGS algorithm with the input objective function
     * \param[in] objectiveFunction Objective function that can be represented as a sum of functions
     */
    Batch(services::SharedPtr<sum_of_functions::Batch> objectiveFunction) :
        optimization_solver::Batch(), parameter(objectiveFunction)
    {
        initialize();
    }

    /**
     * Constructs an LBFGS algorithm by copying input objects of another LBFGS algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) : parameter(other.parameter)
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
     * Returns the structure that contains results of the LBFGS algorithm
     * \return Structure that contains results of the LBFGS algorithm
     */
    services::SharedPtr<Result> getResult()
    {
        return services::staticPointerCast<Result, Result>(_result);
    }

    /*
     * Registers user-allocated memory to store results of the LBFGS algorithm
     * \param[in] result  Structure to store  results of the LBFGS algorithm
     */
    void setResult(services::SharedPtr<Result> result)
    {
        _result = result;
        _res = _result.get();
    }

    /**
     * Returns a pointer to the newly allocated LBFGS algorithm with a copy of input objects
     * of this LBFGS algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
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
};

} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace lbfgs
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal

#endif
