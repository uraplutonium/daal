/* file: lbfgs_types.h */
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
//  Implementation of limited memory Broyden-Fletcher-Goldfarb-Shanno
//  algorithm types.
//--
*/
#ifndef __LBFGS_TYPES_H__
#define __LBFGS_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/optimization_solver/iterative_solver/iterative_solver_batch.h"
#include "algorithms/optimization_solver/objective_function/sum_of_functions_batch.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
/**
 * \brief Contains classes for computing the limited memory Broyden-Fletcher-Goldfarb-Shanno algorithm
 */
namespace lbfgs
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__LBFGS__METHOD"></a>
 * Available methods for computing LBFGS
 */
enum Method
{
    defaultDense = 0       /*!< Default: performance-oriented method. */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__LBFGS__PARAMETER"></a>
 * \brief %Parameter class for LBFGS algorithm
 *
 * \snippet optimization_solver/lbfgs/lbfgs_types.h Parameter source code
 */
/* [Parameter source code] */
struct Parameter : public optimization_solver::iterative_solver::Parameter
{
    /**
     * Constructs the parameters of LBFGS algorithm
     * \param[in] function                  Objective function that can be represented as sum
     * \param[in] nIterations               Maximal number of iterations of the algorithm
     * \param[in] accuracyThreshold         Accuracy of the LBFGS algorithm
     * \param[in] batchSize                 Number of observations to compute the stochastic gradient
     * \param[in] correctionPairBatchSize   The number of observations to compute the sub-sampled Hessian for correction pairs computation
     * \param[in] m                         Memory parameter of LBFGS
     * \param[in] L                         The number of iterations between the curvature estimates calculations
     * \param[in] seed                      Seed for random choosing terms from objective function
     */
    Parameter(services::SharedPtr<sum_of_functions::Batch> function = services::SharedPtr<sum_of_functions::Batch>(),
              size_t nIterations = 100, double accuracyThreshold = 1.0e-5,
              size_t batchSize = 10, size_t correctionPairBatchSize = 100,
              size_t m = 10, size_t L = 10, size_t seed = 777) :
        optimization_solver::iterative_solver::Parameter(function, nIterations, accuracyThreshold),
        batchSize(batchSize), correctionPairBatchSize(correctionPairBatchSize), m(m), L(L), seed(seed),
        stepLengthSequence(new data_management::HomogenNumericTable<>(1, 1, data_management::NumericTableIface::doAllocate, 1.0))
    {}

    virtual ~Parameter() {}

    size_t m;                       /*!< Memory parameter of LBFGS.
                                         The maximum number of correction pairs that define the approximation
                                         of inverse Hessian matrix. */
    size_t L;                       /*!< The number of iterations between the curvature estimates calculations */
    size_t seed;                    /*!< Seed for random choosing terms from objective function. */

    size_t batchSize;               /*!< Number of observations to compute the stochastic gradient. */
    /** Numeric table of size nIterations x batchSize that represent indices that will be used instead of random values
        for the stochastic gradient computations. If not set then random indices will be chosen. */
    services::SharedPtr<data_management::NumericTable> batchIndices;

    size_t correctionPairBatchSize; /*!< Number of observations to compute the sub-sampled Hessian for correction pairs computation */
    /** Numeric table of size (nIterations / L) x correctionPairBatchSize that represent indices that will be used
        instead of random values for the sub-sampled Hessian matrix computations. If not set then random indices will be chosen. */
    services::SharedPtr<data_management::NumericTable> correctionPairBatchIndices;
    /** Numeric table of size:
            - 1 x nIterations that contains values of the step-length sequence a(k), for k = 1, ..., nIterations, or
            - 1 x 1           that contains value of step length at each iteration a(1) = ... = a(nIterations) */
    services::SharedPtr<data_management::NumericTable> stepLengthSequence;

    /**
    * Checks the correctness of the parameter
    */
    virtual void check() const
    {
        iterative_solver::Parameter::check();

        if(m == 0)
        {
            services::SharedPtr<services::Error> error(new services::Error());
            error->setId(services::ErrorIncorrectParameter);
            error->addStringDetail(services::ArgumentName, "m");
            this->_errors->add(error);
            return;
        }

        if(L == 0)
        {
            services::SharedPtr<services::Error> error(new services::Error());
            error->setId(services::ErrorIncorrectParameter);
            error->addStringDetail(services::ArgumentName, "L");
            this->_errors->add(error);
            return;
        }

        if(batchSize == 0)
        {
            services::SharedPtr<services::Error> error(new services::Error());
            error->setId(services::ErrorIncorrectParameter);
            error->addStringDetail(services::ArgumentName, "batchSize");
            this->_errors->add(error);
            return;
        }

        if(batchIndices.get() != NULL)
        {
            services::SharedPtr<services::Error> error(new services::Error());
            if(batchIndices->getNumberOfRows() != nIterations) { error->setId(services::ErrorIncorrectNumberOfObservations); }
            if(batchIndices->getNumberOfColumns() != batchSize) { error->setId(services::ErrorIncorrectNumberOfFeatures); }
            if(error->id() != services::NoErrorMessageFound)
            {
                error->addStringDetail(services::ArgumentName, "batchIndices");
                this->_errors->add(error);
                return;
            }
        }

        if(correctionPairBatchIndices.get() != NULL)
        {
            services::SharedPtr<services::Error> error(new services::Error());
            if(correctionPairBatchIndices->getNumberOfRows() != (nIterations / L)) { error->setId(services::ErrorIncorrectNumberOfObservations); }
            if(correctionPairBatchIndices->getNumberOfColumns() != correctionPairBatchSize) { error->setId(services::ErrorIncorrectNumberOfFeatures); }
            if(error->id() != services::NoErrorMessageFound)
            {
                error->addStringDetail(services::ArgumentName, "correctionPairBatchIndices");
                this->_errors->add(error);
                return;
            }
        }

        if(stepLengthSequence.get() != NULL)
        {
            services::SharedPtr<services::Error> error(new services::Error());
            if(stepLengthSequence->getNumberOfRows() != 1) { error->setId(services::ErrorIncorrectNumberOfObservations); }
            if(stepLengthSequence->getNumberOfColumns() != 1 && stepLengthSequence->getNumberOfColumns() != nIterations)
            {
                error->setId(services::ErrorIncorrectNumberOfFeatures);
            }
            if(error->id() != services::NoErrorMessageFound)
            {
                error->addStringDetail(services::ArgumentName, "stepLengthSequence");
                this->_errors->add(error);
                return;
            }
        }
    }
};
/* [Parameter source code] */

} // namespace interface1
using interface1::Parameter;

} // namespace lbfgs
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif
