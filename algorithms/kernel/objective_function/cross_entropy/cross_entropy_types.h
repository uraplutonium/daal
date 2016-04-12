/* file: cross_entropy_types.h */
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
//  Implementation of cross entropy objective function interface.
//--
*/

#ifndef __CROSS_ENTROPY_TYPES_H__
#define __CROSS_ENTROPY_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "../sum_of_loss_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for computing the cross entropy objective function
 */
namespace optimization_solver
{
namespace internal
{
namespace cross_entropy
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__CROSS_ENTROPY__METHOD"></a>
 * Available methods for computing results of cross entropy objective function
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method. */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__CROSS_ENTROPY__PARAMETER"></a>
 * \brief %Parameter for cross entropy objective function
 *
 * \snippet optimization_solver/objective_function/cross_entropy_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public sum_of_loss::Parameter
{
    /**
     * Constructs the parameter of cross entropy objective function
     * \param[in] numberOfTerms    The number of terms in the function
     * \param[in] batchIndices     Numeric table of size 1 x m where m is batch size that represent
                                   a batch of indices used to compute the function results, e.g.,
                                   value of the sum of the functions. If no indices are provided,
                                   all terms will be used in the computations.
     * \param[in] resultsToCompute 64 bit integer flag that indicates the results to compute
     */
    Parameter(size_t numberOfTerms,
              services::SharedPtr<data_management::NumericTable> batchIndices = services::SharedPtr<data_management::NumericTable>(),
              const DAAL_UINT64 resultsToCompute = objective_function::gradient);

    /**
     * Constructs an Parameter by copying input objects and parameters of another Parameter
     * \param[in] other An object to be used as the source to initialize object
     */
    Parameter(const Parameter &other);

    /**
     * Checks the correctness of the parameter
     */
    virtual void check() const
    {
        sum_of_loss::Parameter::check();
    }

    virtual ~Parameter() {}
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__CROSS_ENTROPY__INPUT"></a>
 * \brief %Input objects for the cross entropy objective function
 */
class Input : public sum_of_loss::Input
{
public:
    /** Default constructor */
    Input() : sum_of_loss::Input(2)
    {}

    /** Destructor */
    virtual ~Input() {}

    /**
     * Checks the correctness of the input
     * \param[in] par       Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        sum_of_loss::Input::check(par, method);
    }
};

} // namespace interface1
using interface1::Parameter;
using interface1::Input;

} // namespace cross_entropy
} // namespace internal
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif
