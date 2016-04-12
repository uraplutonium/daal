/* file: objective_function_constructors.cpp */
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
//  Implementation of Objective Function Batch Parameter constructor
//--
*/

#include "services/daal_defines.h"
#include "objective_function_types.h"
#include "sum_of_functions_types.h"
#include "sum_of_loss_types.h"
#include "cross_entropy/cross_entropy_types.h"
#include "cross_entropy/cross_entropy_batch.h"
#include "precomputed/precomputed_types.h"
#include "precomputed/precomputed_batch.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for the Objective function
 */
namespace optimization_solver
{
namespace objective_function
{
namespace interface1
{

Parameter::Parameter(const DAAL_UINT64 resultsToCompute) : resultsToCompute(resultsToCompute) {}

Parameter::Parameter(const Parameter &other) : resultsToCompute(other.resultsToCompute) {}

}
}

namespace sum_of_functions
{
namespace interface1
{

Parameter::Parameter(size_t numberOfTerms, services::SharedPtr<data_management::NumericTable> batchIndices,
                     const DAAL_UINT64 resultsToCompute) :
    numberOfTerms(numberOfTerms),
    objective_function::Parameter(resultsToCompute),
    batchIndices(batchIndices) {};

Parameter::Parameter(const Parameter &other) :
    numberOfTerms(other.numberOfTerms),
    objective_function::Parameter(other.resultsToCompute),
    batchIndices(other.batchIndices) {};

}
}

namespace internal
{
namespace sum_of_loss
{
Parameter::Parameter(size_t numberOfTerms, services::SharedPtr<data_management::NumericTable> batchIndices,
                     const DAAL_UINT64 resultsToCompute) :
    sum_of_functions::Parameter(numberOfTerms, batchIndices, resultsToCompute) {}

Parameter::Parameter(const Parameter &other) : sum_of_functions::Parameter(other) {}

}
}

namespace internal
{
namespace cross_entropy
{

Parameter::Parameter(size_t numberOfTerms, services::SharedPtr<data_management::NumericTable> batchIndices,
                     const DAAL_UINT64 resultsToCompute) :
    sum_of_loss::Parameter(numberOfTerms, batchIndices, resultsToCompute) {}

Parameter::Parameter(const Parameter &other) : sum_of_loss::Parameter(other) {}

namespace interface1
{

template<typename algorithmFPType, Method method>
Batch<algorithmFPType, method>::Batch() : parameter(1), sum_of_loss::Batch(1, &input, &parameter)
{
    initialize();
}

template class Batch<double, defaultDense>;
template class Batch<float, defaultDense>;
}
}

namespace precomputed
{
namespace interface1
{

Parameter::Parameter(size_t numberOfTerms, services::SharedPtr<data_management::NumericTable> batchIndices,
                     const DAAL_UINT64 resultsToCompute) :
    sum_of_functions::Parameter(numberOfTerms, batchIndices, resultsToCompute) {};

Parameter::Parameter(const Parameter &other) : sum_of_functions::Parameter(other) {}

template<typename algorithmFPType, Method method>
Batch<algorithmFPType, method>::Batch() : parameter(1), sum_of_functions::Batch(1, &input, &parameter)
{
    initialize();
}

template class Batch<double, defaultDense>;
template class Batch<float, defaultDense>;
}
}
}


}
}
}
