/* file: sum_of_loss_batch.h */
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
//  Implementation of the Mean squared error objective function types.
//--
*/

#ifndef __SUM_OF_LOSS_BATCH_H__
#define __SUM_OF_LOSS_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "sum_of_functions_batch.h"
#include "sum_of_loss_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace internal
{
namespace sum_of_loss
{

namespace interface1
{

class DAAL_EXPORT Batch : public sum_of_functions::Batch
{
public:
    Batch(size_t numberOfTerms, sum_of_loss::Input *sumOfFunctionsInput, sum_of_loss::Parameter *sumOfFunctionsParameter) :
        sum_of_functions::Batch(numberOfTerms, sumOfFunctionsInput, sumOfFunctionsParameter)
    {
        initialize();
    }

    virtual ~Batch() {}

    Batch(const Batch &other) :
        sum_of_functions::Batch(other.sumOfFunctionsParameter->numberOfTerms, other.sumOfFunctionsInput, other.sumOfFunctionsParameter)
    {
        initialize();
    }

    virtual Input *getInput() = 0;

    void allocate()
    {
        allocateResult();
    }

protected:
    void initialize()
    {}
};
} // namespace interface1
using interface1::Batch;

} // namespace sum_of_loss
} // namespace internal
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif
