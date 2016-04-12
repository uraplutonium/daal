/* file: cross_entropy_dense_default_batch_impl.i */
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
//  Implementation of cross_entropy algorithm
//--
*/

#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "service_data_utils.h"
#include "service_math.h"

#include "cross_entropy_batch.h"
#include "../precomputed/precomputed_batch.h"
#include "../sum_of_loss_types.h"

using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace internal
{
namespace cross_entropy
{

/**
 *  \brief Kernel for cross_entropy objective function calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
inline void CrossEntropyKernel<algorithmFPType, method, cpu>::compute(
    Input *input,
    objective_function::Result *result,
    Parameter *parameter)
{
    SharedPtr<NumericTable> probabilitiesTable = input->get(sum_of_loss::probabilities);
    SharedPtr<NumericTable> groundTruthTable = input->get(sum_of_loss::groundTruth);

    size_t nRows = probabilitiesTable->getNumberOfRows();
    size_t nFeatures = probabilitiesTable->getNumberOfColumns();

    BlockMicroTable<algorithmFPType, readOnly, cpu> probabilitiesMt( probabilitiesTable.get() );
    algorithmFPType *probabilitiesArray;
    probabilitiesMt.getBlockOfRows( 0, nRows, &probabilitiesArray );

    BlockMicroTable<int, readOnly, cpu> groundTruthMt( groundTruthTable.get() );
    int *groundTruthArray;
    groundTruthMt.getBlockOfRows( 0, nRows, &groundTruthArray );

    algorithmFPType minValFpType = daal::data_feature_utils::internal::MinVal<algorithmFPType, cpu>::get();

    if(parameter->resultsToCompute & objective_function::value)
    {
        algorithmFPType crossEntropyValue = 0;
        for(size_t i = 0; i < nRows; i++)
        {
            crossEntropyValue -= sLog<cpu>(sMax<algorithmFPType, cpu>(probabilitiesArray[i * nFeatures + groundTruthArray[i]], minValFpType));
        }

        SharedPtr<NumericTable> valueTable = result->get(objective_function::resultCollection, objective_function::valueIdx);
        BlockMicroTable<algorithmFPType, writeOnly, cpu> valueMt( valueTable.get() );
        algorithmFPType *value;
        valueMt.getBlockOfRows( 0, 1, &value );
        value[0] = crossEntropyValue / nRows;
        valueMt.release();
    }
    if(parameter->resultsToCompute & objective_function::gradient)
    {
        SharedPtr<NumericTable> gradientTable = result->get(objective_function::resultCollection, objective_function::gradientIdx);

        BlockMicroTable<algorithmFPType, writeOnly, cpu> gradientMt( gradientTable.get() );
        algorithmFPType *gradient;
        gradientMt.getBlockOfRows( 0, nRows, &gradient );

        algorithmFPType zero = 0.0;
        algorithmFPType invMinusN = -1.0 / nRows;
        for(size_t i = 0; i < nRows; i++)
        {
            for(size_t j = 0; j < nFeatures; j++)
            {
                gradient[i * nFeatures + j] = 0.0;
                if(j == groundTruthArray[i])
                {
                    gradient[i * nFeatures + j] = invMinusN / sMax<algorithmFPType, cpu>(probabilitiesArray[i * nFeatures + j], minValFpType);
                }
            }
        }
        gradientMt.release();
    }
    groundTruthMt.release();
    probabilitiesMt.release();
}

} // namespace cross_entropy
} // namespace internal
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
