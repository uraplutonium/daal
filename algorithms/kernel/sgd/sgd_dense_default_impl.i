/* file: sgd_dense_default_impl.i */
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
//  Implementation of sgd algorithm
//--
*/

#ifndef __SGD_DENSE_DEFAULT_IMPL_I__
#define __SGD_DENSE_DEFAULT_IMPL_I__

#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "service_math.h"
#include "service_utils.h"
#include "service_numeric_table.h"

using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sgd
{
namespace internal
{
/**
 *  \brief Kernel for SGD calculation
 */
template<typename algorithmFPType, CpuType cpu>
void SGDKernel<algorithmFPType, defaultDense, cpu>::compute(Input *input, Result *result, Parameter<defaultDense> *parameter)
{
    size_t nFeatures = input->get(inputArgument)->getNumberOfColumns();
    BlockMicroTable<algorithmFPType, writeOnly, cpu> mtWorkValue(result->get(minimum).get());
    algorithmFPType *workValue;
    mtWorkValue.getBlockOfRows(0, 1, &workValue);
    BlockMicroTable<algorithmFPType, readOnly, cpu> mtStartValue(input->get(inputArgument).get());
    algorithmFPType *startValueArray;
    mtStartValue.getBlockOfRows(0, 1, &startValueArray);
    daal_memcpy_s(workValue, nFeatures * sizeof(algorithmFPType), startValueArray, nFeatures * sizeof(algorithmFPType));
    mtStartValue.release();

    size_t maxIterations = parameter->nIterations;
    /* if maxIterations == 0, set result as start point, the number of executed iters to 0 */
    if(maxIterations == 0)
    {
        BlockMicroTable<int, writeOnly, cpu> mtNIterations(result->get(nIterations).get());
        int *nProceededIterations = NULL;
        mtNIterations.getBlockOfRows(0, 1, &nProceededIterations);
        nProceededIterations[0] = 0;
        mtNIterations.release();
        mtWorkValue.release();
        return;
    }

    /*Het random indices for SGD from parameter or from rng generator*/
    bool isPredefinedbatchIndices = (parameter->batchIndices.get() != NULL) ? true : false;

    SharedPtr<sum_of_functions::Batch> function = parameter->function;
    BlockMicroTable<int, readOnly, cpu> mtPredefinedBatchIndices((parameter->batchIndices).get());
    int *randomTerm = NULL;
    if(isPredefinedbatchIndices)
    {
        size_t nReadRows = mtPredefinedBatchIndices.getBlockOfRows(0, maxIterations, &randomTerm);
        if(nReadRows != maxIterations)
        {
            mtPredefinedBatchIndices.release();
            this->_errors->add(ErrorMemoryAllocationFailed);
            return;
        }
    }
    else
    {
        randomTerm = (int *) daal_malloc(maxIterations * sizeof(int));
        if(!randomTerm) { this->_errors->add(ErrorMemoryAllocationFailed); return; }
        getRandom(0, function->sumOfFunctionsParameter->numberOfTerms, randomTerm, maxIterations, parameter->seed);
    }
    SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu>> ntBatchIndices =
                SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu>>(
                    new daal::internal::HomogenNumericTableCPU<int, cpu>(NULL, 1, 1));

    SharedPtr<NumericTable> previousBatchIndices = function->sumOfFunctionsParameter->batchIndices;
    function->sumOfFunctionsParameter->batchIndices = ntBatchIndices;

    SharedPtr<NumericTable> previousSumOfFunctionsInputNT = function->sumOfFunctionsInput->get(sum_of_functions::argument);
    function->sumOfFunctionsInput->set(sum_of_functions::argument, SharedPtr<NumericTable>(
                                           SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>>(
                                                   new daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>(workValue, nFeatures, 1))
                                       ));

    BlockMicroTable<algorithmFPType, readOnly, cpu> mtLearningRate(parameter->learningRateSequence.get());
    algorithmFPType *learningRateArray;
    mtLearningRate.getBlockOfRows(0, 1, &learningRateArray);
    size_t learningRateLength = mtLearningRate.getFullNumberOfColumns();
    double accuracyThreshold = parameter->accuracyThreshold;
    algorithmFPType pointNorm = 0, gradientNorm = 0, gradientThreshold, one = 1.0;

    BlockMicroTable<int, writeOnly, cpu> mtNIterations(result->get(nIterations).get());
    int *nProceededIterations = NULL;
    mtNIterations.getBlockOfRows(0, 1, &nProceededIterations);
    *nProceededIterations = (int)maxIterations;
    BlockDescriptor<algorithmFPType> gradientBlock;
    SharedPtr<NumericTable> ntGradient;
    algorithmFPType learningRate;
    for(size_t epoch = 0; epoch < maxIterations; epoch++)
    {
        ntBatchIndices->setArray(&(randomTerm[epoch]));
        function->compute();
        if(function->getErrors()->size() != 0)
        {
            if(isPredefinedbatchIndices)
            {
                mtPredefinedBatchIndices.release();
            }
            else
            {
                daal_free(randomTerm);
            }
            *nProceededIterations = (int)epoch;
            function->sumOfFunctionsParameter->batchIndices = previousBatchIndices;
            function->sumOfFunctionsInput->set(sum_of_functions::argument, previousSumOfFunctionsInputNT);
            mtWorkValue.release();
            return;
        }

        ntGradient = function->getResult()->get(objective_function::resultCollection, objective_function::gradientIdx);
        ntGradient->getBlockOfRows(0, 1, readOnly, gradientBlock);

        algorithmFPType *gradient = gradientBlock.getBlockPtr();
        pointNorm    = vectorNorm(workValue, nFeatures);
        gradientNorm = vectorNorm(gradient, nFeatures);
        gradientThreshold = accuracyThreshold * daal::sMax<algorithmFPType, cpu>(one, pointNorm);
        if(gradientNorm <= gradientThreshold)
        {
            ntGradient->releaseBlockOfRows(gradientBlock);
            *nProceededIterations = (int)epoch;
            break;
        }

        (learningRateLength > 1) ? learningRate = learningRateArray[epoch] : learningRate = learningRateArray[0];
        for(size_t j = 0; j < nFeatures; j++)
        {
            workValue[j] = workValue[j] - learningRate * gradient[j];
        }
        ntGradient->releaseBlockOfRows(gradientBlock);
    }
    mtNIterations.release();
    mtLearningRate.release();
    function->sumOfFunctionsParameter->batchIndices = previousBatchIndices;
    function->sumOfFunctionsInput->set(sum_of_functions::argument, previousSumOfFunctionsInputNT);
    mtWorkValue.release();
    if(isPredefinedbatchIndices)
    {
        mtPredefinedBatchIndices.release();
    }
    else
    {
        daal_free(randomTerm);
    }

    return;
}

} // namespace daal::internal

} // namespace sgd

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
