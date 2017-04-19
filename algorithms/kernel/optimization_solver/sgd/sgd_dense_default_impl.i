/* file: sgd_dense_default_impl.i */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
#include "iterative_solver_kernel.h"
#include "threading.h"

using namespace daal::internal;
using namespace daal::services;
using namespace daal::algorithms::optimization_solver::iterative_solver::internal;


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
services::Status SGDKernel<algorithmFPType, defaultDense, cpu>::compute(NumericTable *inputArgument, NumericTable *minimum, NumericTable *nIterations,
        Parameter<defaultDense> *parameter, NumericTable *learningRateSequence,
        NumericTable *batchIndices, OptionalArgument *optionalArgument, OptionalArgument *optionalResult)
{
    const size_t nRows = inputArgument->getNumberOfRows();

    //init workValue
    {
        processByBlocks<cpu>(minimum->getNumberOfRows(), this->_errors.get(),  [ = ](size_t startOffset, size_t nRowsInBlock)
        {
            WriteRows<algorithmFPType, cpu, NumericTable> workValueBD(*minimum, startOffset, nRowsInBlock);
            algorithmFPType *minArray = workValueBD.get();
            ReadRows<algorithmFPType, cpu, NumericTable> startValueBD(*inputArgument, startOffset, nRowsInBlock);
            const algorithmFPType *startValueArray = startValueBD.get();
            if( minArray != startValueArray )
            {
                daal_memcpy_s(minArray, nRowsInBlock * sizeof(algorithmFPType), startValueArray, nRowsInBlock * sizeof(algorithmFPType));
            }
        });
    }

    const size_t nIter = parameter->nIterations;
    /* if nIter == 0, set result as start point, the number of executed iters to 0 */
    if(nIter == 0)
    {
        WriteRows<int, cpu, NumericTable> nIterationsBD(*nIterations, 0, 1);
        int *nProceededIterations = nIterationsBD.get();
        nProceededIterations[0] = 0;
        DAAL_RETURN_STATUS()
    }

    NumericTable *lastIterationInput = (optionalArgument) ? NumericTable::cast(optionalArgument->get(iterative_solver::lastIteration)).get() : nullptr;
    NumericTable *lastIterationResult = (optionalResult) ? NumericTable::cast(optionalResult->get(iterative_solver::lastIteration)).get() : nullptr;

    sum_of_functions::BatchPtr function = parameter->function;

    const size_t nTerms = function->sumOfFunctionsParameter->numberOfTerms;
    ReadRows<int, cpu, NumericTable> predefinedBatchIndicesBD(batchIndices, 0, nIter);
    const bool bGenerateAllIndices = !(parameter->batchIndices || parameter->optionalResultRequired);
    TArray<int, cpu> aPredefinedBatchIndices(bGenerateAllIndices ? nIter : 0);
    if(bGenerateAllIndices)
    {
        /*Get random indices for SGD from rng generator*/
        DAAL_CHECK(aPredefinedBatchIndices.get(), ErrorMemoryAllocationFailed);
        getRandom(0, nTerms, aPredefinedBatchIndices.get(), nIter, parameter->seed);
    }
    using namespace iterative_solver::internal;
    const int *predefinedBatchIndices = predefinedBatchIndicesBD.get() ? predefinedBatchIndicesBD.get() : aPredefinedBatchIndices.get();
    RngTask<int, cpu> rngTask(predefinedBatchIndices, 1);
    DAAL_CHECK((predefinedBatchIndices) || rngTask.init(optionalArgument, nTerms, parameter->seed, sgd::rngState), ErrorMemoryAllocationFailed);

    SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu>> ntBatchIndices(new daal::internal::HomogenNumericTableCPU<int, cpu>(NULL, 1, 1));

    SharedPtr<NumericTable> minimimWrapper(minimum, EmptyDeleter<NumericTable>());
    function->sumOfFunctionsParameter->batchIndices = ntBatchIndices;
    function->sumOfFunctionsInput->set(sum_of_functions::argument, minimimWrapper);

    ReadRows<algorithmFPType, cpu, NumericTable> learningRateBD(*learningRateSequence, 0, learningRateSequence->getNumberOfRows());
    const algorithmFPType *learningRateArray = learningRateBD.get();
    const size_t learningRateLength = learningRateSequence->getNumberOfColumns();
    const double accuracyThreshold = parameter->accuracyThreshold;

    WriteRows<int, cpu, NumericTable> nIterationsBD(*nIterations, 0, 1);
    int *nProceededIterations = nIterationsBD.get();
    *nProceededIterations = (int)nIter;

    size_t startIteration = 0, epoch = 0, nProceededIters = 0;
    if(lastIterationInput != nullptr)
    {
        ReadRows<int, cpu, NumericTable> lastIterationInputBD(lastIterationInput, 0, 1);
        const int *lastIterationInputArray = lastIterationInputBD.get();
        startIteration = lastIterationInputArray[0];
    }
    for(epoch = startIteration; epoch < (startIteration + nIter); epoch++)
    {
        ntBatchIndices->setArray(const_cast<int *>(rngTask.get(*this->_errors)), ntBatchIndices->getNumberOfRows());
        services::Status s = function->computeNoThrow();
        if(!s)
        {
            this->_errors->add(function->getErrors()->getErrors());
            nProceededIterations[0] = nProceededIters;
            DAAL_RETURN_STATUS()
        }

        NumericTable *gradient = function->getResult()->get(objective_function::gradientIdx).get();
        if(nIter != 1)
        {
            const algorithmFPType pointNorm = vectorNorm(minimum);
            const algorithmFPType gradientNorm = vectorNorm(gradient);
            const algorithmFPType one = 1.0;
            const algorithmFPType gradientThreshold = accuracyThreshold * daal::internal::Math<algorithmFPType, cpu>::sMax(one, pointNorm);
            if(gradientNorm < gradientThreshold)
            {
                nProceededIterations[0] = (int)nProceededIters;
                break;
            }
        }

        const algorithmFPType learningRate = learningRateArray[epoch % learningRateLength];

        processByBlocks<cpu>(nRows, this->_errors.get(), [ = ](size_t startOffset, size_t nRowsInBlock)
        {
            WriteRows<algorithmFPType, cpu, NumericTable> workValueBD(*minimum, startOffset, nRowsInBlock);
            algorithmFPType *workLocal = workValueBD.get();
            ReadRows<algorithmFPType, cpu, NumericTable> ntGradientBD(*gradient, startOffset, nRowsInBlock);
            const algorithmFPType *gradientLocal = ntGradientBD.get();
            PRAGMA_SIMD_ASSERT
            for(int j = 0; j < nRowsInBlock; j++)
            {
                workLocal[j] = workLocal[j] - learningRate * gradientLocal[j];
            }
        },
        256);
        nProceededIters++;
    }
    if(lastIterationResult)
    {
        WriteRows<int, cpu, NumericTable> lastIterationResultBD(lastIterationResult, 0, 1);
        int *lastIterationResultArray = lastIterationResultBD.get();
        lastIterationResultArray[0] = startIteration + nProceededIters;
    }
    if(parameter->optionalResultRequired && !rngTask.save(optionalResult, sgd::rngState, *this->_errors))
    {
        this->_errors->add(ErrorMemoryAllocationFailed);
        DAAL_RETURN_STATUS()
    }
    DAAL_RETURN_STATUS()
}

} // namespace daal::internal

} // namespace sgd

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
