/* file: sgd_dense_minibatch_impl.i */
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
//  Implementation of sgd miniBatch algorithm
//
// Mu Li, Tong Zhang, Yuqiang Chen, Alexander J. Smola Efficient Mini-batch Training for Stochastic Optimization
//--
*/

#ifndef __SGD_DENSE_MINIBATCH_IMPL_I__
#define __SGD_DENSE_MINIBATCH_IMPL_I__

#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "service_math.h"
#include "service_utils.h"
#include "service_rng.h"

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
 *  \brief Kernel for SGD miniBatch calculation
 */
template<typename algorithmFPType, CpuType cpu>
services::Status SGDKernel<algorithmFPType, miniBatch, cpu>::compute(NumericTable *inputArgument, NumericTable *minimum, NumericTable *nIterations,
    Parameter<miniBatch> *parameter, NumericTable *learningRateSequence,
    NumericTable *batchIndices, OptionalArgument *optionalArgument, OptionalArgument *optionalResult)
{
    const size_t argumentSize = inputArgument->getNumberOfRows();
    const size_t nIter = parameter->nIterations;
    const size_t L = parameter->innerNIterations;
    const size_t batchSize     = parameter->batchSize;

    /* if nIter == 0, set result as start point, the number of executed iters to 0 */
    if(nIter == 0 || L == 0)
    {
        SGDMiniBatchTask<algorithmFPType, cpu> task(
            this->_errors,
            argumentSize,
            minimum,
            inputArgument,
            nIterations
        );
        task.nProceededIterations[0] = 0;
        DAAL_RETURN_STATUS()
    }

    SharedPtr<sum_of_functions::Batch> function = parameter->function;
    const size_t nTerms = function->sumOfFunctionsParameter->numberOfTerms;
    NumericTable *conservativeSequence = parameter->conservativeSequence.get();

    SGDMiniBatchTask<algorithmFPType, cpu> task(
        this->_errors,
        batchSize,
        argumentSize,
        nIter,
        nTerms,
        minimum,
        inputArgument,
        learningRateSequence,
        conservativeSequence,
        nIterations,
        batchIndices,
        optionalArgument ? NumericTable::cast(optionalArgument->get(sgd::pastWorkValue)).get() : nullptr,
        optionalResult ? NumericTable::cast(optionalResult->get(sgd::pastWorkValue)).get() : nullptr,
        optionalArgument ? NumericTable::cast(optionalArgument->get(iterative_solver::lastIteration)).get() : nullptr,
        optionalResult ? NumericTable::cast(optionalResult->get(iterative_solver::lastIteration)).get() : nullptr
    );
    if(this->_errors->size() != 0) DAAL_RETURN_STATUS()

    NumericTablePtr previousArgument = function->sumOfFunctionsInput->get(sum_of_functions::argument);
    function->sumOfFunctionsInput->set(sum_of_functions::argument, task.ntWorkValue);

    double accuracyThreshold = parameter->accuracyThreshold;

    BlockDescriptor<algorithmFPType> gradientBlock;
    NumericTablePtr ntGradient;
    algorithmFPType learningRate, consCoeff;

    NumericTablePtr previousBatchIndices = function->sumOfFunctionsParameter->batchIndices;
    function->sumOfFunctionsParameter->batchIndices = task.ntBatchIndices;

    ReadRows<int, cpu> predefinedBatchIndicesBD(batchIndices, 0, nIter);
    using namespace iterative_solver::internal;
    RngTask<int, cpu> rngTask(predefinedBatchIndicesBD.get(), batchSize);
    DAAL_CHECK(batchIndices || rngTask.init(optionalArgument, nTerms, parameter->seed, sgd::rngState), ErrorMemoryAllocationFailed);

    size_t epoch;
    for(epoch = task.startIteration; epoch < (task.startIteration + nIter); epoch++)
    {
        if(epoch % L == 0 || epoch == task.startIteration)
        {
            learningRate = task.learningRateArray[(epoch / L) % task.learningRateLength];
            consCoeff = task.consCoeffsArray[(epoch / L) % task.consCoeffsLength];
            if(task.indicesStatus == user || task.indicesStatus == random)
            {
            task.ntBatchIndices->setArray(const_cast<int*>(rngTask.get(*this->_errors, RngTask<int, cpu>::eUniformWithoutReplacement)),
                task.ntBatchIndices->getNumberOfRows());
            }
        }
        services::Status s = function->computeNoThrow();
        if(!s)
        {
            task.nProceededIterations[0] = (int)(epoch - task.startIteration);
            this->_errors->add(function->getErrors()->getErrors());
            break;
        }

        ntGradient = function->getResult()->get(objective_function::gradientIdx);
        ntGradient->getBlockOfRows(0, argumentSize, readOnly, gradientBlock);
        algorithmFPType *gradient = gradientBlock.getBlockPtr();

        if(epoch % L == 0)
        {
            if(nIter > 1)
            {
                double pointNorm    = vectorNorm(task.workValue, argumentSize);
                double gradientNorm = vectorNorm(gradient, argumentSize);

                double gradientThreshold = accuracyThreshold * daal::internal::Math<algorithmFPType, cpu>::sMax(1.0, pointNorm);
                if(gradientNorm < gradientThreshold) { ntGradient->releaseBlockOfRows(gradientBlock); break; }
            }
            daal_memcpy_s(task.prevWorkValue, argumentSize * sizeof(algorithmFPType), task.workValue, argumentSize * sizeof(algorithmFPType));
        }

        task.makeStep(gradient, learningRate, consCoeff, argumentSize);

        ntGradient->releaseBlockOfRows(gradientBlock);
    }
    task.nProceededIterations[0] = (int)task.nProceededIters;
    function->sumOfFunctionsParameter->batchIndices = previousBatchIndices;
    function->sumOfFunctionsInput->set(sum_of_functions::argument, previousArgument);

    if(parameter->optionalResultRequired && !rngTask.save(optionalResult, sgd::rngState, *this->_errors))
    {
        this->_errors->add(ErrorMemoryAllocationFailed);
        DAAL_RETURN_STATUS()
    }
    DAAL_RETURN_STATUS()
}

template<typename algorithmFPType, CpuType cpu>
void SGDMiniBatchTask<algorithmFPType, cpu>::makeStep(
    const algorithmFPType *gradient,
    algorithmFPType learningRate,
    algorithmFPType consCoeff,
    size_t argumentSize)
{
    for(size_t j = 0; j < argumentSize; j++)
    {
        workValue[j] = workValue[j] - learningRate * (gradient[j] + consCoeff * (workValue[j] - prevWorkValue[j]));
    }
    nProceededIters++;
}

template<typename algorithmFPType, CpuType cpu>
SGDMiniBatchTask<algorithmFPType, cpu>::SGDMiniBatchTask(
    const services::KernelErrorCollectionPtr& errors_,
    size_t argumentSize_,
    NumericTable *resultTable,
    NumericTable *startValueTable,
    NumericTable *nIterationsTable
) :
    _errors(errors_),
    argumentSize(argumentSize_),
    mtWorkValue(resultTable),
    mtNIterations(nIterationsTable),
    workValue(NULL),
    nProceededIterations(NULL),
    learningRateArray(NULL),
    consCoeffsArray(NULL),
    prevWorkValue(NULL)
{
    mtWorkValue.getBlockOfRows(0, argumentSize, &workValue);
    mtNIterations.getBlockOfRows(0, 1, &nProceededIterations);
    setStartValue(startValueTable);
}

template<typename algorithmFPType, CpuType cpu>
SGDMiniBatchTask<algorithmFPType, cpu>::~SGDMiniBatchTask()
{
    mtNIterations.release();

    mtConsCoeffs.release();
    mtLearningRate.release();

    if(indicesStatus == user)
    {
        mtPredefinedBatchIndices.release();
    }

    if(lastIterationResult)
    {
        WriteRows<int, cpu, NumericTable> lastIterationResultBD(lastIterationResult.get(), 0, 1);
        int *lastIterationResultArray = lastIterationResultBD.get();
        lastIterationResultArray[0] = startIteration + nProceededIters;
    }

    if(pastWorkValueResult)
    {
        WriteRows<algorithmFPType, cpu, NumericTable> pastWorkValueResultBD(pastWorkValueResult.get(), 0, pastWorkValueResult->getNumberOfRows());
        algorithmFPType *pastWorkValueResultArray = pastWorkValueResultBD.get();
        daal_memcpy_s(pastWorkValueResultArray, argumentSize * sizeof(algorithmFPType), prevWorkValue, argumentSize * sizeof(algorithmFPType));
    }

    mtWorkValue.release();
    if(prevWorkValue)
    {
        daal_free(prevWorkValue);
    }
}

template<typename algorithmFPType, CpuType cpu>
void SGDMiniBatchTask<algorithmFPType, cpu>::setStartValue(NumericTable *startValueTable)
{
    BlockMicroTable<algorithmFPType, readOnly, cpu> mtStartValue(startValueTable);
    algorithmFPType *startValueArray;
    mtStartValue.getBlockOfRows(0, argumentSize, &startValueArray);
    daal_memcpy_s(workValue, argumentSize * sizeof(algorithmFPType), startValueArray, argumentSize * sizeof(algorithmFPType));
    mtStartValue.release();
}

template<typename algorithmFPType, CpuType cpu>
SGDMiniBatchTask<algorithmFPType, cpu>::SGDMiniBatchTask(
    const services::KernelErrorCollectionPtr &errors_,
    size_t batchSize_,
    size_t argumentSize_,
    size_t nIter_,
    size_t nTerms_,
    NumericTable *resultTable,
    NumericTable *startValueTable,
    NumericTable *learningRateSequenceTable,
    NumericTable *conservativeSequenceTable,
    NumericTable *nIterationsTable,
    NumericTable *batchIndicesTable,
    NumericTable *pastWorkValueInput,
    NumericTable *pastWorkValueResultNT,
    NumericTable *lastIterationInput,
    NumericTable *lastIterationResultNT
) :
    _errors(errors_),
    batchSize(batchSize_),
    argumentSize(argumentSize_),
    nIter(nIter_),
    nTerms(nTerms_),
    mtWorkValue(resultTable),
    mtLearningRate(learningRateSequenceTable),
    mtConsCoeffs(conservativeSequenceTable),
    mtNIterations(nIterationsTable),
    mtPredefinedBatchIndices(batchIndicesTable),
    workValue(NULL),
    nProceededIterations(NULL),
    learningRateArray(NULL),
    consCoeffsArray(NULL),
    prevWorkValue(NULL),
    lastIterationResult(lastIterationResultNT, EmptyDeleter<NumericTable>()),
    pastWorkValueResult(pastWorkValueResultNT, EmptyDeleter<NumericTable>()),
    startIteration(0),
    nProceededIters(0)
{
    mtWorkValue.getBlockOfRows(0, argumentSize, &workValue);
    setStartValue(startValueTable);

    ntWorkValue = SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu>>(new HomogenNumericTableCPU<algorithmFPType, cpu>(workValue, 1, argumentSize));

    mtLearningRate.getBlockOfRows(0, mtLearningRate.getFullNumberOfRows(), &learningRateArray);
    learningRateLength = mtLearningRate.getFullNumberOfColumns();

    mtConsCoeffs.getBlockOfRows(0, mtConsCoeffs.getFullNumberOfRows(), &consCoeffsArray);
    consCoeffsLength = mtConsCoeffs.getFullNumberOfColumns();

    mtNIterations.getBlockOfRows(0, 1, &nProceededIterations);
    nProceededIterations[0] = 0;

    prevWorkValue = (algorithmFPType *) daal_malloc(argumentSize * sizeof(algorithmFPType));

    if(batchIndicesTable != NULL)
    {
        indicesStatus = user;
    }
    else
    {
        if(batchSize < nTerms)
        {
            indicesStatus = random;
        }
        else
        {
            indicesStatus = all;
        }
    }

    if(indicesStatus == user || indicesStatus == random)
    {
        ntBatchIndices = SharedPtr<HomogenNumericTableCPU<int, cpu>>(new HomogenNumericTableCPU<int, cpu>(NULL, batchSize, 1));
    }
    else if(indicesStatus == all)
    {
        ntBatchIndices = SharedPtr<HomogenNumericTableCPU<int, cpu>>();
    }

    if(lastIterationInput)
    {
        ReadRows<int, cpu, NumericTable> lastIterationInputBD(lastIterationInput, 0, 1);
        const int *lastIterationInputArray = lastIterationInputBD.get();
        startIteration = lastIterationInputArray[0];
    }

    if(pastWorkValueInput)
    {
        ReadRows<algorithmFPType, cpu, NumericTable> pastWorkValueInputBD(pastWorkValueInput, 0, pastWorkValueInput->getNumberOfRows());
        const algorithmFPType *pastWorkValueInputArray = pastWorkValueInputBD.get();
        daal_memcpy_s(prevWorkValue, argumentSize * sizeof(algorithmFPType), pastWorkValueInputArray, argumentSize * sizeof(algorithmFPType));
    }
}

} // namespace daal::internal
} // namespace sgd
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

#endif
