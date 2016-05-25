/* file: sgd_dense_minibatch_impl.i */
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
void SGDKernel<algorithmFPType, miniBatch, cpu>::compute(iterative_solver::Input *input, iterative_solver::Result *result, Parameter<miniBatch> *parameter)
{
    size_t nFeatures     = input->get(iterative_solver::inputArgument)->getNumberOfColumns();
    size_t maxIterations = parameter->nIterations;
    size_t L             = parameter->innerNIterations;
    size_t batchSize     = parameter->batchSize;

    /* if maxIterations == 0, set result as start point, the number of executed iters to 0 */
    if(maxIterations == 0 || L == 0)
    {
        SGDMiniBatchTask<algorithmFPType, cpu> task(
            this->_errors,
            nFeatures,
            result->get(iterative_solver::minimum),
            input->get(iterative_solver::inputArgument),
            result->get(iterative_solver::nIterations)
        );
        task.nProceededIterations[0] = 0;
        return;
    }

    SharedPtr<sum_of_functions::Batch> function = parameter->function;
    size_t nTerms = function->sumOfFunctionsParameter->numberOfTerms;

    SGDMiniBatchTask<algorithmFPType, cpu> task(
        this->_errors,
        batchSize,
        nFeatures,
        maxIterations,
        nTerms,
        result->get(iterative_solver::minimum),
        input->get(iterative_solver::inputArgument),
        parameter->learningRateSequence,
        parameter->conservativeSequence,
        result->get(iterative_solver::nIterations),
        parameter->batchIndices
    );
    if(this->_errors->size() != 0) { return; }

    SharedPtr<NumericTable> previousArgument = function->sumOfFunctionsInput->get(sum_of_functions::argument);
    function->sumOfFunctionsInput->set(sum_of_functions::argument, task.ntWorkValue);

    double accuracyThreshold = parameter->accuracyThreshold;
    double pointNorm = 0, gradientNorm = 0, gradientThreshold = 0, one = 1.0;

    BlockDescriptor<algorithmFPType> gradientBlock;
    SharedPtr<NumericTable> ntGradient;
    algorithmFPType learningRate, consCoeff;

    SharedPtr<NumericTable> previousBatchIndices = function->sumOfFunctionsParameter->batchIndices;
    function->sumOfFunctionsParameter->batchIndices = task.ntBatchIndices;

    IntRng<int,cpu> rng(parameter->seed);

    size_t epoch;
    for(epoch = 0; epoch < maxIterations; epoch++)
    {
        if(task.indicesStatus == user)
        {
            task.getNextPredefinedIndices(epoch);
            if(function->getErrors()->size() != 0) { break; }
            task.ntBatchIndices->setArray(task.randomTerm);
        }
        else if (task.indicesStatus == random)
        {
            rng.uniformWithoutReplacement(batchSize, 0, nTerms, task.randomTerm);
        }

        function->compute();
        if(function->getErrors()->size() != 0) { break; }
        ntGradient = function->getResult()->get(objective_function::resultCollection, objective_function::gradientIdx);
        ntGradient->getBlockOfRows(0, 1, readOnly, gradientBlock);

        algorithmFPType *gradient = gradientBlock.getBlockPtr();
        pointNorm    = vectorNorm(task.workValue, nFeatures);
        gradientNorm = vectorNorm(gradient, nFeatures);

        gradientThreshold = accuracyThreshold * daal::sMax<algorithmFPType, cpu>(one, pointNorm);
        if(gradientNorm <= gradientThreshold) { ntGradient->releaseBlockOfRows(gradientBlock); break; }

        daal_memcpy_s(task.prevWorkValue, nFeatures * sizeof(algorithmFPType), task.workValue, nFeatures * sizeof(algorithmFPType));

        (task.learningRateLength > 1) ? learningRate = task.learningRateArray[epoch] : learningRate = task.learningRateArray[0];
        (task.consCoeffsLength > 1) ? consCoeff = task.consCoeffsArray[epoch] : consCoeff = task.consCoeffsArray[0];

        makeStep(task.workValue, task.prevWorkValue, gradient, learningRate, consCoeff, nFeatures);

        ntGradient->releaseBlockOfRows(gradientBlock);

        for(size_t inner = 1; inner < L; inner++)
        {
            function->compute();
            if(function->getErrors()->size() != 0)
            {
                task.nProceededIterations[0] = (int)epoch;
                function->sumOfFunctionsParameter->batchIndices = previousBatchIndices;
                function->sumOfFunctionsInput->set(sum_of_functions::argument, previousArgument);
                return;
            }
            ntGradient = function->getResult()->get(objective_function::resultCollection, objective_function::gradientIdx);
            ntGradient->getBlockOfRows(0, 1, readOnly, gradientBlock);

            makeStep(task.workValue, task.prevWorkValue, gradient, learningRate, consCoeff, nFeatures);

            ntGradient->releaseBlockOfRows(gradientBlock);
        }
    }
    task.nProceededIterations[0] = (int)epoch;
    function->sumOfFunctionsParameter->batchIndices = previousBatchIndices;
    function->sumOfFunctionsInput->set(sum_of_functions::argument, previousArgument);
    return;
}

template<typename algorithmFPType, CpuType cpu>
void SGDKernel<algorithmFPType, miniBatch, cpu>::makeStep(
    algorithmFPType *workValue,
    algorithmFPType *prevWorkValue,
    algorithmFPType *gradient,
    algorithmFPType learningRate,
    algorithmFPType consCoeff,
    size_t nFeatures)
{
    for(size_t j = 0; j < nFeatures; j++)
    {
        workValue[j] = workValue[j] - learningRate * (gradient[j] + consCoeff * (workValue[j] - prevWorkValue[j]));
    }
}

template<typename algorithmFPType, CpuType cpu>
SGDMiniBatchTask<algorithmFPType, cpu>::SGDMiniBatchTask(
    SharedPtr<services::KernelErrorCollection> errors_,
    size_t nFeatures_,
    SharedPtr<NumericTable> resultTable,
    SharedPtr<NumericTable> startValueTable,
    SharedPtr<NumericTable> nIterationsTable
) :
    _errors(errors_),
    nFeatures(nFeatures_),
    mtWorkValue(resultTable.get()),
    mtNIterations(nIterationsTable.get()),
    workValue(NULL),
    nProceededIterations(NULL),
    learningRateArray(NULL),
    consCoeffsArray(NULL),
    prevWorkValue(NULL),
    randomTerm(NULL)
{
    mtWorkValue.getBlockOfRows(0, 1, &workValue);
    mtNIterations.getBlockOfRows(0, 1, &nProceededIterations);
    setStartValue(startValueTable);
}

template<typename algorithmFPType, CpuType cpu>
SGDMiniBatchTask<algorithmFPType, cpu>::~SGDMiniBatchTask()
{
    mtNIterations.release();
    mtWorkValue.release();

    mtConsCoeffs.release();
    mtLearningRate.release();

    if(prevWorkValue) {daal_free(prevWorkValue);}

    if(indicesStatus == user)
    {
        mtPredefinedBatchIndices.release();
    }
    else if(indicesStatus == random)
    {
        daal_free(randomTerm);
    }
}

template<typename algorithmFPType, CpuType cpu>
void SGDMiniBatchTask<algorithmFPType, cpu>::setStartValue(SharedPtr<NumericTable> startValueTable)
{
    BlockMicroTable<algorithmFPType, readOnly, cpu> mtStartValue(startValueTable.get());
    algorithmFPType *startValueArray;
    mtStartValue.getBlockOfRows(0, 1, &startValueArray);
    daal_memcpy_s(workValue, nFeatures * sizeof(algorithmFPType), startValueArray, nFeatures * sizeof(algorithmFPType));
    mtStartValue.release();
}

template<typename algorithmFPType, CpuType cpu>
SGDMiniBatchTask<algorithmFPType, cpu>::SGDMiniBatchTask(
    SharedPtr<services::KernelErrorCollection> errors_,
    size_t batchSize_,
    size_t nFeatures_,
    size_t maxIterations_,
    size_t nTerms_,
    SharedPtr<NumericTable> resultTable,
    SharedPtr<NumericTable> startValueTable,
    SharedPtr<NumericTable> learningRateSequenceTable,
    SharedPtr<NumericTable> conservativeSequenceTable,
    SharedPtr<NumericTable> nIterationsTable,
    SharedPtr<NumericTable> batchIndicesTable
) :
    _errors(errors_),
    batchSize(batchSize_),
    nFeatures(nFeatures_),
    maxIterations(maxIterations_),
    nTerms(nTerms_),
    mtWorkValue(resultTable.get()),
    mtLearningRate(learningRateSequenceTable.get()),
    mtConsCoeffs(conservativeSequenceTable.get()),
    mtNIterations(nIterationsTable.get()),
    mtPredefinedBatchIndices(batchIndicesTable.get()),
    workValue(NULL),
    nProceededIterations(NULL),
    learningRateArray(NULL),
    consCoeffsArray(NULL),
    prevWorkValue(NULL),
    randomTerm(NULL)
{
    mtWorkValue.getBlockOfRows(0, 1, &workValue);
    setStartValue(startValueTable);

    ntWorkValue = SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu>>(new HomogenNumericTableCPU<algorithmFPType, cpu>(workValue, nFeatures, 1));

    mtLearningRate.getBlockOfRows(0, 1, &learningRateArray);
    learningRateLength = mtLearningRate.getFullNumberOfColumns();

    mtConsCoeffs.getBlockOfRows(0, 1, &consCoeffsArray);
    consCoeffsLength = mtConsCoeffs.getFullNumberOfColumns();

    mtNIterations.getBlockOfRows(0, 1, &nProceededIterations);
    nProceededIterations[0] = 0;

    prevWorkValue = (algorithmFPType *) daal_malloc(nFeatures * sizeof(algorithmFPType));

    if(batchIndicesTable.get() != NULL)
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

    if(indicesStatus == user)
    {
        ntBatchIndices = SharedPtr<HomogenNumericTableCPU<int, cpu>>(new HomogenNumericTableCPU<int, cpu>(NULL, batchSize, 1));
    }
    else if(indicesStatus == random)
    {
        randomTerm = (int *) daal_malloc(batchSize * sizeof(int));
        if(!randomTerm) { this->_errors->add(ErrorMemoryAllocationFailed); return; }
        ntBatchIndices = SharedPtr<HomogenNumericTableCPU<int, cpu>>(new HomogenNumericTableCPU<int, cpu>(randomTerm, batchSize, 1));
    }
    else if(indicesStatus == all)
    {
        ntBatchIndices = SharedPtr<HomogenNumericTableCPU<int, cpu>>();
    }
}

template<typename algorithmFPType, CpuType cpu>
void SGDMiniBatchTask<algorithmFPType, cpu>::getNextPredefinedIndices(size_t iter)
{
    if(indicesStatus == user)
    {
        size_t nReadRows = mtPredefinedBatchIndices.getBlockOfRows(iter, 1, &randomTerm);
        if(nReadRows != 1)
        {
            mtPredefinedBatchIndices.release();
            this->_errors->add(ErrorMemoryAllocationFailed);
            return;
        }
    }
}

} // namespace daal::internal
} // namespace sgd
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

#endif
