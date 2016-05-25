/* file: lbfgs_dense_default_impl.i */
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
//  Implementation of LBFGS algorithm
//--
*/

#ifndef __LBFGS_DENSE_DEFAULT_IMPL__
#define __LBFGS_DENSE_DEFAULT_IMPL__

#include "service_blas.h"
#include "service_rng.h"

using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace lbfgs
{
namespace internal
{

/**
 * \brief Kernel for LBFGS calculation
 */
template<typename algorithmFPType, CpuType cpu>
void LBFGSKernel<algorithmFPType, defaultDense, cpu>::compute(
            iterative_solver::Input *input, iterative_solver::Result *result, Parameter *parameter)
{
    size_t argumentSize = input->get(iterative_solver::inputArgument)->getNumberOfColumns();
    size_t nIterations = parameter->nIterations;
    if (nIterations == 0)
    {
        /* Initialize the resulting argument of objective function with the input argument */
        LBFGSTask<algorithmFPType, cpu> task(argumentSize,
            result->get(iterative_solver::minimum).get(), input->get(iterative_solver::inputArgument).get(), _errors);
        setNIterations(result->get(iterative_solver::nIterations).get(), 0);
        return;
    }
    size_t L = parameter->L;
    size_t m = parameter->m;
    double accuracyThreshold = parameter->accuracyThreshold;

    SharedPtr<sum_of_functions::Batch> gradientFunction = parameter->function;
    size_t nTerms = gradientFunction->sumOfFunctionsParameter->numberOfTerms;

    size_t batchSize = ((parameter->batchSize < nTerms) ? parameter->batchSize : nTerms);
    size_t correctionPairBatchSize = ((parameter->correctionPairBatchSize < nTerms) ? parameter->correctionPairBatchSize : nTerms);

    size_t nStepLength = parameter->stepLengthSequence->getNumberOfColumns();

    LBFGSTask<algorithmFPType, cpu> task(argumentSize, nTerms, m, batchSize, parameter->batchIndices.get(),
        correctionPairBatchSize, parameter->correctionPairBatchIndices.get(),
        result->get(iterative_solver::minimum).get(), input->get(iterative_solver::inputArgument).get(),
        nStepLength, parameter->stepLengthSequence.get(), _errors);
    if (this->_errors->size() != 0) { return; }

    SharedPtr<NumericTable> argumentTable(
        new HomogenNumericTableCPU<algorithmFPType, cpu>(task.argument, argumentSize, 1));
    gradientFunction->sumOfFunctionsParameter->batchIndices     = task.ntBatchIndices;
    gradientFunction->sumOfFunctionsParameter->resultsToCompute = objective_function::gradient;
    gradientFunction->sumOfFunctionsInput->set(sum_of_functions::argument, argumentTable);

    SharedPtr<sum_of_functions::Batch> hessianFunction = gradientFunction->clone();
    hessianFunction->sumOfFunctionsParameter->batchIndices = task.ntCorrectionPairBatchIndices;
    hessianFunction->sumOfFunctionsParameter->resultsToCompute = objective_function::hessian;
    hessianFunction->sumOfFunctionsInput->set(sum_of_functions::argument, task.argumentLCurTable);

    BlockDescriptor<algorithmFPType> gradientBlock;
    BlockDescriptor<algorithmFPType> hessianBlock;
    SharedPtr<NumericTable> ntGradient;
    SharedPtr<NumericTable> ntHessian;

    algorithmFPType invL = (algorithmFPType)1.0 / (algorithmFPType)L;

    algorithmFPType *argument      = task.argument;
    algorithmFPType *argumentLCur  = task.argumentLCur;
    algorithmFPType *argumentLPrev = task.argumentLPrev;

    IntRng<int,cpu> rng(parameter->seed);

    size_t nLIterations = nIterations / L;
    size_t correctionIndex = m - 1;
    size_t t = 0;
    size_t epoch = 0;

    for (; t < nLIterations;)
    {
        for (; epoch < (t + 1) * L; epoch++)
        {
            for (size_t j = 0; j < argumentSize; j++)
            {
                argumentLCur[j] += argument[j];
            }

            if (!updateArgument(t, epoch, m, correctionIndex, nTerms, argumentSize, batchSize, nStepLength,
                                accuracyThreshold, gradientFunction, ntGradient, gradientBlock, task, rng,
                                argument))
            { setNIterations(result->get(iterative_solver::nIterations).get(), epoch); return; }
        }

        for (size_t j = 0; j < argumentSize; j++)
        {
            argumentLCur[j] *= invL;
        }

        t++;
        if (t >= 2)
        {
            /* Compute new correction pair */
            correctionIndex = mod(correctionIndex + 1, m);

            updateBatchIndices(nTerms, correctionPairBatchSize, &(task.correctionPairBatchIndices),
                               task.correctionPairBatchIndicesStatus, task.mtCorrectionPairBatchIndices,
                               task.ntCorrectionPairBatchIndices, rng);

            hessianFunction->compute();
            if (hessianFunction->getErrors()->size() != 0) { setNIterations(result->get(iterative_solver::nIterations).get(), epoch); return; }
            if (task.correctionPairBatchIndicesStatus == user) { task.mtCorrectionPairBatchIndices.release(); }

            ntHessian = hessianFunction->getResult()->get(objective_function::resultCollection, objective_function::hessianIdx);
            ntHessian->getBlockOfRows(0, argumentSize, readOnly, hessianBlock);

            computeCorrectionPair(argumentSize, argumentLCur, argumentLPrev, hessianBlock.getBlockPtr(),
                task.correctionS + correctionIndex * argumentSize,
                task.correctionY + correctionIndex * argumentSize, &(task.rho[correctionIndex]));

            ntHessian->releaseBlockOfRows(hessianBlock);
        }
        for (size_t j = 0; j < argumentSize; j++)
        {
            argumentLPrev[j] = argumentLCur[j];
            argumentLCur[j] = 0.0;
        }
    }

    for (; epoch < nIterations; epoch++)
    {
        if (!updateArgument(t, epoch, m, correctionIndex, nTerms, argumentSize, batchSize, nStepLength,
                            accuracyThreshold, gradientFunction, ntGradient, gradientBlock, task, rng,
                            argument))
        { setNIterations(result->get(iterative_solver::nIterations).get(), epoch); return; }
    }

    setNIterations(result->get(iterative_solver::nIterations).get(), epoch);
}

template<typename algorithmFPType, CpuType cpu>
void LBFGSKernel<algorithmFPType, defaultDense, cpu>::setNIterations(NumericTable *nIterationsTable, size_t nIterations)
{
    BlockMicroTable<int, writeOnly, cpu> mtNIterations(nIterationsTable);
    int *nProceededIterations = NULL;
    mtNIterations.getBlockOfRows(0, 1, &nProceededIterations);
    *nProceededIterations = (int)nIterations;
    mtNIterations.release();
}

/**
 * Computes the dot product of two vectors
 *
 * \param[in] n     Number of elements in each input vector
 * \param[in] x     Array that contains elements of the first input vector
 * \param[in] y     Array that contains elements of the second input vector
 * \return Resulting dot product
 */
template<typename algorithmFPType, CpuType cpu>
algorithmFPType LBFGSKernel<algorithmFPType, defaultDense, cpu>::dotProduct(
            size_t n, const algorithmFPType *x, const algorithmFPType *y)
{
    algorithmFPType dot = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        dot += x[i] * y[i];
    }
    return dot;
}

/**
 * Updates argument of the objective function
 *
 * \param[in] t                     Index of the outer loop of LBFGS algorithm
 * \param[in] epoch                 Index of the inner loop of LBFGS algorithm
 * \param[in] m                     Memory parameter of LBFGS
 * \param[in] correctionIndex       Index of starting correction pair in a cyclic buffer
 * \param[in] argumentSize          Number of elements in the argument of objective function
 * \param[in] batchSize             Number of terms of objective function to be used in stochastic gradient
 *                                  computation
 * \param[in] nStepLength           Number of values in the provided step-length sequence
 * \param[in] accuracyThreshold     Accuracy of the LBFGS algorithm
 * \param[in] gradientFunction      Objective function for stochastic gradient computations
 * \param[in] ntGradient            Numeric table that stores the stochastic gradient
 * \param[in] gradientBlock         Block descriptor related to the stochastic gradient
 * \param[in] task                  Structure for storing data used in itermediate computations in LBFGS algorithm
 * \param[in] rng                   Random number generator
 * \param[in,out] argument          Array that contains the argument of objective function
 *
 * \return Flag. True if the argument was updated successfully. False, otherwise.
 */
template<typename algorithmFPType, CpuType cpu>
bool LBFGSKernel<algorithmFPType, defaultDense, cpu>::updateArgument(
            size_t t, size_t epoch, size_t m, size_t correctionIndex, size_t nTerms,
            size_t argumentSize, size_t batchSize, size_t nStepLength, algorithmFPType accuracyThreshold,
            SharedPtr<sum_of_functions::Batch> &gradientFunction,
            SharedPtr<NumericTable> &ntGradient, BlockDescriptor<algorithmFPType> &gradientBlock,
            LBFGSTask<algorithmFPType, cpu> &task, daal::internal::IntRng<int,cpu> &rng,
            algorithmFPType *argument)
{
    const algorithmFPType one = 1.0;

    updateBatchIndices(nTerms, batchSize, &(task.batchIndices), task.batchIndicesStatus,
                       task.mtBatchIndices, task.ntBatchIndices, rng);

    gradientFunction->compute();
    if (gradientFunction->getErrors()->size() != 0) { return false; }
    if (task.batchIndicesStatus == user) { task.mtBatchIndices.release(); }

    ntGradient = gradientFunction->getResult()->get(objective_function::resultCollection, objective_function::gradientIdx);
    ntGradient->getBlockOfRows(0, 1, readOnly, gradientBlock);

    algorithmFPType *gradient = gradientBlock.getBlockPtr();

    /* Check accuracy */
    if (dotProduct(argumentSize, gradient, gradient) <
        accuracyThreshold * daal::sMax<algorithmFPType, cpu>(one, dotProduct(argumentSize, argument, argument)))
    { return false; }

    /* Get step length on this iteration */
    algorithmFPType stepLength = ((nStepLength > 1) ? task.stepLength[epoch] : task.stepLength[0]);

    if (t >= 2)
    {
        /* Compute H * gradient */
        twoLoopRecursion(argumentSize, m, correctionIndex, gradient, task.correctionS, task.correctionY,
                         task.rho, task.alpha);
    }

    /* Update argument */
    for (size_t j = 0; j < argumentSize; j++)
    {
        argument[j] -= stepLength * gradient[j];
    }
    ntGradient->releaseBlockOfRows(gradientBlock);

    return true;
}

/**
 * Updates the array of objective function terms indices that are used in stochastic gradient
 * or Hessian matrix computations
 *
 * \param[in]  epoch              Iteration of LBFGS algorithm
 * \param[in]  nTerms             Full number of summands (terms) in objective function
 * \param[in]  batchSize          Number of terms of objective function to be used in stochastic gradient
 *                                or Hessian matrix computations
 * \param[out] batchIndices       Array of indices of objective function terms that are used
 *                                in stochastic gradient or Hessian matrix computations
 * \param[in]  batchIndicesStatus Status of the indices of objective function terms
 * \param[in]  ntBatchIndices     Numeric table that stores the indices of objective function terms
 * \param[in]  rng                Random number generator
 */
template<typename algorithmFPType, CpuType cpu>
void LBFGSKernel<algorithmFPType, defaultDense, cpu>::updateBatchIndices(
            size_t nTerms, size_t batchSize, int **batchIndices, IndicesStatus batchIndicesStatus,
            daal::internal::BlockMicroTable<int, readOnly, cpu> &mtBatchIndices,
            services::SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu> > &ntBatchIndices,
            daal::internal::IntRng<int, cpu> &rng)
{
    if (batchIndicesStatus == all)
    {
        return;
    }
    else if (batchIndicesStatus == user)
    {
        mtBatchIndices.getBlockOfRows(0, 1, batchIndices);
        ntBatchIndices->setArray(*batchIndices);
    }
    else // if (batchIndicesStatus == random)
    {
        rng.uniformWithoutReplacement(batchSize, 0, (int)nTerms, *batchIndices);
    }
}

/**
 * Two-loop recursion algorithm that computes approximation of inverse Hessian matrix
 * multiplied by input gradient vector from a set of correction pairs (s(j), y(j)), j = 1,...,m.
 *
 * See Algorithm 7.4 in [2].
 *
 * \param[in]  argumentSize    Number of elements in the argument of objective function
 * \param[in]  m               Number of correction pairs
 * \param[in]  correctionIndex Index of starting correction pair in a cyclic buffer
 * \param[in,out] gradient     On input:  Gradient vector.
 *                             On output: iterative_solver::Result of two-loop recursion.
 * \param[in]  correctionS     Array of correction pairs parts s(1), ..., s(m). See formula (2.1) in [1]
 * \param[in]  correctionY     Array of correction pairs parts y(1), ..., y(m). See formula (2.2) in [1]
 * \param[in]  rho             Array of parameters rho of BFGS update. See formula (7.17) in [2]
 * \param[in]  alpha           Array for intermediate results
 */
template<typename algorithmFPType, CpuType cpu>
void LBFGSKernel<algorithmFPType, defaultDense, cpu>::twoLoopRecursion(
            size_t argumentSize, size_t m, size_t correctionIndex, algorithmFPType *gradient,
            const algorithmFPType *correctionS, const algorithmFPType *correctionY, const algorithmFPType *rho,
            algorithmFPType *alpha)
{
    size_t index = 0;

    for (size_t k = 0; k < m; k++)
    {
        index = mod(correctionIndex + m - 1 - k, m);
        const algorithmFPType *correctionSPtr = correctionS + index * argumentSize;
        const algorithmFPType *correctionYPtr = correctionY + index * argumentSize;

        alpha[index] = rho[index] * dotProduct(argumentSize, correctionSPtr, gradient);

        for (size_t j = 0; j < argumentSize; j++)
        {
            gradient[j] -= alpha[index] * correctionYPtr[j];
        }
    }

    for (size_t k = 0; k < m; k++)
    {
        index = mod(correctionIndex + k, m);
        const algorithmFPType *correctionSPtr = correctionS + index * argumentSize;
        const algorithmFPType *correctionYPtr = correctionY + index * argumentSize;

        algorithmFPType beta = rho[index] * dotProduct(argumentSize, correctionYPtr, gradient);

        for (size_t j = 0; j < argumentSize; j++)
        {
            gradient[j] += correctionSPtr[j] * (alpha[index] - beta);
        }
    }
}

/**
 * Computes the correction pair (s, y) and the corresponding value rho
 *
 * \param[in]  argumentSize   Number of elements in the argument of objective function
 * \param[in]  argumentLCur   Average of objective function arguments for the last L iterations. See formula (2.1) in [1]
 * \param[in]  argumentLPrev  Average of objective function arguments for the previous L iterations. See formula (2.1) in [1]
 * \param[in]  hessian        Approximation of Hessian matrix of the objective function on the current iteration
 * \param[out] s              Part of the correction pair. See formula (2.1) in [1]
 * \param[out] y              Part of the correction pair. See formula (2.2) in [1]
 * \param[out] rhoPtr         Pointer to the value rho of BFGS update that corresponds to pair (s, y).
 *                            See formula (7.17) in [2]
 */
template<typename algorithmFPType, CpuType cpu>
void LBFGSKernel<algorithmFPType, defaultDense, cpu>::computeCorrectionPair(
            size_t argumentSize,
            const algorithmFPType *argumentLCur, const algorithmFPType *argumentLPrev, algorithmFPType *hessian,
            algorithmFPType *s, algorithmFPType *y, algorithmFPType *rhoPtr)
{
    for (size_t j = 0; j < argumentSize; j++)
    {
        s[j] = argumentLCur[j] - argumentLPrev[j];
    }

    char trans = 'N';
    algorithmFPType zero = 0.0;
    algorithmFPType one = 1.0;
    MKL_INT n = (MKL_INT)argumentSize;
    MKL_INT ione = 1;
    Blas<algorithmFPType, cpu>::xgemv(&trans, &n, &n, &one, hessian, &n, s, &ione, &zero, y, &ione);

    algorithmFPType rho = dotProduct(argumentSize, s, y);
    if (rho != 0.0) // threshold
    {
        rho = 1.0 / rho;
    }
    *rhoPtr = rho;
}

/**
 * Creates structure for storing data used in itermediate computations in LBFGS algorithm
 *
 * \param[in] argumentSize      Number of elements in the argument of objective function
 * \param[in] nTerms            Full number of summands (terms) in objective function
 * \param[in] m                 Memory parameter of LBFGS. Maximal number of correction pairs
 * \param[in] batchSize         Number of terms to compute the stochastic gradient
 * \param[in] batchIndicesTable Numeric table that represent indices that will be used
 *                              instead of random values for the stochastic gradient computations
 * \param[in] correctionPairBatchSize           Number of terms to compute the sub-sampled Hessian
 *                                              for correction pairs computation
 * \param[in] correctionPairBatchIndicesTable   Numeric table that represent indices that will be used
 *                                              instead of random values for the sub-sampled Hessian matrix
 *                                              computations
 * \param[in] argumentTable     Numeric table that stores the argument of objective function
 * \param[in] startValueTable   Numeric table that stores the starting point, the initial argument of objective function
 * \param[in] nStepLength       Number of values in the provided step-length sequence
 * \param[in] stepLengthTable   Numeric table that contains values of the step-length sequence
 * \param[in] _errors           Error collection of LBFGS algorithm
 */
template<typename algorithmFPType, CpuType cpu>
LBFGSTask<algorithmFPType, cpu>::LBFGSTask(
            size_t argumentSize, size_t nTerms, size_t m,
            size_t batchSize, NumericTable *batchIndicesTable,
            size_t correctionPairBatchSize, NumericTable *correctionPairBatchIndicesTable,
            NumericTable *argumentTable, NumericTable *startValueTable,
            size_t nStepLength, NumericTable *stepLengthTable,
            services::SharedPtr<services::KernelErrorCollection> &_errors) :
    mtBatchIndices(batchIndicesTable), mtCorrectionPairBatchIndices(correctionPairBatchIndicesTable),
    mtArgument(argumentTable), mtStepLength(stepLengthTable), _errors(_errors)
{
    /* Allocate memory to store intermediate results */
    argumentLCur  = (algorithmFPType *)daal::services::internal::service_calloc<algorithmFPType, cpu>(argumentSize);
    argumentLPrev = (algorithmFPType *)daal::services::internal::service_calloc<algorithmFPType, cpu>(argumentSize);
    rho            = (algorithmFPType *)daal::services::internal::service_calloc<algorithmFPType, cpu>(m);
    alpha          = (algorithmFPType *)daal_malloc(m * sizeof(algorithmFPType));
    correctionS = (algorithmFPType *)daal::services::internal::service_calloc<algorithmFPType, cpu>(m * argumentSize);
    correctionY = (algorithmFPType *)daal::services::internal::service_calloc<algorithmFPType, cpu>(m * argumentSize);
    if (!argumentLCur || !argumentLPrev || !rho || !alpha || !correctionS  || !correctionY)
    { this->_errors->add(ErrorMemoryAllocationFailed); return; }

    /* Initialize work value with a start value provided by user */
    setStartArgument(argumentSize, startValueTable);

    /* Get step-length sequence */
    mtStepLength.getBlockOfRows(0, 1, &stepLength);

    /* Create numeric table for storing the average of work values for the last L iterations */
    argumentLCurTable = SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> >(
        new HomogenNumericTableCPU<algorithmFPType, cpu>(argumentLCur, argumentSize, 1));
    if (!argumentLCurTable)
    { this->_errors->add(ErrorMemoryAllocationFailed); return; }

    /* Initialize indices for objective function gradient computations */
    batchIndices = NULL;
    batchIndicesStatus = all;
    if (batchSize < nTerms)
    {
        getBatchIndices(batchSize, batchIndicesTable, &batchIndices, mtBatchIndices, &batchIndicesStatus,
                        ntBatchIndices);
        if (this->_errors->size() != 0) { return; }
    }

    /* Initialize indices for objective function Hessian computations */
    correctionPairBatchIndices = NULL;
    correctionPairBatchIndicesStatus = all;
    if (correctionPairBatchSize < nTerms)
    {
        getBatchIndices(correctionPairBatchSize, correctionPairBatchIndicesTable, &correctionPairBatchIndices,
                        mtCorrectionPairBatchIndices, &correctionPairBatchIndicesStatus, ntCorrectionPairBatchIndices);
        if (this->_errors->size() != 0) { return; }
    }
}

/**
 * Creates structure for storing data used in itermediate computations in LBFGS algorithm
 * if number of iterations is 0
 *
 * \param[in] argumentSize      Number of elements in the argument of objective function
 * \param[in] argumentTable     Numeric table that stores the argument of objective function
 * \param[in] startValueTable   Numeric table that stores the starting point, the initial argument of objective function
 * \param[in] _errors           Error collection of LBFGS algorithm
 */
template<typename algorithmFPType, CpuType cpu>
LBFGSTask<algorithmFPType, cpu>::LBFGSTask(
            size_t argumentSize, NumericTable *argumentTable, NumericTable *startValueTable,
            services::SharedPtr<services::KernelErrorCollection> &_errors):
    mtArgument(argumentTable),
    argumentLCur(NULL), argumentLPrev(NULL), correctionS(NULL), correctionY(NULL),
    rho(NULL), alpha(NULL), stepLength(NULL), batchIndices(NULL), correctionPairBatchIndices(NULL),
    batchIndicesStatus(all), correctionPairBatchIndicesStatus(all), _errors(_errors)
{
    /* Initialize work value with a start value provided by user */
    setStartArgument(argumentSize, startValueTable);
}

template<typename algorithmFPType, CpuType cpu>
LBFGSTask<algorithmFPType, cpu>::~LBFGSTask()
{
    if (argumentLCur)  { daal_free(argumentLCur);  }
    if (argumentLPrev) { daal_free(argumentLPrev); }
    if (rho)           { daal_free(rho);           }
    if (alpha)         { daal_free(alpha);         }
    if (correctionS)   { daal_free(correctionS);   }
    if (correctionY)   { daal_free(correctionY);   }
    if (argument)      { mtArgument.release();     }
    if (stepLength)    { mtStepLength.release();   }
    releaseBatchIndices(batchIndices, mtBatchIndices, batchIndicesStatus);
    releaseBatchIndices(correctionPairBatchIndices, mtCorrectionPairBatchIndices, correctionPairBatchIndicesStatus);
}

/**
 * Sets the initial argument of objective function
 *
 * \param[in] argumentSize      Number of elements in the argument of objective function
 * \param[in] startValueTable   Numeric table that stores the starting point, the initial argument of objective function
 */
template<typename algorithmFPType, CpuType cpu>
void LBFGSTask<algorithmFPType, cpu>::setStartArgument(size_t argumentSize, NumericTable *startValueTable)
{
    /* Initialize work value with a start value provided by user */
    mtArgument.getBlockOfRows(0, 1, &argument);

    BlockMicroTable<algorithmFPType, readOnly, cpu> mtStartValue(startValueTable);
    algorithmFPType *startValue;
    mtStartValue.getBlockOfRows(0, 1, &startValue);
    daal_memcpy_s(argument, argumentSize * sizeof(algorithmFPType), startValue, argumentSize * sizeof(algorithmFPType));
    mtStartValue.release();
}

/**
 * Returns the array that contains the indices of objective function terms used for
 * stochastic gradient or sub-sampled Hessian matrix computation
 *
 * \param[in]  size             Number of indices
 * \param[in]  indicesTable     Numeric table that represent indices that will be used
 *                              instead of random values for the stochastic gradient
 *                              or sub-sampled Hessian matrix computations
 * \param[out] indices          Resulting array that contains the indices provided by user
 *                              or memory for storing randomly generated indices
 * \param[out] mtIndices        Micro-table that stores the indices
 * \param[out] indicesStatusPtr Status of the indices array
 * \param[out] ntIndices        Numeric table that stores the indices
 */
template<typename algorithmFPType, CpuType cpu>
void LBFGSTask<algorithmFPType, cpu>::getBatchIndices(
            size_t size, NumericTable *indicesTable, int **indices,
            daal::internal::BlockMicroTable<int, readOnly, cpu> &mtIndices, IndicesStatus *indicesStatusPtr,
            services::SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu> > &ntIndices)
{
    if (indicesTable)
    {
        *indicesStatusPtr = user;
    }
    else
    {
        *indicesStatusPtr = random;
        *indices = (int *)daal_malloc(size * sizeof(int));
        if (!(*indices)) { this->_errors->add(ErrorMemoryAllocationFailed); return; }
    }

    ntIndices = SharedPtr<HomogenNumericTableCPU<int, cpu> >(
        new HomogenNumericTableCPU<int, cpu>(*indices, size, 1));
    if (!ntIndices)
    { this->_errors->add(ErrorMemoryAllocationFailed); return; }
}

/**
 * Releases the array that contains the indices of objective function terms used for
 * stochastic gradient or sub-sampled Hessian matrix computation
 *
 * \param[in]  indices       Array that contains the indices provided by user
 *                           or memory for storing randomly generated indices
 * \param[in]  mtIndices     Micro-table that stores the indices
 * \param[in]  indicesStatus Status of the array of indices
 */
template<typename algorithmFPType, CpuType cpu>
void LBFGSTask<algorithmFPType, cpu>::releaseBatchIndices(int *indices,
            daal::internal::BlockMicroTable<int, readOnly, cpu> &mtIndices, IndicesStatus indicesStatus)
{
    if (indicesStatus == random)
    {
        daal_free(indices);
    }
}

} // namespace daal::internal

} // namespace lbfgs

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
