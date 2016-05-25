/* file: lbfgs_base.h */
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

//++
//  Declaration of template function that computes LBFGS.
//--


#ifndef __LBFGS_BASE_H__
#define __LBFGS_BASE_H__

#include "lbfgs_batch.h"
#include "kernel.h"
#include "numeric_table.h"
#include "service_math.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

using namespace daal::data_management;

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
 * Statuses of the indices of objective function terms that are used for gradient and Hessian computation
 */
enum IndicesStatus
{
    random = 0,     /*!< Indices of the terms are generated randomly */
    user   = 1,     /*!< Indices of the terms are provided by user */
    all    = 2      /*!< All objective function terms are used for computations */
};

/**
 *  \brief Structure for storing data used in itermediate computations in LBFGS algorithm
 */
template<typename algorithmFPType, CpuType cpu>
struct LBFGSTask
{
    LBFGSTask(size_t argumentSize, size_t nTerms, size_t m,
            size_t batchSize, NumericTable *batchIndicesTable,
            size_t correctionPairBatchSize, NumericTable *correctionPairBatchIndicesTable,
            NumericTable *argumentTable, NumericTable *startValueTable,
            size_t nStepLength, NumericTable *stepLengthTable,
            services::SharedPtr<services::KernelErrorCollection> &_errors);

    LBFGSTask(size_t argumentSize, NumericTable *argumentTable, NumericTable *startValueTable,
            services::SharedPtr<services::KernelErrorCollection> &_errors);

    virtual ~LBFGSTask();

    /*
     * Sets the initial argument of objective function
     */
    void setStartArgument(size_t argumentSize, NumericTable *startValueTable);

    /*
     * Returns array of batch indices provided by user or the memory allcated for sampled batch indices
     */
    void getBatchIndices(size_t size, NumericTable *indicesTable, int **indices,
            daal::internal::BlockMicroTable<int, readOnly, cpu> &mtIndices, IndicesStatus *indicesStatusPtr,
            services::SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu> > &ntIndices);

    /*
     * Releases the memory allocated to store the batch indices
     */
    void releaseBatchIndices(int *indices,
            daal::internal::BlockMicroTable<int, readOnly, cpu> &mtIndices, IndicesStatus indicesStatus);

    IndicesStatus batchIndicesStatus;                /*!< Status of the objective function indices for gradient computation */
    IndicesStatus correctionPairBatchIndicesStatus;  /*!< Status of the objective function indices for Hessian computation */
    int *batchIndices;                       /*!< Array that contains the batch indices */
    int *correctionPairBatchIndices;         /*!< Array that contains the correction pair batch indices */
    algorithmFPType *argument;               /*!< Argument of the objective function. The optimized value */
    algorithmFPType *argumentLCur;           /*!< Average of objective function arguments for last L iterations. See formula (2.1) in [1] */
    algorithmFPType *argumentLPrev;          /*!< Average of objective function arguments for previous L iterations. See formula (2.1) in [1] */
    algorithmFPType *correctionS;            /*!< Array of correction pairs parts s(1), ..., s(m). See formula (2.1) in [1] */
    algorithmFPType *correctionY;            /*!< Array of correction pairs parts y(1), ..., y(m). See formula (2.2) in [1] */
    algorithmFPType *rho;                    /*!< Array of parameters rho of BFGS update. See formula (7.17) in [2] */
    algorithmFPType *alpha;                  /*!< Intermediate values used in two-loop recursion. See algorithm 7.4 in [2] */
    algorithmFPType *stepLength;             /*!< Array that stores step-length sequence */

    /** Micro-table that stores the work value */
    daal::internal::BlockMicroTable<algorithmFPType, writeOnly, cpu> mtArgument;
    /** Micro-table that stores the step-length sequence */
    daal::internal::BlockMicroTable<algorithmFPType, readOnly, cpu>  mtStepLength;

    /** Micro-table that stores the batch indices */
    daal::internal::BlockMicroTable<int, readOnly, cpu> mtBatchIndices;
    /** Micro-table that stores the correction pair batch indices */
    daal::internal::BlockMicroTable<int, readOnly, cpu> mtCorrectionPairBatchIndices;

    /** Numeric table that stores the batch indices */
    services::SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu> > ntBatchIndices;
    /** Numeric table that stores the correction pair batch indices */
    services::SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu> > ntCorrectionPairBatchIndices;
    /** Numeric table that stores the average of work values for last L iterations */
    services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > argumentLCurTable;

    /** Error collection of LBFGS algorithm */
    services::SharedPtr<services::KernelErrorCollection> _errors;
};

/**
 *  \brief Kernel for LBFGS computation
 *  for different floating point types of intermediate calculations and methods
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class LBFGSKernel {};

} // namespace daal::internal

} // namespace lbfgs

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
