/* file: lbfgs_dense_default_kernel.h */
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


#ifndef __LBFGS_DENSE_DEFAULT_KERNEL_H__
#define __LBFGS_DENSE_DEFAULT_KERNEL_H__

#include "lbfgs_base.h"
#include "service_rng.h"

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
 *  \brief Kernel for LBFGS computation
 */
template<typename algorithmFPType, CpuType cpu>
class LBFGSKernel<algorithmFPType, defaultDense, cpu> : public Kernel
{
public:
    void compute(iterative_solver::Input *input, iterative_solver::Result *result, Parameter *parameter);

protected:
    inline size_t mod(size_t a, size_t m) { return (a - (a / m) * m); }

    /*
     * Computes the dot product of two vectors
     */
    algorithmFPType dotProduct(size_t n, const algorithmFPType *x, const algorithmFPType *y);

    /*
     * Updates argument of the objective function
     */
    bool updateArgument(size_t t, size_t epoch, size_t m, size_t correctionIndex, size_t nTerms,
            size_t argumentSize, size_t batchSize, size_t nStepLength, algorithmFPType accuracyThreshold,
            services::SharedPtr<sum_of_functions::Batch> &gradientFunction,
            services::SharedPtr<NumericTable> &ntGradient, BlockDescriptor<algorithmFPType> &gradientBlock,
            LBFGSTask<algorithmFPType, cpu> &task, daal::internal::IntRng<int, cpu> &rng,
            algorithmFPType *argument);

    /*
     * Updates the array of objective function terms indices that are used in stochastic gradient
     * or Hessian matrix computations
     */
    void updateBatchIndices(size_t nTerms, size_t batchSize, int **batchIndices, IndicesStatus batchIndicesStatus,
            daal::internal::BlockMicroTable<int, readOnly, cpu> &mtBatchIndices,
            services::SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu> > &ntBatchIndices,
            daal::internal::IntRng<int,cpu> &rng);

    /*
     * Two-loop recursion algorithm that computes approximation of inverse Hessian matrix
     * multiplied by input gradient vector from a set of correction pairs (s(j), y(j)), j = 1,...,m.
     */
    void twoLoopRecursion(size_t argumentSize, size_t m, size_t correctionIndex, algorithmFPType *gradient,
                const algorithmFPType *correctionS, const algorithmFPType *correctionY, const algorithmFPType *rho,
                algorithmFPType *alpha);

    /*
     * Computes the correction pair (s, y) and the corresponding value rho
     */
    void computeCorrectionPair(size_t argumentSize, const algorithmFPType *argumentLCur, const algorithmFPType *argumentLPrev,
                algorithmFPType *hessian, algorithmFPType *s, algorithmFPType *y, algorithmFPType *rhoPtr);

    void setNIterations(NumericTable *nIterationsTable, size_t nIterations);
};

} // namespace daal::internal

} // namespace lbfgs

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
