/* file: svm_train_kernel.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Declaration of template structs that calculate SVM Training functions.
//--
*/

#ifndef __SVM_TRAIN_KERNEL_H__
#define __SVM_TRAIN_KERNEL_H__

#include "numeric_table.h"
#include "model.h"
#include "daal_defines.h"
#include "svm_train_types.h"
#include "kernel.h"
#include "service_numeric_table.h"

using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;

#include "svm_train_boser_cache.i"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace internal
{

enum SVMVectorStatus
{
    free    = 0x0,
    up      = 0x1,
    low     = 0x2,
    shrink  = 0x4
};

template <typename algorithmFPType, CpuType cpu>
struct SVMTrainTask
{
    static const size_t kernelFunctionBlockSize = 1024; /* Size of the block of kernel function elements */

    SVMTrainTask(size_t nVectors) : _cache(nullptr), _nVectors(nVectors){}

    Status setup(const Parameter& svmPar, const NumericTablePtr& xTable, NumericTable& yTable);

    /* Perform Sequential Minimum Optimization (SMO) algorithm to find optimal coefficients alpha */
    Status compute(const Parameter& svmPar);

    /* Write support vectors and classification coefficients into model */
    Status setResultsToModel(const NumericTable& xTable, Model& model, algorithmFPType C) const;

    ~SVMTrainTask();

protected:
    Status init(algorithmFPType C);

    inline void updateI(algorithmFPType C, size_t index);

    bool findMaximumViolatingPair(size_t nActiveVectors, algorithmFPType tau, int& Bi, int& Bj,
        algorithmFPType& delta, algorithmFPType& ma, algorithmFPType& Ma, algorithmFPType& curEps, Status& s) const;

    Status reconstructGradient(size_t& nActiveVectors);

    algorithmFPType WSSi(size_t nActiveVectors, int& Bi) const;

    void WSSjLocalBaseline(const size_t jStart, const size_t jEnd,
        const algorithmFPType *KiBlock, const algorithmFPType GMax, const algorithmFPType Kii, const algorithmFPType tau,
        int &Bj, algorithmFPType &GMin, algorithmFPType &GMin2, algorithmFPType &delta) const;

    void WSSjLocal(const size_t jStart, const size_t jEnd,
        const algorithmFPType *KiBlock, const algorithmFPType GMax, const algorithmFPType Kii, const algorithmFPType tau,
        int &Bj, algorithmFPType &GMin, algorithmFPType &GMin2, algorithmFPType &delta) const;

    Status WSSj(size_t nActiveVectors, algorithmFPType tau, int Bi, algorithmFPType GMax, int& Bj,
        algorithmFPType& delta, algorithmFPType& res) const;

    Status update(size_t nActiveVectors, algorithmFPType C, int Bi, int Bj, algorithmFPType delta);

    size_t updateShrinkingFlags(size_t nActiveVectors, algorithmFPType C, algorithmFPType ma, algorithmFPType Ma);

    /*** Methods used in shrinking ***/
    size_t doShrink(size_t nActiveVectors);

    /**
    * \brief Write support vectors and classification coefficients into output model
    */
    Status setSVCoefficients(size_t nSV, Model& model) const;
    Status setSVIndices(size_t nSV, Model& model) const;
    Status setSV_Dense(Model& model, const NumericTable& xTable, size_t nSV) const;
    Status setSV_CSR(Model& model, const NumericTable& xTable, size_t nSV) const;
    algorithmFPType calculateBias(algorithmFPType C) const;

    inline void updateAlpha(algorithmFPType C, int Bi, int Bj, algorithmFPType delta,
        algorithmFPType& newDeltai, algorithmFPType& newDeltaj);

protected:
    const size_t _nVectors; //Number of observations in the input data set
    TArray<algorithmFPType, cpu> _y; //Array of class labels
    TArray<algorithmFPType, cpu> _alpha; //Array of classification coefficients
    TArray<algorithmFPType, cpu> _grad; //Objective function gradient
    TArray<algorithmFPType, cpu> _kernelDiag; //diagonal elements of the matrix Q (kernel(x[i], x[i]))
    TArray<char, cpu> _I; // array of flags I_LOW and I_UP
    SVMCacheIface<algorithmFPType, cpu> *_cache; //caches matrix Q (kernel(x[i], x[j])) values
};

template <Method method, typename algorithmFPType, CpuType cpu>
struct SVMTrainImpl : public Kernel
{
    services::Status compute(const NumericTablePtr& xTable, NumericTable& yTable,
        daal::algorithms::Model *r, const daal::algorithms::Parameter *par);
};

} // namespace internal

} // namespace training

} // namespace svm

} // namespace algorithms

} // namespace daal

#endif
