/* file: iterative_solver_kernel.h */
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

//++
//  Declaration of template function that calculate iterative solver.
//--


#ifndef __ITERATIVE_SOLVER_KERNEL_H__
#define __ITERATIVE_SOLVER_KERNEL_H__

#include "kernel.h"
#include "service_rng.h"
#include "service_math.h"
#include "service_numeric_table.h"
#include "service_error_handling.h"
#include "data_management/data/memory_block.h"
#include "threading.h"
#include "uniform_kernel.h"
#include "uniform_impl.i"

using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace iterative_solver
{
namespace internal
{
/**
 *  \brief Kernel for iterative_solver calculation
 *  in case floating point type of intermediate calculations
 *  and method of calculations are different
 */

template<CpuType cpu, typename F>
void processByBlocks(size_t nRows, const F &processBlock, size_t minRowsNumInBlock = 1024, size_t blockStartThreshold = 5000)
{
    if(nRows < blockStartThreshold)// if number of rows is less that blockStartThreshold do sequential mode
    {
        processBlock(0, nRows);
        return;
    }

    size_t nBlocks = nRows / minRowsNumInBlock;
    nBlocks += (nBlocks * minRowsNumInBlock != nRows);

    daal::threader_for(nBlocks, nBlocks, [=](size_t block)
    {
        const size_t nRowsInBlock = (block == nBlocks - 1) ? (nRows - block * minRowsNumInBlock) : minRowsNumInBlock;
        processBlock(block * minRowsNumInBlock, nRowsInBlock);
    } );
}

template<typename algorithmFPType, CpuType cpu>
class IterativeSolverKernel : public Kernel
{
public:
    static services::Status vectorNorm(NumericTable *vecNT, algorithmFPType &res,
                                       size_t minRowsNumInBlock = 256, size_t blockStartThreshold = 5000)
    {
        res = 0;
        daal::tls<algorithmFPType *> normTls( [ = ]()-> algorithmFPType*
        {
            algorithmFPType *normPtr = (algorithmFPType *)daal_malloc(sizeof(algorithmFPType));
            *normPtr = 0;
            return normPtr;
        } );

        SafeStatus safeStat;
        processByBlocks<cpu>(vecNT->getNumberOfRows(), [=, &normTls, &safeStat](size_t startOffset, size_t nRowsInBlock)
        {
            WriteRows<algorithmFPType, cpu, NumericTable> vecBD;
            algorithmFPType *vecLocal = vecBD.set(vecNT, startOffset, nRowsInBlock);
            DAAL_CHECK_BLOCK_STATUS_THR(vecBD);
            algorithmFPType *normPtr = normTls.local();
            DAAL_CHECK_THR(normPtr, services::ErrorMemoryAllocationFailed);
            PRAGMA_VECTOR_ALWAYS
            PRAGMA_IVDEP
            for(int j = 0; j < nRowsInBlock; j++)
            {
                *normPtr += vecLocal[j] * vecLocal[j];
            }
        },
        minRowsNumInBlock, blockStartThreshold);
        normTls.reduce( [ =, &res ](algorithmFPType * normPtr)-> void
        {
            res += *normPtr;
            daal_free( normPtr );
        });
        res = daal::internal::Math<algorithmFPType, cpu>::sSqrt(res); // change to sqNorm
        return safeStat.detach();
    }

    static services::Status vectorNorm(const algorithmFPType *vec, size_t nElements, algorithmFPType &res,
                                       size_t minRowsNumInBlock = 256, size_t blockStartThreshold = 5000)
    {
        res = 0;
        if(nElements < blockStartThreshold)
        {
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for(size_t j = 0; j < nElements; j++)
            {
                res += vec[j] * vec[j];
            }
            res = daal::internal::Math<algorithmFPType, cpu>::sSqrt(res); // change to sqNorm
            return services::Status();
        }
        daal::tls<algorithmFPType *> normTls( [ = ]()-> algorithmFPType*
        {
            algorithmFPType *normPtr = (algorithmFPType *)daal_malloc(sizeof(algorithmFPType));
            *normPtr = 0;
            return normPtr;
        } );

        SafeStatus safeStat;
        processByBlocks<cpu>(nElements, [ =, &normTls, &safeStat](size_t startOffset, size_t nRowsInBlock)
        {
            algorithmFPType *normPtr = normTls.local();
            DAAL_CHECK_THR(normPtr, services::ErrorMemoryAllocationFailed);
            const algorithmFPType *vecLocal = &vec[startOffset];
            PRAGMA_VECTOR_ALWAYS
            for(int j = 0; j < nRowsInBlock; j++)
            {
                *normPtr += vecLocal[j] * vecLocal[j];
            }
        },
        minRowsNumInBlock, blockStartThreshold);

        normTls.reduce( [ =, &res ](algorithmFPType * normPtr)-> void
        {
            res += *normPtr;
            daal_free( normPtr );
        });
        res = daal::internal::Math<algorithmFPType, cpu>::sSqrt(res); // change to sqNorm
        return safeStat.detach();
    }

    static services::Status getRandom(int minVal, int maxVal, int *randomValue, int nRandomValues, engines::BatchBase &engine)
    {
        return distributions::uniform::internal::UniformKernel<int, distributions::uniform::defaultDense, cpu>::compute(
            minVal, maxVal, engine, nRandomValues, randomValue);
    }
};

template<typename algorithmFPType, CpuType cpu>
class RngTask
{
public:
    RngTask(const int *predefined, size_t size) : _predefined(predefined), _size(size),
        _maxVal(0), _values(0) {}

    bool init(int maxVal, engines::BatchBase &engine)
    {
        _engine = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(&engine);
        if(!_engine)
            return false;

        _values.reset(_size);
        if(!_values.get())
            return false;

        _maxVal = maxVal;
        return true;
    }

    services::Status get(const int *& pValues)
    {
        if(_predefined)
        {
            pValues = _predefined;
            _predefined += _size;
            return services::Status();
        }
        DAAL_CHECK(!RNGsType().uniformWithoutReplacement((int)_size, _values.get(), _engine->getState(), 0, _maxVal), ErrorIncorrectErrorcodeFromGenerator);

        pValues = _values.get();
        return Status();
    }

protected:
    typedef daal::internal::RNGs <int, cpu> RNGsType;
    const int *_predefined;
    size_t _size;
    TArray<int, cpu> _values;
    int _maxVal;
    daal::algorithms::engines::internal::BatchBaseImpl* _engine;
};

} // namespace daal::internal
} // namespace iterative_solver
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

#endif
