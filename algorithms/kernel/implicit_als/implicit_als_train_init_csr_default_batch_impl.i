/* file: implicit_als_train_init_csr_default_batch_impl.i */
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
//  Implementation of defaultDense method for impicit ALS initialization
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_INIT_CSR_DEFAULT_BATCH_IMPL_I__
#define __IMPLICIT_ALS_TRAIN_INIT_CSR_DEFAULT_BATCH_IMPL_I__

#include "service_memory.h"
#include "service_spblas.h"
#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{
namespace internal
{

using namespace daal::services::internal;
using namespace daal::internal;


template <typename algorithmFPType, CpuType cpu>
services::Status ImplicitALSInitKernel<algorithmFPType, fastCSR, cpu>::computeSumByColumnsCSR(const algorithmFPType *data, const size_t *colIndices, const size_t *rowOffsets, const size_t nUsers, const size_t nItems,
    const size_t nFactors, algorithmFPType* const itemsFactors, algorithmFPType* const itemsSum, algorithmFPType* const notNullElemSum, const bool oneAsBase)
{
    const size_t notNullElem = rowOffsets[nUsers] - rowOffsets[0];

    const size_t nThreads = threader_get_threads_number();
    const size_t nBlocks = (nFactors < nThreads ? nFactors : nThreads);
    const size_t blockSize = notNullElem/nBlocks;

    TArray<algorithmFPType*, cpu> itemsSumB(nBlocks);
    algorithmFPType** arrSum = itemsSumB.get();
    DAAL_CHECK_MALLOC(arrSum);

    daal::threader_for(nBlocks, nBlocks, [&](size_t i)
    {
        algorithmFPType* s = itemsFactors + nItems * i;
        arrSum[i] = s;
        services::internal::service_memset_seq<algorithmFPType, cpu>(s, algorithmFPType(0.0), nItems);

        const size_t low = blockSize * i;
        const size_t high = ((i != nBlocks-1) ? blockSize * (i+1) : notNullElem);

        if (oneAsBase) s--; // shift to align with corrrect indexing
        for(size_t j = low; j < high; ++j)
        {
            s[colIndices[j]] += data[j];
        }
    });

    services::Status st = reduceSumByColumns(arrSum, nItems, nBlocks, itemsSum);

    daal::threader_for(nBlocks, nBlocks, [&](size_t i)
    {
        algorithmFPType* s = arrSum[i];
        services::internal::service_memset_seq<algorithmFPType, cpu>(s, algorithmFPType(0.0), nItems);

        const size_t low = blockSize * i;
        const size_t high = ((i != nBlocks-1) ? blockSize * (i+1) : notNullElem);

        if (oneAsBase) s--; // shift to align with corrrect indexing
        for(size_t j = low; j < high; ++j)
        {
            s[colIndices[j]]++;
        }
    });

    st |= reduceSumByColumns(arrSum, nItems, nBlocks, notNullElemSum);

    return st;
}

template <typename algorithmFPType, CpuType cpu>
services::Status ImplicitALSInitKernel<algorithmFPType, fastCSR, cpu>::reduceSumByColumns(algorithmFPType** arrSum, const size_t nItems, const size_t nBlocks, algorithmFPType* const arrForReduce)
{
    services::internal::service_memset_seq<algorithmFPType, cpu>(arrForReduce, algorithmFPType(0.0), nItems);

    const size_t nThreads = threader_get_threads_number();
    const size_t nBlocksForReduction = nThreads;
    const size_t blockSizeForReduction = nItems/nBlocksForReduction;

    daal::threader_for(nBlocksForReduction, nBlocksForReduction, [&](size_t iBlock)
    {
        const size_t start = blockSizeForReduction * iBlock;
        const size_t end = ((iBlock != nBlocksForReduction-1) ? blockSizeForReduction * (iBlock+1) : nItems);

        for(size_t k = 0; k < nBlocks; ++k)
        {
            algorithmFPType* const s = arrSum[k];

            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for(size_t j = start; j < end; ++j)
            {
                arrForReduce[j] += s[j];
            }
        }
    });

    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status ImplicitALSInitKernel<algorithmFPType, fastCSR, cpu>::compute(
                 const NumericTable *dataTable, NumericTable *itemsFactorsTable, NumericTable* usersFactorsTable, const Parameter *parameter, engines::BatchBase &engine)
{
    const size_t nUsers = dataTable->getNumberOfRows();
    const size_t nItems = dataTable->getNumberOfColumns();
    const size_t nFactors = parameter->nFactors;

    const size_t bufSz = (nItems > nFactors ? nItems : nFactors);
    TArray<algorithmFPType, cpu> itemsSumArr(bufSz);
    algorithmFPType* const itemsSum = itemsSumArr.get();
    DAAL_CHECK_MALLOC(itemsSum);

    TArray<algorithmFPType, cpu> notNullElemSumArr(bufSz);
    algorithmFPType* const notNullElemSum = notNullElemSumArr.get();
    DAAL_CHECK_MALLOC(notNullElemSum);

    WriteOnlyRows<algorithmFPType, cpu> mtItemsFactors(itemsFactorsTable, 0, nItems);
    algorithmFPType* itemsFactors = mtItemsFactors.get();
    DAAL_CHECK_MALLOC(itemsFactors);

    const CSRNumericTableIface* csrIface = dynamic_cast<const CSRNumericTableIface *>(dataTable);
    DAAL_CHECK(csrIface, ErrorIncorrectInputNumericTable);
    ReadRowsCSR<algorithmFPType, cpu> mtData(*const_cast<CSRNumericTableIface *>(csrIface), 0, nUsers);
    DAAL_CHECK_BLOCK_STATUS(mtData);
    const algorithmFPType *data = mtData.values();
    const size_t *colIndices = mtData.cols();
    const size_t *rowOffsets = mtData.rows();

    const bool oneAsBase = rowOffsets[0] == 1;

    auto s = computeSumByColumnsCSR(data, colIndices, rowOffsets, nUsers, nItems, nFactors, itemsFactors, itemsSum, notNullElemSum, oneAsBase);

    s |= this->randFactors(nItems, nFactors, itemsFactors, engine);

    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nItems; i++) // if number of not null elems is equal 0
    {
        notNullElemSum[i] = (notNullElemSum[i] == algorithmFPType(0.0) ? algorithmFPType(1.0) : notNullElemSum[i]);
    }

    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nItems; i++)
    {
        itemsFactors[i * nFactors] = itemsSum[i] / notNullElemSum[i];
    }

    return s;
}

}
}
}
}
}
}

#endif
