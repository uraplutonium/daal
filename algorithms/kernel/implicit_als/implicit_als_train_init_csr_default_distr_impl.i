/* file: implicit_als_train_init_csr_default_distr_impl.i */
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
//  Implementation of impicit ALS model initialization in distributed mode
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_INIT_CSR_DEFAULT_DISTR_IMPL_I__
#define __IMPLICIT_ALS_TRAIN_INIT_CSR_DEFAULT_DISTR_IMPL_I__

#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "service_memory.h"
#include "service_rng.h"
#include "service_spblas.h"
#include "service_sort.h"
#include "implicit_als_train_dense_default_batch_common.i"
#include "implicit_als_train_utils.h"

using namespace daal::services;
using namespace daal::services::internal;
using namespace daal::internal;

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
template <CpuType cpu>
class Partition
{
public:
    Partition(NumericTable *partitionTable, size_t fullNUsers)
    {
        size_t nRows = partitionTable->getNumberOfRows();
        partitionRows.set(partitionTable, 0, nRows);
        if (nRows > 1)
        {
            nParts = nRows - 1;
            partition = const_cast<int *>(partitionRows.get());
        }
        else
        {
            nParts = (size_t)((partitionRows.get())[0]);
            partitionPtr.reset(nParts + 1);
            partition = partitionPtr.get();
            size_t nUsersInPart = fullNUsers / nParts;
            partition[0] = 0;
            for (size_t i = 1; i < nParts; i++)
            {
                partition[i] = partition[i - 1] + nUsersInPart;
            }
            partition[nParts] = fullNUsers;
        }
    }

    int *get()
    {
        return partition;
    }

    size_t nParts;
private:
    ReadRows<int, cpu> partitionRows;
    TArray<int, cpu> partitionPtr;
    int *partition;
};

template <typename algorithmFPType, CpuType cpu>
services::Status ImplicitALSInitDistrKernel<algorithmFPType, fastCSR, cpu>::compute(
            const NumericTable *dataTable, const NumericTable *partitionTable,
            NumericTable **dataParts, NumericTable **blocksToLocal,
            NumericTable **userOffsets, NumericTable *itemsFactorsTable, const Parameter *parameter)
{
    size_t nItems = dataTable->getNumberOfRows();
    size_t nUsers = dataTable->getNumberOfColumns();
    size_t fullNUsers = parameter->fullNUsers;
    ReadRowsCSR<algorithmFPType, cpu> mtData(
        dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(dataTable)), 0, nItems);

    const algorithmFPType *tdata = mtData.values();
    const size_t *rowIndices = mtData.cols();
    const size_t *colOffsets = mtData.rows();

    size_t partitionNRows = partitionTable->getNumberOfRows();
    Partition<cpu> partitionObj(const_cast<NumericTable *>(partitionTable), fullNUsers);
    size_t nParts = partitionObj.nParts;
    int *partition = partitionObj.get();

    computeOffsets(nParts, partition, userOffsets);

    /* Split input data table into sub-parts using the partition */
    transposeAndSplitCSRTable(nItems, fullNUsers, tdata, rowIndices, colOffsets,
        nParts, partition, dataParts);

    if (0 != computeBlocksToLocal(nItems, fullNUsers, rowIndices, colOffsets,
        nParts, partition, blocksToLocal))
    { this->_errors->add(ErrorMemoryAllocationFailed); DAAL_RETURN_STATUS(); }

    WriteRows<algorithmFPType, cpu> partialFactors(itemsFactorsTable, 0, nItems);
    algorithmFPType *itemsFactors = partialFactors.get();

    /* Initialize item factors */
    size_t seed = parameter->seed;
    size_t nFactors = parameter->nFactors;
    computePartialFactors(nUsers, nItems, nFactors, parameter->fullNUsers, seed,
                          tdata, rowIndices, colOffsets, itemsFactors);

    BaseRNGs<cpu> baseRng(seed);
    RNGs<int, cpu> rng;

    TArray<int, cpu> randBuffer(nFactors);
    if (!randBuffer.get()) { this->_errors->add(ErrorIncorrectNumberOfRowsInInputNumericTable); DAAL_RETURN_STATUS(); }

    for (size_t i = 0; i < nItems; i++)
    {
        this->randFactors(1, nFactors, itemsFactors + i * nFactors, randBuffer.get(), rng, baseRng, this->_errors.get());
    }
    DAAL_RETURN_STATUS();
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSInitDistrKernelBase<algorithmFPType, fastCSR, cpu>::computeOffsets(
        size_t nParts, const int *partition, NumericTable **offsets)
{
    for (size_t i = 0; i < nParts; i++)
    {
        WriteRows<int, cpu> offsetRows(offsets[i], 0, 1);
        int *offset = offsetRows.get();
        offset[0] = partition[i];
    }
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSInitDistrKernel<algorithmFPType, fastCSR, cpu>::transposeAndSplitCSRTable(size_t nItems, size_t fullNUsers,
            const algorithmFPType *tdata, const size_t *rowIndices, const size_t *colOffsets,
            size_t nParts, const int *partition, NumericTable **dataParts)
{
    size_t nValues = colOffsets[nItems] - colOffsets[0];

    TArray<size_t, cpu> rowOffsetsPtr(fullNUsers + 1);
    TArray<size_t, cpu> colIndicesPtr(nValues);
    TArray<algorithmFPType, cpu> dataPtr(nValues);
    size_t *rowOffsets = rowOffsetsPtr.get();
    size_t *colIndices = colIndicesPtr.get();
    algorithmFPType *data = dataPtr.get();
    if (!rowOffsets || !colIndices || !data)
    { this->_errors->add(ErrorMemoryAllocationFailed); return; }

    int errcode = training::internal::csr2csc<algorithmFPType, cpu>(fullNUsers, nItems, tdata, rowIndices, colOffsets, data, colIndices, rowOffsets);
    if (errcode != 0)
    { this->_errors->add(ErrorMemoryAllocationFailed); return; }

    for (size_t i = 0; i < nParts; i++)
    {
        size_t nRowsPart = partition[i + 1] - partition[i];
        size_t nValuesPart = rowOffsets[partition[i + 1]] - rowOffsets[partition[i]];
        CSRNumericTable *dataPartTable = static_cast<CSRNumericTable *>(dataParts[i]);
        dataPartTable->allocateDataMemory(nValuesPart);
        WriteRowsCSR<algorithmFPType, cpu> dataPartRows(dataPartTable, 0, nRowsPart);
        size_t *rowOffsetsPart = const_cast<size_t *>(dataPartRows.rows());
        size_t *colIndicesPart = const_cast<size_t *>(dataPartRows.cols());
        algorithmFPType *dataPart = dataPartRows.values();
        if (!rowOffsetsPart || !colIndicesPart || !dataPart)
        { this->_errors->add(ErrorMemoryAllocationFailed); return; }

        size_t rowOffsetDiff = rowOffsets[partition[i]] - 1;
        for (size_t j = 0; j < nRowsPart + 1; j++)
        {
            rowOffsetsPart[j] = rowOffsets[j + partition[i]] - rowOffsetDiff;
        }
        size_t offset = rowOffsets[partition[i]] - 1;
        for (size_t j = 0; j < nValuesPart; j++)
        {
            colIndicesPart[j] = colIndices[j + offset];
            dataPart[j]       = data[j + offset];
        }
    }
}

template <typename algorithmFPType, CpuType cpu>
int ImplicitALSInitDistrKernelBase<algorithmFPType, fastCSR, cpu>::computeBlocksToLocal(
            size_t nItems, size_t fullNUsers,
            const size_t *rowIndices, const size_t *colOffsets,
            size_t nParts, const int *partition, NumericTable **blocksToLocal)
{
    const int memError = 1;
    TArray<bool, cpu> blockFlagsPtr(nItems * nParts);
    bool *blockFlags = blockFlagsPtr.get();
    if (!blockFlags)
    { return memError; }

    for (size_t i = 0; i < nItems; i++)
    {
        for (size_t k = 1; k < nParts + 1; k++)
        {
            blockFlags[(k - 1) * nItems + i] = false;
            for (size_t j = colOffsets[i] - 1; j < colOffsets[i + 1] - 1; j++)
            {
                if (partition[k - 1] <= rowIndices[j] - 1 && rowIndices[j] - 1 < partition[k])
                {
                    blockFlags[(k - 1) * nItems + i] = true;
                }
            }
        }
    }

    TArray<size_t, cpu> blocksToLocalSizePtr(nParts);
    size_t *blocksToLocalSize = blocksToLocalSizePtr.get();
    if (!blocksToLocalSize)
    { return memError; }
    for (size_t i = 0; i < nParts; i++)
    {
        blocksToLocalSize[i] = 0;
        for (size_t j = 0; j < nItems; j++)
        {
            blocksToLocalSize[i] += (blockFlags[i * nItems + j] ? 1 : 0);
        }
    }
    for (size_t i = 0; i < nParts; i++)
    {
        blocksToLocal[i]->resize(blocksToLocalSize[i]);

        WriteRows<int, cpu> blocksToLocalRows(blocksToLocal[i], 0, blocksToLocalSize[i]);
        int *blocksToLocalData = blocksToLocalRows.get();
        size_t indexId = 0;

        for (size_t j = 0; j < nItems; j++)
        {
            if (blockFlags[i * nItems + j])
            {
                blocksToLocalData[indexId++] = (int)j;
            }
        }
    }

    return 0;
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSInitDistrKernel<algorithmFPType, fastCSR, cpu>::computePartialFactors(
            size_t nUsers, size_t nItems, size_t nFactors, size_t fullNUsers, size_t seed,
            const algorithmFPType *tdata, const size_t *rowIndices, const size_t *colOffsets, algorithmFPType *partialFactors)
{
    algorithmFPType one = 1.0;
    size_t bufSz = (nItems > nFactors ? nItems : nFactors);

    TArray<algorithmFPType, cpu> onesPtr(nUsers);
    TArray<algorithmFPType, cpu> itemsSumPtr(bufSz);
    algorithmFPType *ones = onesPtr.get();
    algorithmFPType *itemsSum = itemsSumPtr.get();
    if (!ones || !itemsSum) { this->_errors->add(ErrorMemoryAllocationFailed); return; }

    service_memset<algorithmFPType, cpu>(ones, one, nUsers);

    /* Parameters of CSRMV function */
    char transa = 'N';
    algorithmFPType alpha = 1.0;
    algorithmFPType beta  = 0.0;
    char matdescra[6] = {'\0', '\0', '\0', '\0', '\0', '\0'};
    matdescra[0] = 'G';        // general matrix
    matdescra[3] = 'F';        // 1-based indexing

    /* Compute sum of columns of input matrix */
    SpBlas<algorithmFPType, cpu>::xcsrmv(&transa, (DAAL_INT *)&nItems, (DAAL_INT *)&nUsers, &alpha, matdescra,
                        tdata, (DAAL_INT *)rowIndices, (DAAL_INT *)colOffsets, (DAAL_INT *)(colOffsets + 1),
                        ones, &beta, itemsSum);

    algorithmFPType invFullNUsers = one / (algorithmFPType)fullNUsers;
    for (size_t i = 0; i < nItems; i++)
    {
        partialFactors[i * nFactors] = itemsSum[i] * invFullNUsers;
    }
}

template <typename algorithmFPType, CpuType cpu>
services::Status ImplicitALSInitDistrStep2Kernel<algorithmFPType, fastCSR, cpu>::compute(
            size_t nParts,
            NumericTable **dataParts, NumericTable *dataTable, NumericTable **blocksToLocal,
            NumericTable **itemOffsets)
{
    size_t nRows = dataTable->getNumberOfRows();
    size_t nCols = dataTable->getNumberOfColumns();
    CSRNumericTable *csrDataTable = dynamic_cast<CSRNumericTable *>(dataTable);

    size_t nValues = 0;
    for (size_t i = 0; i < nParts; i++)
    {
        nValues += dynamic_cast<CSRNumericTable *>(dataParts[i])->getDataSize();
    }
    csrDataTable->allocateDataMemory(nValues);
    WriteRowsCSR<algorithmFPType, cpu> dataTableRows(csrDataTable, 0, nRows);
    algorithmFPType *data = dataTableRows.values();
    size_t *rowOffsets = const_cast<size_t *>(dataTableRows.rows());
    size_t *colIndices = const_cast<size_t *>(dataTableRows.cols());

    mergeCSRTables(nParts, dataParts, nRows, data, rowOffsets, colIndices);

    TArray<int, cpu> partitionPtr(nParts + 1);
    int *partition = partitionPtr.get();
    partition[0] = 0;
    for (size_t i = 1; i < nParts + 1; i++)
    {
        partition[i] = partition[i - 1] + dataParts[i-1]->getNumberOfColumns();
    }
    computeOffsets(nParts, partition, itemOffsets);

    if (0 != computeBlocksToLocal(nRows, nCols, colIndices, rowOffsets,
        nParts, partition, blocksToLocal))
    { this->_errors->add(ErrorMemoryAllocationFailed); DAAL_RETURN_STATUS(); }
    DAAL_RETURN_STATUS();
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSInitDistrStep2Kernel<algorithmFPType, fastCSR, cpu>::mergeCSRTables(
            size_t nParts, NumericTable **dataParts, size_t nRows, algorithmFPType *data,
            size_t *rowOffsets, size_t *colIndices)
{
    TArray<ReadRowsCSR<algorithmFPType, cpu>, cpu> dataPartTables(nParts);
    TArray<const algorithmFPType*, cpu> dataPart(nParts);
    TArray<const size_t*, cpu> rowOffsetsPart(nParts);
    TArray<const size_t*, cpu> colIndicesPart(nParts);

    for (size_t p = 0; p < nParts; p++)
    {
        dataPartTables[p].set(dynamic_cast<CSRNumericTableIface *>(dataParts[p]), 0, nRows);

        dataPart[p] = dataPartTables[p].values();
        rowOffsetsPart[p] = dataPartTables[p].rows();
        colIndicesPart[p] = dataPartTables[p].cols();
    }
    rowOffsets[0] = 1;

    for (size_t i = 1; i < nRows + 1; i++)
    {
        rowOffsets[i] = rowOffsets[i - 1];
        for (size_t p = 0; p < nParts; p++)
        {
            rowOffsets[i] += (rowOffsetsPart[p][i] - rowOffsetsPart[p][i - 1]);
        }
    }

    TArray<size_t, cpu> colIndicesOffsets(nParts);
    colIndicesOffsets[0] = 0;
    for (size_t i = 1; i < nParts; i++)
    {
        colIndicesOffsets[i] = colIndicesOffsets[i - 1] + dataParts[i - 1]->getNumberOfColumns();
    }

    for (size_t i = 1; i < nRows + 1; i++)
    {
        size_t fullNValues = rowOffsets[i] - rowOffsets[i - 1];
        TArray<size_t, cpu> colIndicesBufferPtr(fullNValues);
        TArray<algorithmFPType, cpu> dataBufferPtr(fullNValues);
        size_t *colIndicesBuffer = colIndicesBufferPtr.get();
        algorithmFPType *dataBuffer = dataBufferPtr.get();
        for (size_t p = 0; p < nParts; p++)
        {
            size_t startCol = rowOffsetsPart[p][i - 1] - 1;
            size_t nValues = rowOffsetsPart[p][i] - rowOffsetsPart[p][i - 1];
            for (size_t j = 0; j < nValues; j++)
            {
                colIndicesBuffer[j] = colIndicesPart[p][startCol + j] + colIndicesOffsets[p];
                dataBuffer[j] = dataPart[p][startCol + j];
            }
            colIndicesBuffer += nValues;
            dataBuffer       += nValues;
        }
        colIndicesBuffer = colIndicesBufferPtr.get();
        dataBuffer = dataBufferPtr.get();
        algorithms::internal::qSort<size_t, algorithmFPType, cpu>(fullNValues, colIndicesBuffer, dataBuffer);
        daal_memcpy_s(colIndices + rowOffsets[i - 1] - 1, fullNValues * sizeof(size_t),
                      colIndicesBuffer,                   fullNValues * sizeof(size_t));
        daal_memcpy_s(data + rowOffsets[i - 1] - 1, fullNValues * sizeof(algorithmFPType),
                      dataBuffer,                   fullNValues * sizeof(algorithmFPType));
    }
}

}
}
}
}
}
}

#endif
