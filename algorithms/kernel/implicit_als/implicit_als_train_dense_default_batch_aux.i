/* file: implicit_als_train_dense_default_batch_aux.i */
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
//  Auxiliary functions needed to train impicit ALS with fastCSR method
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_DENSE_DEFAULT_BATCH_AUX_I__
#define __IMPLICIT_ALS_TRAIN_DENSE_DEFAULT_BATCH_AUX_I__

#include "implicit_als_train_utils.h"

using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
ImplicitALSTrainTaskBase<algorithmFPType, cpu>::ImplicitALSTrainTaskBase(
            const NumericTable *dataTable, implicit_als::Model *initModel,
            implicit_als::Model *model, const Parameter *parameter,
            ImplicitALSTrainKernelBase<algorithmFPType, cpu> *algorithm) :
    mtItemsFactors(model->getItemsFactors().get()),
    mtUsersFactors(model->getUsersFactors().get()),
    nItems(dataTable->getNumberOfColumns()), nUsers(dataTable->getNumberOfRows()),
    itemsFactors(NULL), usersFactors(NULL), xtx(NULL), lhs(NULL),
    _algorithm(algorithm),
    nFactors(parameter->nFactors)
{
    daal::internal::BlockMicroTable<algorithmFPType, readOnly,  cpu> mtInitItemsFactors(
                initModel->getItemsFactors().get());
    algorithmFPType *initItemsFactors;

    size_t nRowsRead;
    nRowsRead = mtInitItemsFactors.getBlockOfRows(0, nItems, &initItemsFactors);
    if (nRowsRead < nItems)
    { _algorithm->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }

    nRowsRead = mtItemsFactors.getBlockOfRows(0, nItems, &itemsFactors);
    if (nRowsRead < nItems)
    { _algorithm->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }

    if (itemsFactors != initItemsFactors)
    {
        daal::services::daal_memcpy_s(itemsFactors,     nItems * nFactors * sizeof(algorithmFPType),
                                      initItemsFactors, nItems * nFactors * sizeof(algorithmFPType));
    }
    mtInitItemsFactors.release();

    nRowsRead = mtUsersFactors.getBlockOfRows(0, nUsers, &usersFactors);
    if (nRowsRead < nUsers)
    { _algorithm->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }

    xtx = (algorithmFPType *)daal::services::daal_malloc(nFactors * nFactors * sizeof(algorithmFPType));
    if (!xtx)
    { _algorithm->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    lhs = new daal::tls<algorithmFPType *>([ = ]() -> algorithmFPType*
    {
        return (algorithmFPType *)daal::services::daal_malloc(nFactors * nFactors * sizeof(algorithmFPType));
    });

    if (!lhs)
    {  _algorithm->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    if (!lhs->local())
    {  _algorithm->_errors->add(services::ErrorMemoryAllocationFailed); return; }
}

template <typename algorithmFPType, CpuType cpu>
ImplicitALSTrainTaskBase<algorithmFPType, cpu>::~ImplicitALSTrainTaskBase()
{
    if (itemsFactors) { mtItemsFactors.release(); }
    if (usersFactors) { mtUsersFactors.release(); }

    if (xtx) { daal::services::daal_free(xtx); }

    if (lhs)
    {
        lhs->reduce([](algorithmFPType* lhsData)
        {
            if(lhsData) { daal::services::daal_free(lhsData); }
        } );
        delete lhs;
    }
}

template <typename algorithmFPType, CpuType cpu>
ImplicitALSTrainTask<algorithmFPType, fastCSR, cpu>::ImplicitALSTrainTask(
            const NumericTable *dataTable, implicit_als::Model *initModel,
            implicit_als::Model *model, const Parameter *parameter,
            ImplicitALSTrainKernelBase<algorithmFPType, cpu> *algorithm) :
    ImplicitALSTrainTaskBase<algorithmFPType, cpu>(dataTable, initModel, model, parameter, algorithm),
    mtData(dataTable),
    data(NULL),  colIndices(NULL), rowOffsets(NULL),
    tdata(NULL), rowIndices(NULL), colOffsets(NULL)
{
    getData();
    if (!_algorithm->_errors->isEmpty()) { return; }

    size_t nNonNull = rowOffsets[nUsers] - rowOffsets[0];
    tdata = (algorithmFPType *)daal::services::daal_malloc(nNonNull * sizeof(algorithmFPType));
    rowIndices = (size_t *)daal::services::daal_malloc(nNonNull * sizeof(size_t));
    colOffsets = (size_t *)daal::services::daal_malloc((nUsers + 1) * sizeof(size_t));
    if (!tdata || !rowIndices || !colOffsets)
    { _algorithm->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    int errcode = csr2csc<algorithmFPType, cpu>(nUsers, nItems, data, colIndices, rowOffsets, tdata, rowIndices, colOffsets);
    if (errcode != 0)
    { _algorithm->_errors->add(services::ErrorMemoryAllocationFailed); return; }
}

template <typename algorithmFPType, CpuType cpu>
ImplicitALSTrainTask<algorithmFPType, fastCSR, cpu>::~ImplicitALSTrainTask()
{
    if (data) { mtData.release(); }

    if (tdata)      { daal::services::daal_free(tdata); }
    if (rowIndices) { daal::services::daal_free(rowIndices); }
    if (colOffsets) { daal::services::daal_free(colOffsets); }
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainTask<algorithmFPType, fastCSR, cpu>::getData()
{
    size_t nUsersRead = mtData.getSparseBlock(0, nUsers, &data, &colIndices, &rowOffsets);
    if (nUsersRead < nUsers)
    { _algorithm->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }
}


template <typename algorithmFPType, CpuType cpu>
ImplicitALSTrainTask<algorithmFPType, defaultDense, cpu>::ImplicitALSTrainTask(
            const NumericTable *dataTable, implicit_als::Model *initModel,
            implicit_als::Model *model, const Parameter *parameter,
            ImplicitALSTrainKernelBase<algorithmFPType, cpu> *algorithm) :
    ImplicitALSTrainTaskBase<algorithmFPType, cpu>(dataTable, initModel, model, parameter, algorithm),
    mtData(dataTable), data(NULL), tdata(NULL)
{
    getData();
    if (!_algorithm->_errors->isEmpty()) { return; }

    tdata = (algorithmFPType *)daal::services::daal_malloc(nItems * nUsers * sizeof(algorithmFPType));
    if (!tdata)
    { _algorithm->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    transpose(nUsers, nItems, data, tdata);
}

template <typename algorithmFPType, CpuType cpu>
ImplicitALSTrainTask<algorithmFPType, defaultDense, cpu>::~ImplicitALSTrainTask()
{
    if (data) { mtData.release(); }
    if (tdata) { daal::services::daal_free(tdata); }
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainTask<algorithmFPType, defaultDense, cpu>::getData()
{
    size_t nUsersRead = mtData.getBlockOfRows(0, nUsers, &data);
    if (nUsersRead < nUsers)
    { _algorithm->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainTask<algorithmFPType, defaultDense, cpu>::transpose(
            size_t nRows, size_t nCols, algorithmFPType *data, algorithmFPType *tdata)
{
    for (size_t i = 0; i < nRows; i++)
    {
        for (size_t j = 0; j < nCols; j++)
        {
            tdata[j * nRows + i] = data[i * nCols + j];
        }
    }
}

}
}
}
}
}

#endif
