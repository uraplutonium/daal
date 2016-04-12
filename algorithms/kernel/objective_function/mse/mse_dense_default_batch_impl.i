/* file: mse_dense_default_batch_impl.i */
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
//  Implementation of mse algorithm
//--
*/

#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "service_blas.h"

using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace mse
{
namespace internal
{

/**
 *  \brief Kernel for mse objective function calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
inline void MSEKernel<algorithmFPType, method, cpu>::compute(
    Input *input,
    objective_function::Result *result,
    Parameter *parameter)
{
    BlockMicroTable<algorithmFPType, readOnly, cpu> mtData(input->get(data).get());
    size_t nTheta = mtData.getFullNumberOfColumns();
    const algorithmFPType zero = 0.0;
    algorithmFPType *xMultTheta = NULL;

    algorithmFPType *argumentArray, *dependentVariablesArray, *theta, *theta0, *data, *gradient = NULL;

    BlockMicroTable<algorithmFPType, readOnly, cpu> mtArgument(input->get(argument).get());
    size_t nFeatures = mtArgument.getFullNumberOfColumns();
    mtArgument.getBlockOfRows(0, 1, &argumentArray);
    theta0 = &argumentArray[0];
    theta = &argumentArray[1];

    BlockMicroTable<algorithmFPType, writeOnly, cpu> *mtGradient = NULL;
    BlockMicroTable<algorithmFPType, writeOnly, cpu> *mtValue = NULL;
    BlockMicroTable<algorithmFPType, writeOnly, cpu> *mtHessian = NULL;
    algorithmFPType *value = NULL;
    bool valueFlag = ((parameter->resultsToCompute & objective_function::value) != 0) ? true : false;
    if(valueFlag)
    {
        mtValue = new BlockMicroTable<algorithmFPType, writeOnly, cpu>(result->get(objective_function::resultCollection,
                                                                                   objective_function::valueIdx).get());
        mtValue->getBlockOfRows(0, 1, &value);
        value[0] = zero;
    }

    algorithmFPType *hessian = NULL;
    bool hessianFlag = ((parameter->resultsToCompute & objective_function::hessian) != 0) ? true : false;
    if(hessianFlag)
    {
        mtHessian = new BlockMicroTable<algorithmFPType, writeOnly, cpu>(result->get(objective_function::resultCollection,
                                                                                     objective_function::hessianIdx).get());
        mtHessian->getBlockOfRows(0, nFeatures, &hessian);
        for(size_t j = 0; j < nFeatures * nFeatures; j++)
        {
            hessian[j] = zero;
        }
    }
    bool gradientFlag = ((parameter->resultsToCompute & objective_function::gradient) != 0) ? true : false;
    if (gradientFlag)
    {
        mtGradient = new BlockMicroTable<algorithmFPType, writeOnly, cpu>(result->get(objective_function::resultCollection,
                                                                                      objective_function::gradientIdx).get());
        mtGradient->getBlockOfRows(0, 1, &gradient);
        for(size_t j = 0; j < nFeatures; j++)
        {
            gradient[j] = zero;
        }
    }

    NumericTable *indices = parameter->batchIndices.get();
    size_t batchSize = 0;
    if(indices != NULL)
    {
        batchSize = indices->getNumberOfColumns();
    }

    BlockMicroTable<algorithmFPType, readOnly, cpu> mtDependentVariables(input->get(dependentVariables).get());
    if(batchSize != 0)
    {
        size_t blockSizeDefault = 512; // max number of data rows in processed block
        size_t blockSize = blockSizeDefault;
        size_t nBlocks = batchSize / blockSizeDefault;
        nBlocks += (nBlocks * blockSizeDefault != batchSize);
        if(nBlocks == 1) { blockSize = batchSize; }

        algorithmFPType *memory  = (algorithmFPType *) daal_malloc((blockSize * nTheta + 2 * blockSize) * sizeof(algorithmFPType));
        if(!memory) { this->_errors->add(ErrorMemoryAllocationFailed); return; }

        algorithmFPType *blockData = memory;
        algorithmFPType *blockDependentVariables = &memory[blockSize * nTheta];
        xMultTheta = &blockDependentVariables[blockSize];

        int *indicesArray = NULL;
        BlockMicroTable<int, readOnly, cpu> mtIndices(indices);
        mtIndices.getBlockOfRows(0, 1, &indicesArray);
        size_t index;
        for(size_t block = 0; block < nBlocks; block++)
        {
            if( block == nBlocks - 1 )
            {
                blockSize = batchSize - block * blockSizeDefault;
            }

            for(size_t idx = 0; idx < blockSize; idx++)
            {
                index = indicesArray[blockSizeDefault * block + idx];
                mtData.getBlockOfRows(index, 1, &data);
                mtDependentVariables.getBlockOfRows(index, 1, &dependentVariablesArray);

                for(size_t j = 0; j < nTheta; j++)
                {
                    blockData[idx * nTheta + j] = data[j];
                }
                blockDependentVariables[idx] = dependentVariablesArray[0];

                mtData.release();
                mtDependentVariables.release();
            }

            computeMSE(blockSize,  nTheta, valueFlag, hessianFlag, gradientFlag, blockData, theta, theta0,
                       blockDependentVariables, value, gradient, hessian, xMultTheta);
        }
        mtIndices.release();
        daal_free(memory);
    }
    else
    {
        batchSize = mtData.getFullNumberOfRows();

        xMultTheta = (algorithmFPType *) daal_malloc(batchSize * sizeof(algorithmFPType));
        if(!xMultTheta) { this->_errors->add(ErrorMemoryAllocationFailed); return; }

        mtData.getBlockOfRows(0, batchSize, &data);
        mtDependentVariables.getBlockOfRows(0, batchSize, &dependentVariablesArray);

        computeMSE(batchSize, nTheta, valueFlag, hessianFlag, gradientFlag, data, theta, theta0,
                   dependentVariablesArray, value, gradient, hessian, xMultTheta);

        mtData.release();
        mtDependentVariables.release();
        daal_free(xMultTheta);
    }
    mtArgument.release();

    const algorithmFPType one = 1.0;
    algorithmFPType batchSizeInv = (algorithmFPType)one / batchSize;
    if (gradientFlag)
    {
        for(size_t j = 0; j < nFeatures; j++)
        {
            gradient[j] *= batchSizeInv;
        }
    }

    if (valueFlag) {value[0] /= (algorithmFPType)(2 * batchSize);}
    if (hessianFlag)
    {
        hessian[0] = one;
        for(size_t j = 1; j < nFeatures * nFeatures; j++)
        {
            hessian[j] *= batchSizeInv;
        }
    }

    if (valueFlag)
    {
        mtValue->release();
        delete mtValue;
    }
    if (hessianFlag)
    {
        mtHessian->release();
        delete mtHessian;
    }
    if (gradientFlag)
    {
        mtGradient->release();
        delete mtGradient;
    }
    return;
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline void MSEKernel<algorithmFPType, method, cpu>::computeMSE(
    size_t blockSize, size_t nTheta, bool valueFlag, bool hessianFlag, bool gradientFlag,
    algorithmFPType *data,
    algorithmFPType *theta,
    algorithmFPType *theta0,
    algorithmFPType *dependentVariablesArray,
    algorithmFPType *value,
    algorithmFPType *gradient,
    algorithmFPType *hessian,
    algorithmFPType *xMultTheta)
{
    char trans = 'T';
    algorithmFPType one = 1.0;
    algorithmFPType zero = 0.0;
    MKL_INT n   = (MKL_INT)blockSize;
    MKL_INT dim = (MKL_INT)nTheta;
    MKL_INT ione = 1;

    if (gradientFlag || valueFlag)
    {
        Blas<algorithmFPType, cpu>::xgemv(&trans, &dim, &n, &one, data, &dim, theta, &ione, &zero, xMultTheta, &ione);

        for(size_t i = 0; i < blockSize; i++)
        {
            xMultTheta[i] = xMultTheta[i] + theta0[0] - dependentVariablesArray[i];
        }
    }

    if (gradientFlag)
    {
        for(size_t i = 0; i < blockSize; i++)
        {
            gradient[0] += xMultTheta[i];
            for(size_t j = 0; j < nTheta; j++)
            {
                gradient[j + 1] += xMultTheta[i] * data[i * nTheta + j];
            }
        }
    }
    if (valueFlag)
    {
        for(size_t i = 0; i < blockSize; i++)
        {
            value[0] += xMultTheta[i] * xMultTheta[i];
        }
    }
    if (hessianFlag)
    {
        char uplo  = 'U';
        char notrans = 'N';
        MKL_INT nFeatures = dim + 1;

        Blas<algorithmFPType, cpu>::xsyrk(&uplo, &notrans, &dim, &n, &one, data, &dim, &one,
                           hessian + nFeatures + 1, &nFeatures);

        for (size_t i = 0; i < blockSize; i++)
        {
            for (size_t j = 0; j < nTheta; j++)
            {
                hessian[j + 1] += data[i * nTheta + j];
            }
        }

        for (size_t i = 0; i < nFeatures; i++)
        {
            for (size_t j = 1; j < i; j++)
            {
                hessian[j * nFeatures + i] = hessian[i * nFeatures + j];
            }
            hessian[i * nFeatures] = hessian[i];
        }
    }
}

} // namespace daal::internal

} // namespace mse

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal
