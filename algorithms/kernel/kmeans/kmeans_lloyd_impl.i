/* file: kmeans_lloyd_impl.i */
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
//  Implementation of auxiliary functions used in Lloyd method
//  of K-means algorithm.
//--
*/

#include "service_memory.h"
#include "service_micro_table.h"

#include "threading.h"
#include "service_blas.h"
#include "service_spblas.h"

using namespace daal::services::internal;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace internal
{

template<typename interm, CpuType cpu>
struct task
{
    int      dim;
    int      clNum;
    interm * cCenters;
    daal::tls<interm*> * goalFunc;
    daal::tls<int   *> * cS0;
    daal::tls<interm*> * cS1;

    int max_block_size;

    interm* clSq;
    daal::tls<interm*> * mkl_buff;
};

template<typename interm, CpuType cpu>
void * kmeansInitTask(int dim, int clNum, interm * centroids)
{
    struct task<interm,cpu> * t;
    t = (task<interm,cpu> *)daal::services::daal_malloc(sizeof(struct task<interm,cpu>));

    t->dim       = dim;
    t->clNum     = clNum;
    t->cCenters  = centroids;
    t->goalFunc  = new daal::tls<interm*>( [=]()-> interm* { return service_calloc<interm,cpu>(1);         } );
    t->cS0       = new daal::tls<int   *>( [=]()-> int   * { return service_calloc<int,cpu>(clNum);        } );
    t->cS1       = new daal::tls<interm*>( [=]()-> interm* { return service_calloc<interm,cpu>(clNum*dim); } );

    t->max_block_size = 512;

    t->clSq      = service_calloc<interm,cpu>(clNum);
    for(size_t k=0;k<clNum;k++)
    {
        for(size_t j=0;j<dim;j++)
        {
            t->clSq[k] += centroids[k*dim + j]*centroids[k*dim + j] * 0.5;
        }
    }
    t->mkl_buff  = new daal::tls<interm*>( [=]()-> interm* { return service_calloc<interm,cpu>(t->max_block_size*clNum); } );

    void * task_id;
    *(size_t*)(&task_id) = (size_t)t;

    return task_id;
}

template<typename interm, CpuType cpu, int assignFlag>
void addNTToTaskThreadedDense(void * task_id, const NumericTable * ntData, interm *catCoef, NumericTable * ntAssign = 0 )
{
    struct task<interm,cpu> * t = static_cast<task<interm,cpu> *>(task_id);

    size_t n = ntData->getNumberOfRows();

    size_t blockSizeDeafult = t->max_block_size;

    size_t nBlocks = n / blockSizeDeafult;
    nBlocks += (nBlocks*blockSizeDeafult != n);

    daal::threader_for( nBlocks, nBlocks, [=](int k)
    {
        size_t blockSize = blockSizeDeafult;
        if( k == nBlocks-1 )
        {
            blockSize = n - k*blockSizeDeafult;
        }

        BlockDescriptor<int> assignBlock;

        BlockMicroTable<interm, readOnly,  cpu> mtData( ntData );
        interm* data;

        int*    cS0 = t->cS0->local();
        interm* cS1 = t->cS1->local();
        interm* trg = t->goalFunc->local();

        mtData.getBlockOfRows( k*blockSizeDeafult, blockSize, &data );

        int* assignments = 0;

        if(assignFlag)
        {
            ntAssign->getBlockOfRows( k*blockSizeDeafult, blockSize, writeOnly, assignBlock );
            assignments = assignBlock.getBlockPtr();
        }

        size_t p = t->dim;
        size_t nClusters = t->clNum;

        interm* x_clusters = t->mkl_buff->local();
        interm* inClusters = t->cCenters;
        interm* clustersSq = t->clSq;

        char transa = 't';
        char transb = 'n';
        MKL_INT _m = nClusters;
        MKL_INT _n = blockSize;
        MKL_INT _k = p;
        interm alpha = 1.0;
        MKL_INT lda = p;
        MKL_INT ldy = p;
        interm beta = 0.0;
        MKL_INT ldaty = nClusters;

        Blas<interm, cpu>::xxgemm(&transa, &transb, &_m, &_n, &_k, &alpha, inClusters,
            &lda, data, &ldy, &beta, x_clusters, &ldaty);

        for (size_t i = 0; i < blockSize; i++)
        {
            interm minGoalVal = clustersSq[0] - x_clusters[i*nClusters];
            size_t minIdx = 0;

            for (size_t j = 0; j < nClusters; j++)
            {
                if( minGoalVal > clustersSq[j] - x_clusters[i*nClusters + j] )
                {
                    minGoalVal = clustersSq[j] - x_clusters[i*nClusters + j];
                    minIdx = j;
                }
            }

            minGoalVal *= 2.0;

            for (size_t j = 0; j < p; j++)
            {
                cS1[minIdx * p + j] += data[i*p + j];
                minGoalVal += data[ i*p + j ] * data[ i*p + j ];
            }

            *trg += minGoalVal;

            cS0[minIdx]++;

            if(assignFlag)
            {
                assignments[i] = (int)minIdx;
            }
        }

        if(assignFlag)
        {
            ntAssign->releaseBlockOfRows( assignBlock );
        }
        mtData.release();
    } );
}

template<typename interm, CpuType cpu, int assignFlag>
void addNTToTaskThreadedCSR(void * task_id, const NumericTable * ntDataGen, interm *catCoef, NumericTable * ntAssign = 0 )
{
    CSRNumericTableIface *ntData  = dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(ntDataGen));

    struct task<interm,cpu> * t = static_cast<task<interm,cpu> *>(task_id);

    size_t n = ntDataGen->getNumberOfRows();

    size_t blockSizeDeafult = t->max_block_size;

    size_t nBlocks = n / blockSizeDeafult;
    nBlocks += (nBlocks*blockSizeDeafult != n);

    daal::threader_for( nBlocks, nBlocks, [=](int k)
    {
        size_t blockSize = blockSizeDeafult;
        if( k == nBlocks-1 )
        {
            blockSize = n - k*blockSizeDeafult;
        }

        BlockDescriptor<int> assignBlock;
        CSRBlockDescriptor<interm> dataBlock;

        ntData->getSparseBlock( k*blockSizeDeafult, blockSize, readOnly, dataBlock );

        interm *data        = dataBlock.getBlockValuesPtr();
        size_t *colIdx      = dataBlock.getBlockColumnIndicesPtr();
        size_t *rowIdx      = dataBlock.getBlockRowIndicesPtr();

        int*    cS0 = t->cS0->local();
        interm* cS1 = t->cS1->local();
        interm* trg = t->goalFunc->local();

        int* assignments = 0;

        if(assignFlag)
        {
            ntAssign->getBlockOfRows( k*blockSizeDeafult, blockSize, writeOnly, assignBlock );
            assignments = assignBlock.getBlockPtr();
        }

        size_t p = t->dim;
        size_t nClusters = t->clNum;

        interm* x_clusters = t->mkl_buff->local();
        interm* inClusters = t->cCenters;
        interm* clustersSq = t->clSq;

        {
            char transa = 'n';
            MKL_INT _n = blockSize;
            MKL_INT _p = p;
            MKL_INT _c = nClusters;
            interm alpha = 1.0;
            interm beta  = 0.0;
            MKL_INT ldaty = blockSize;
            char matdescra[6] = {'G',0,0,'F',0,0};

            SpBlas<interm, cpu>::xxcsrmm(&transa, &_n, &_c, &_p, &alpha, matdescra,
                         data, (MKL_INT*)colIdx, (MKL_INT*)rowIdx,
                         inClusters, &_p, &beta, x_clusters, &_n);
        }

        size_t csrCursor=0;
        for (size_t i = 0; i < blockSize; i++)
        {
            interm minGoalVal = clustersSq[0] - x_clusters[i];
            size_t minIdx = 0;

            for (size_t j = 0; j < nClusters; j++)
            {
                if( minGoalVal > clustersSq[j] - x_clusters[i + j*blockSize] )
                {
                    minGoalVal = clustersSq[j] - x_clusters[i + j*blockSize];
                    minIdx = j;
                }
            }

            minGoalVal *= 2.0;

            size_t valuesNum = rowIdx[i+1]-rowIdx[i];
            for (size_t j = 0; j < valuesNum; j++)
            {
                cS1[minIdx * p + colIdx[csrCursor]-1] += data[csrCursor];
                minGoalVal += data[csrCursor]*data[csrCursor];
                csrCursor++;
            }

            *trg += minGoalVal;

            cS0[minIdx]++;

            if(assignFlag)
            {
                assignments[i] = (int)minIdx;
            }
        }

        if(assignFlag)
        {
            ntAssign->releaseBlockOfRows( assignBlock );
        }

        ntData->releaseSparseBlock(dataBlock);
    } );
}

template<Method method, typename interm, CpuType cpu, int assignFlag>
void addNTToTaskThreaded(void * task_id, const NumericTable * ntData, interm *catCoef, NumericTable * ntAssign = 0 )
{
    if(method == lloydDense)
    {
        addNTToTaskThreadedDense<interm,cpu,assignFlag>( task_id, ntData, catCoef, ntAssign );
    }
    else if(method == lloydCSR)
    {
        addNTToTaskThreadedCSR<interm,cpu,assignFlag>( task_id, ntData, catCoef, ntAssign );
    }
}

template<Method method, typename interm, CpuType cpu>
void getNTAssignmentsThreaded(void * task_id, const NumericTable * ntData, const NumericTable * ntAssign, interm *catCoef )
{
    struct task<interm,cpu> * t = static_cast<task<interm,cpu> *>(task_id);

    size_t n = ntData->getNumberOfRows();

    size_t blockSizeDeafult = t->max_block_size;

    size_t nBlocks = n / blockSizeDeafult;
    nBlocks += (nBlocks*blockSizeDeafult != n);

    daal::threader_for( nBlocks, nBlocks, [=](int k)
    {
        size_t blockSize = blockSizeDeafult;
        if( k == nBlocks-1 )
        {
            blockSize = n - k*blockSizeDeafult;
        }

        BlockMicroTable<interm, readOnly,  cpu> mtData( ntData );
        BlockMicroTable<int   , writeOnly, cpu> mtAssign( ntAssign );
        interm* data;
        int*    assign;

        mtData  .getBlockOfRows( k*blockSizeDeafult, blockSize, &data   );
        mtAssign.getBlockOfRows( k*blockSizeDeafult, blockSize, &assign );

        size_t p = t->dim;
        size_t nClusters = t->clNum;

        interm* x_clusters = t->mkl_buff->local();
        interm* inClusters = t->cCenters;
        interm* clustersSq = t->clSq;

        char transa = 't';
        char transb = 'n';
        MKL_INT _m = nClusters;
        MKL_INT _n = blockSize;
        MKL_INT _k = p;
        interm alpha = 1.0;
        MKL_INT lda = p;
        MKL_INT ldy = p;
        interm beta = 0.0;
        MKL_INT ldaty = nClusters;

        Blas<interm, cpu>::xxgemm(&transa, &transb, &_m, &_n, &_k, &alpha, inClusters,
            &lda, data, &ldy, &beta, x_clusters, &ldaty);

        for (size_t i = 0; i < blockSize; i++)
        {
            interm minGoalVal = clustersSq[0] - x_clusters[i*nClusters];
            size_t minIdx = 0;

            for (size_t j = 0; j < nClusters; j++)
            {
                if( minGoalVal > clustersSq[j] - x_clusters[i*nClusters + j] )
                {
                    minGoalVal = clustersSq[j] - x_clusters[i*nClusters + j];
                    minIdx = j;
                }
            }

            assign[i] = minIdx;
        }

        mtAssign.release();
        mtData.release();
    } );
}

template<typename interm, CpuType cpu>
int kmeansUpdateCluster(void * task_id, int jidx, interm *s1)
{
    int i, j;
    struct task<interm,cpu> * t = static_cast<task<interm,cpu> *>(task_id);

    int idx   = (int)jidx;
    int dim   = t->dim;
    int clNum = t->clNum;

    int s0=0;

    t->cS0->reduce( [&](int *v)-> void
    {
        s0 += v[idx];
    } );

    t->cS1->reduce( [=](interm *v)-> void
    {
        int j;
      PRAGMA_IVDEP
        for(j=0;j<dim;j++)
        {
            s1[j] += v[idx*dim + j];
        }
    } );

    return s0;
}

template<typename interm, CpuType cpu>
void kmeansClearClusters(void * task_id, interm *goalFunc)
{
    int i, j;
    struct task<interm,cpu> * t = static_cast<task<interm,cpu> *>(task_id);

    if( t->clNum != 0)
    {
        t->cS0->reduce( [=](int *v)-> void
        {
            daal::services::daal_free( v );
        } );
        delete t->cS0;

        t->cS1->reduce( [=](interm *v)-> void
        {
            daal::services::daal_free( v );
        } );
        delete t->cS1;

        t->mkl_buff->reduce( [=](interm *v)-> void
        {
            daal::services::daal_free( v );
        } );
        delete t->mkl_buff;

        daal::services::daal_free( t->clSq );

        t->clNum = 0;

        if( goalFunc!= 0 )
        {
            *goalFunc = (interm)(0.0);

            t->goalFunc->reduce( [=](interm *v)-> void
            {
                (*goalFunc) += (*v);
            } );
        }
        t->goalFunc->reduce( [=](interm *v)-> void
        {
            daal::services::daal_free( v );
        } );
        delete t->goalFunc;
    }

    daal::services::daal_free(t);
}

} // namespace daal::algorithms::kmeans::internal
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
