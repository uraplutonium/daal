/* file: kmeans_init_impl.h */
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
//  Implementation of kmeans init classes.
//--
*/

#include "algorithms/kmeans/kmeans_init_types.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace internal
{

enum LocalData
{
    numberOfClusters,       //Number of clusters (candidates) selected so far
    closestClusterDistance, //Distance from every row in the input data to its closest cluster (candidate)
    closestCluster,         //parallelPlus only. Index of the closest cluster (candidate) for every row in the input data
    candidateRating,        //parallelPlus only. For each candidate number of closest points among input data
    localDataSize
};

#define isParallelPlusMethod(method) ((method == kmeans::init::parallelPlusDense) || (method == kmeans::init::parallelPlusCSR))

services::Status checkLocalData(const data_management::DataCollection* pInput, const Parameter* par, const char* dataName,
    const data_management::NumericTable* pData, bool bParallelPlus);

}
} // namespace init
} // namespace kmeans
} // namespace algorithm
} // namespace daal
