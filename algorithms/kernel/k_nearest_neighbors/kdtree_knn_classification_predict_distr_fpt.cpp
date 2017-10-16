/* file: kdtree_knn_classification_predict_distr_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
* All Rights Reserved.
*
* If this  software was obtained  under the  Intel Simplified  Software License,
* the following terms apply:
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*
*
* If this  software  was obtained  under the  Apache License,  Version  2.0 (the
* "License"), the following terms apply:
*
* You may  not use this  file except  in compliance  with  the License.  You may
* obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
*
*
* Unless  required  by   applicable  law  or  agreed  to  in  writing,  software
* distributed under the License  is distributed  on an  "AS IS"  BASIS,  WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*
* See the   License  for the   specific  language   governing   permissions  and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of the interface for K-Nearest Neighbors (kNN) model-based prediction
//--
*/

#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace prediction
{
namespace interface1
{

using namespace daal::data_management;
using namespace daal::services;

template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep1::allocate(const daal::algorithms::Input * input,
                                                                     const daal::algorithms::Parameter * parameter, int method)
{
    const DistributedInput<step1Local> * const in = static_cast<const DistributedInput<step1Local> *>(input);
    const size_t numberOfRows = in->getNumberOfRows();
    set(keys, NumericTablePtr(new HomogenNumericTable<algorithmFPType>(1, numberOfRows, NumericTable::doAllocate)));
    return services::Status();
}

template DAAL_EXPORT services::Status DistributedPartialResultStep1::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                           const daal::algorithms::Parameter * parameter,
                                                                                           int method);

template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep2::allocate(const daal::algorithms::Input * input,
                                                                     const daal::algorithms::Parameter * parameter, int method)
{
    const DistributedInput<step2Local> * const in = static_cast<const DistributedInput<step2Local> *>(input);
    const Parameter * const par = static_cast<const Parameter *>(parameter);
    const size_t numberOfRows = in->getNumberOfRows();
    const size_t numberOfColumns = in->getNumberOfColumns();
    set(prediction, NumericTablePtr(new HomogenNumericTable<algorithmFPType>(1, numberOfRows, NumericTable::doAllocate)));

    const KeyValueDataCollectionPtr queryCollection(new KeyValueDataCollection());
    const KeyValueDataCollectionPtr respCollection(new KeyValueDataCollection());

    const KeyValueDataCollectionConstPtr perNodeCommResponses = in->get(communicationResponses);
    const size_t nodeCount = perNodeCommResponses ? perNodeCommResponses->size() : 0;
    for (size_t i = 0; i < nodeCount; ++i)
    {
        (*queryCollection)[perNodeCommResponses->getKeyByIndex(i)] =
            NumericTablePtr(new HomogenNumericTable<algorithmFPType>(2 + numberOfColumns, 0, NumericTable::doNotAllocate));
    }

    const KeyValueDataCollectionConstPtr perNodeCommInputQueries = in->get(communicationInputQueries);
    const int keyValue = in->get(key);
    const size_t inputQueryCount = perNodeCommInputQueries ? perNodeCommInputQueries->size() : 0;
    for (size_t i = 0; i < inputQueryCount; ++i)
    {
        (*respCollection)[perNodeCommInputQueries->getKeyByIndex(i)] =
            NumericTablePtr(new HomogenNumericTable<algorithmFPType>(2 + 3 * par->k, 0, NumericTable::doNotAllocate));
    }
    (*respCollection)[keyValue] = NumericTablePtr(new HomogenNumericTable<algorithmFPType>(2 + 3 * par->k, 0, NumericTable::doNotAllocate));

    set(communicationQueries, queryCollection);
    set(communicationOutputResponses, respCollection);

    return services::Status();
}

template DAAL_EXPORT services::Status DistributedPartialResultStep2::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                                           const daal::algorithms::Parameter * parameter,
                                                                                           int method);

} // namespace interface1
} // namespace prediction
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal
