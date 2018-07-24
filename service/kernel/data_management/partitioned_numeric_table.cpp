/** file partitioned_numeric_table.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation
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

#include "service_defines.h"
#include "service_error_handling.h"
#include "data_management/data/partitioned_numeric_table.h"

namespace daal
{
namespace data_management
{
namespace interface1
{

using KeyType = PartitionedNumericTable::KeyType;

IMPLEMENT_SERIALIZABLE_TAG(PartitionedNumericTable, SERIALIZATION_PARTITIONED_NT_ID);

DAAL_EXPORT PartitionedNumericTable::PartitionedNumericTable() :
    NumericTable(0, 0),
    _mergedTable(MergedNumericTable::create()),
    _partitionsMap(new KeyValueDataCollection())
{
    super::_ddict = _mergedTable->getDictionarySharedPtr();
}

DAAL_EXPORT PartitionedNumericTable::PartitionedNumericTable(services::Status &st) :
    PartitionedNumericTable() { }

/* Function iterates over all partitions which keys are specified
 * by the indexer and executes the given operation on each partition */
template <typename Operation>
static services::Status iterateOver(const internal::Indexer<KeyType> &partitionsIndexer,
                                    bool threadingEnabled, const Operation &operation)
{
    const size_t size = partitionsIndexer.size();

    if (threadingEnabled)
    {
        SafeStatus safeStat;

        daal::threader_for(size, size, [&] (size_t i)
        {
            const KeyType partitionKey = partitionsIndexer[i];
            services::Status status = operation(i, partitionKey);
            DAAL_CHECK_STATUS_THR(status);
        });

        return safeStat.detach();
    }
    else
    {
        services::Status status;

        for (size_t i = 0; i < size; i++)
        {
            const KeyType partitionKey = partitionsIndexer[i];
            DAAL_CHECK_STATUS(status, operation(i, partitionKey));
        }

        return status;
    }
}

template <typename T>
DAAL_EXPORT services::Status PartitionedNumericTable::
    getPartitionedBlockIdx(const internal::Indexer<KeyType> &partitionsIndexer,
                           ReadWriteMode rwflag, PartitionedBlockDescriptor<T> &block)
{
    block.clear();

    const size_t requestedPartitionsNum = partitionsIndexer.size();
    services::Status status = block.reserve(requestedPartitionsNum);
    DAAL_CHECK_STATUS_VAR(status);

    /* Disable threading if rwflag is writeOnly. In case of writeOnly access
     * getBlockOfRows does not perform actuall data reading/conversion. */
    const bool threadingFlag = !(rwflag == writeOnly) &&
                                (requestedPartitionsNum > 1);

    status |= iterateOver(partitionsIndexer, threadingFlag,
        [&] (size_t index, const KeyType &key) -> services::Status
        {
            block.setIndexKeyPair(index, key);

            const NumericTablePtr partition = getPartition(key);
            DAAL_CHECK(partition, services::ErrorKeyIsNotFound);

            auto &blockForPartition = block.getBlockDescriptorByIndex(index);
            return partition->getBlockOfRows(0, getNumberOfRows(), rwflag, blockForPartition);
        });

    return status;
}

template <typename T>
DAAL_EXPORT services::Status PartitionedNumericTable::
    releasePartitionedBlockIdx(const internal::Indexer<KeyType> &partitionsIndexer,
                               PartitionedBlockDescriptor<T> &block)
{
    const size_t requestedPartitionsNum = partitionsIndexer.size();

    /* Disable threading if rwflag is readOnly. In case of readOnly access
     * releaseBlockOfRows does not perform actuall data writing. */
    const bool threadingFlag = !(block.getMode() == readOnly) &&
                                (requestedPartitionsNum > 1);

    return iterateOver(partitionsIndexer, threadingFlag,
        [&] (size_t index, const KeyType &key) -> services::Status
        {
            const NumericTablePtr partition = getPartition(key);
            DAAL_CHECK(partition, services::ErrorKeyIsNotFound);

            auto &blockForPartition = block.getBlockDescriptorByIndex(index);
            return partition->releaseBlockOfRows(blockForPartition);
        });
}

#define DAAL_INSTANTIATE_PARTITIONEDNUMERICTABLE(Type) \
    template DAAL_EXPORT services::Status PartitionedNumericTable::getPartitionedBlockIdx<Type>(     \
        const internal::Indexer<KeyType> &, ReadWriteMode, PartitionedBlockDescriptor<Type> &);      \
                                                                                                     \
    template DAAL_EXPORT services::Status PartitionedNumericTable::releasePartitionedBlockIdx<Type>( \
        const internal::Indexer<KeyType> &, PartitionedBlockDescriptor<Type> &);

DAAL_INSTANTIATE_PARTITIONEDNUMERICTABLE( int    );
DAAL_INSTANTIATE_PARTITIONEDNUMERICTABLE( float  );
DAAL_INSTANTIATE_PARTITIONEDNUMERICTABLE( double );

} // namespace interface1
} // namespace data_management
} // namespace daal
