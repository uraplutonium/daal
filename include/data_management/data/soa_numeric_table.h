/* file: soa_numeric_table.h */
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
//  Implementation of a heterogeneous table stored as a structure of arrays.
//--
*/

#ifndef __SOA_NUMERIC_TABLE_H__
#define __SOA_NUMERIC_TABLE_H__

namespace daal
{
namespace data_management
{

namespace interface1
{
/**
 * @ingroup numeric_tables
 * @{
 */
/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__SOANUMERICTABLE"></a>
 *  \brief Class that provides methods to access data stored as a structure of arrays,
 *         where each (contiguous) array represents values corresponding to a specific feature.
 */
class SOANumericTable : public NumericTable
{
public:

    /**
     *  Constructor for an empty Numeric Table
     *  \param[in]  nColumns      Number of columns in the table
     *  \param[in]  nRows         Number of rows in the table
     *  \param[in]  featuresEqual Flag that makes all features in the NumericTableDictionary equal
     */
    SOANumericTable( size_t nColumns = 0, size_t nRows = 0, DictionaryIface::FeaturesEqual featuresEqual = DictionaryIface::notEqual ):
        NumericTable(nColumns, nRows), _arrays(0), _arraysFeaturesCapacity(0), _arraysInitialized(0), _partialMemStatus(notAllocated)
    {
        _layout = soa;

        if( !resizePointersArray(nColumns) )
        {
            this->_status.add(services::ErrorMemoryAllocationFailed);
            return;
        }
    }

    /**
     *  Constructor for an empty Numeric Table with a predefined NumericTableDictionary
     *  \param[in]  ddict                 Pointer to the predefined NumericTableDictionary
     *  \param[in]  nRows                 Number of rows in the table
     *  \param[in]  memoryAllocationFlag  Flag that controls internal memory allocation for data in the numeric table
     */
    SOANumericTable( NumericTableDictionary *ddict, size_t nRows, AllocationFlag memoryAllocationFlag = notAllocate ):
        NumericTable(ddict), _arrays(0), _arraysFeaturesCapacity(0), _arraysInitialized(0), _partialMemStatus(notAllocated)
    {
        _layout = soa;
        this->_status |= setNumberOfRowsImpl( nRows );
        if( !resizePointersArray( getNumberOfColumns() ) )
        {
            this->_status.add(services::ErrorMemoryAllocationFailed);
            return;
        }
        if( memoryAllocationFlag == doAllocate )
        {
            this->_status |= allocateDataMemoryImpl();
        }
    }

    virtual ~SOANumericTable()
    {
        freeDataMemoryImpl();

        if( _arrays != 0 )
        {
            daal::services::daal_free(_arrays);
        }
    }

    virtual int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return SERIALIZATION_SOA_NT_ID;
    }

    /**
     *  Sets a pointer to an array of values for a given feature
     *  \tparam T       Type of feature values
     *  \param[in]  ptr Pointer to the array of the T type that stores feature values
     *  \param[in]  idx Feature index
     */
    template<typename T>
    services::Status setArray(T *ptr, size_t idx)
    {
        if( _partialMemStatus!=notAllocated && _partialMemStatus!=userAllocated )
        {
            return services::Status(services::ErrorIncorrectNumberOfFeatures);
        }

        if( idx < getNumberOfColumns() && idx < _arraysFeaturesCapacity )
        {
            _ddict->setFeature<T>(idx);

            if( _arrays[idx]==0 && ptr!=0 )
            {
                _arraysInitialized++;
            }

            if( _arrays[idx]!=0 && ptr==0 )
            {
                _arraysInitialized--;
            }

            _arrays[idx] = (void *)ptr;
        }
        else
        {
            return services::Status(services::ErrorIncorrectNumberOfFeatures);
        }

        _partialMemStatus = userAllocated;

        if(_arraysInitialized == getNumberOfColumns())
        {
            _memStatus = userAllocated;
        }
        return services::Status();
    }

    /**
     *  Returns a pointer to an array of values for a given feature
     *  \param[in]  idx Feature index
     *  \return Pointer to the array of values
     */
    void *getArray(size_t idx)
    {
        if( idx < _ddict->getNumberOfFeatures() )
        {
            return _arrays[idx];
        }
        else
        {
            this->_status.add(services::ErrorIncorrectNumberOfFeatures);
            return (void *)0;
        }
    }

    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    services::Status releaseBlockOfRows(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<double>(block);
    }
    services::Status releaseBlockOfRows(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<float>(block);
    }
    services::Status releaseBlockOfRows(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<int>(block);
    }

    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                  ReadWriteMode rwflag, BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    services::Status releaseBlockOfColumnValues(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return releaseTFeature<double>(block);
    }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return releaseTFeature<float>(block);
    }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return releaseTFeature<int>(block);
    }

    DAAL_DEPRECATED_VIRTUAL services::Status setDictionary( NumericTableDictionary *ddict ) DAAL_C11_OVERRIDE
    {
        services::Status s;
        DAAL_CHECK_STATUS(s, NumericTable::setDictionary( ddict ));

        size_t ncol = ddict->getNumberOfFeatures();

        if( !resizePointersArray( ncol ) )
        {
            return services::Status(services::ErrorMemoryAllocationFailed);
        }
        return s;
    }


    void serializeImpl  (InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<InputDataArchive, false>( arch );}

    void deserializeImpl(OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<OutputDataArchive, true>( arch );}

protected:
    void **_arrays;
    size_t _arraysFeaturesCapacity;
    size_t _arraysInitialized;
    MemoryStatus _partialMemStatus;

    bool resizePointersArray(size_t nColumns)
    {
        if( _arraysFeaturesCapacity >= nColumns )
        {
            size_t counter=0;
            for(size_t i=0; i<nColumns; i++)
            {
                counter += (_arrays[i] != 0);
            }
            _arraysInitialized = counter;

            if( _arraysInitialized == nColumns )
            {
                _memStatus = _partialMemStatus;
            }
            else
            {
                _memStatus = notAllocated;
            }

            return true;
        }

        void** oldArrays = _arrays;

        _arrays = (void **)daal::services::daal_malloc( sizeof(void *) * nColumns );

        if( _arrays == 0 )
        {
            _arrays = oldArrays;
            return false;
        }

        for(size_t i = 0; i < _arraysFeaturesCapacity; i++)
        {
            _arrays[i] = oldArrays[i];
        }

        for(size_t i = _arraysFeaturesCapacity; i < nColumns; i++)
        {
            _arrays[i] = 0;
        }

        if(oldArrays)
        {
            daal::services::daal_free( oldArrays );
        }

        _arraysFeaturesCapacity = nColumns;
        _memStatus = notAllocated;

        return true;
    }

    services::Status setNumberOfColumnsImpl(size_t ncol) DAAL_C11_OVERRIDE
    {
        services::Status s;
        DAAL_CHECK_STATUS(s, NumericTable::setNumberOfColumnsImpl(ncol));

        if( !resizePointersArray( ncol ) )
        {
            return services::Status(services::ErrorMemoryAllocationFailed);
        }
        return s;
    }

    services::Status allocateDataMemoryImpl(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE
    {
        freeDataMemoryImpl();

        size_t ncol = _ddict->getNumberOfFeatures();
        size_t nrows = getNumberOfRows();

        if( ncol * nrows == 0 )
        {
            if( nrows == 0 )
            {
                return services::Status(services::ErrorIncorrectNumberOfObservations);
            }
            else
            {
                return services::Status(services::ErrorIncorrectNumberOfFeatures);
            }
        }

        for(size_t i = 0; i < ncol; i++)
        {
            NumericTableFeature f = (*_ddict)[i];
            if( f.typeSize != 0 )
            {
                _arrays[i] = daal::services::daal_malloc( f.typeSize * nrows );
                _arraysInitialized++;
            }
            if( _arrays[i] == 0 )
            {
                freeDataMemoryImpl();
                return services::Status();
            }
        }

        if(_arraysInitialized > 0)
        {
            _partialMemStatus = internallyAllocated;
        }

        if(_arraysInitialized == ncol)
        {
            _memStatus = internallyAllocated;
        }
        return services::Status();
    }

    void freeDataMemoryImpl() DAAL_C11_OVERRIDE
    {
        if( _partialMemStatus == internallyAllocated )
        {
            size_t ncol = getNumberOfColumns();
            for(size_t i = 0; i < _arraysFeaturesCapacity; i++)
            {
                if( _arrays[i] )
                {
                    daal::services::daal_free(_arrays[i]);
                    _arrays[i] = 0;
                }
            }
        }

        for(size_t i = 0; i < _arraysFeaturesCapacity; i++)
        {
            _arrays[i] = 0;
        }
        _arraysInitialized = 0;

        _partialMemStatus = notAllocated;
        _memStatus = notAllocated;
    }

    template<typename Archive, bool onDeserialize>
    void serialImpl( Archive *arch )
    {
        NumericTable::serialImpl<Archive, onDeserialize>( arch );

        if( onDeserialize )
        {
            allocateDataMemoryImpl();
        }

        size_t ncol = _ddict->getNumberOfFeatures();
        size_t nrows = getNumberOfRows();

        for(size_t i = 0; i < ncol; i++)
        {
            NumericTableFeature f = (*_ddict)[i];
            void *ptr = getArray(i);

            arch->set( (char *)ptr, nrows * f.typeSize );
        }
    }

private:

    template <typename T>
    services::Status getTBlock( size_t idx, size_t nrows, ReadWriteMode rwFlag, BlockDescriptor<T>& block )
    {
        size_t ncols = getNumberOfColumns();
        size_t nobs = getNumberOfRows();
        block.setDetails( 0, idx, rwFlag );

        if (idx >= nobs)
        {
            block.resizeBuffer( ncols, 0 );
            return services::Status();
        }

        nrows = ( idx + nrows < nobs ) ? nrows : nobs - idx;

        if( !block.resizeBuffer( ncols, nrows ) )
        {
            return services::Status(services::ErrorMemoryAllocationFailed);
        }

        if( !(block.getRWFlag() & (int)readOnly) ) return services::Status();

        T lbuf[32];

        size_t di = 32;

        T* buffer = block.getBlockPtr();

        for( size_t i = 0 ; i < nrows ; i += di )
        {
            if( i + di > nrows ) { di = nrows - i; }

            for( size_t j = 0 ; j < ncols ; j++ )
            {
                NumericTableFeature &f = (*_ddict)[j];

                char *ptr = (char *)_arrays[j] + (idx + i) * f.typeSize;

                data_feature_utils::getVectorUpCast(f.indexType, data_feature_utils::getInternalNumType<T>())
                ( di, ptr, lbuf );

                for( size_t ii = 0 ; ii < di; ii++ )
                {
                    buffer[ (i + ii)*ncols + j ] = lbuf[ii];
                }
            }
        }
        return services::Status();
    }

    template <typename T>
    services::Status releaseTBlock( size_t idx, size_t nrows, T *buf, ReadWriteMode rwFlag )
    {
        if (rwFlag & (int)writeOnly)
        {
            size_t ncols = getNumberOfColumns();
            T lbuf[32];

            size_t i, ii, j;

            size_t di = 32;

            for( i = 0 ; i < nrows ; i += di )
            {
                if( i + di > nrows ) { di = nrows - i; }

                for( j = 0 ; j < ncols ; j++ )
                {
                    NumericTableFeature &f = (*_ddict)[j];

                    char *ptr = (char *)_arrays[j];
                    char *location = ptr + (idx + i) * f.typeSize;

                    for( ii = 0 ; ii < di; ii++ )
                    {
                        lbuf[ii] = buf[ (i + ii) * ncols + j ];
                    }

                    data_feature_utils::getVectorDownCast(f.indexType, data_feature_utils::getInternalNumType<T>())
                    ( di, lbuf, location );
                }
            }
        }
        return services::Status();
    }

    template <typename T>
    services::Status releaseTBlock( BlockDescriptor<T>& block )
    {
        if(block.getRWFlag() & (int)writeOnly)
        {
            size_t ncols = getNumberOfColumns();
            size_t nrows = block.getNumberOfRows();
            size_t idx   = block.getRowsOffset();
            T lbuf[32];

            size_t di = 32;

            T* blockPtr = block.getBlockPtr();

            for( size_t i = 0 ; i < nrows ; i += di )
            {
                if( i + di > nrows ) { di = nrows - i; }

                for( size_t j = 0 ; j < ncols ; j++ )
                {
                    NumericTableFeature &f = (*_ddict)[j];

                    char *ptr = (char *)_arrays[j] + (idx + i) * f.typeSize;

                    for( size_t ii = 0 ; ii < di; ii++ )
                    {
                        lbuf[ii] = blockPtr[ (i + ii) * ncols + j ];
                    }

                    data_feature_utils::getVectorDownCast(f.indexType, data_feature_utils::getInternalNumType<T>())
                    ( di, lbuf, ptr );
                }
            }
        }
        block.setDetails( 0, 0, 0 );
        return services::Status();
    }

    template <typename T>
    services::Status getTFeature( size_t feat_idx, size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T>& block )
    {
        size_t ncols = getNumberOfColumns();
        size_t nobs = getNumberOfRows();
        block.setDetails( feat_idx, idx, rwFlag );

        if (idx >= nobs)
        {
            block.resizeBuffer( 1, 0 );
            return services::Status();
        }

        nrows = ( idx + nrows < nobs ) ? nrows : nobs - idx;

        char *ptr = (char *)_arrays[feat_idx];

        NumericTableFeature &f = (*_ddict)[feat_idx];

        if( data_feature_utils::getIndexNumType<T>() == f.indexType )
        {
            block.setPtr( (T *)ptr + idx, 1, nrows );
        }
        else
        {
            char *location = ptr + idx * f.typeSize;

            if( !block.resizeBuffer( 1, nrows ) )
            {
                return services::Status(services::ErrorMemoryAllocationFailed);
            }

            if( !(block.getRWFlag() & (int)readOnly) ) return services::Status();

            data_feature_utils::getVectorUpCast(f.indexType, data_feature_utils::getInternalNumType<T>())
            ( nrows, location, block.getBlockPtr() );
        }
        return services::Status();
    }

    template <typename T>
    services::Status releaseTFeature( size_t feat_idx, size_t idx, size_t nrows, T *buf, ReadWriteMode rwFlag )
    {
        if (rwFlag & (int)writeOnly)
        {
            NumericTableFeature &f = (*_ddict)[feat_idx];

            if( data_feature_utils::getIndexNumType<T>() != f.indexType )
            {
                char *ptr      = (char *)_arrays[feat_idx];
                char *location = ptr + idx * f.typeSize;

                data_feature_utils::getVectorDownCast(f.indexType, data_feature_utils::getInternalNumType<T>())
                ( nrows, buf, location );
            }
        }
        return services::Status();
    }

    template <typename T>
    services::Status releaseTFeature( BlockDescriptor<T>& block )
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            size_t feat_idx = block.getColumnsOffset();

            NumericTableFeature &f = (*_ddict)[feat_idx];

            if( data_feature_utils::getIndexNumType<T>() != f.indexType )
            {
                char *ptr = (char *)_arrays[feat_idx] + block.getRowsOffset() * f.typeSize;

                data_feature_utils::getVectorDownCast(f.indexType, data_feature_utils::getInternalNumType<T>())
                ( block.getNumberOfRows(), block.getBlockPtr(), ptr );
            }
        }
        block.setDetails( 0, 0, 0 );
        return services::Status();
    }
};
/** @} */
} // namespace interface1
using interface1::SOANumericTable;

}
} // namespace daal
#endif
