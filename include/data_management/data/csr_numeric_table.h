/* file: csr_numeric_table.h */
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
//  Implementation of a compressed sparse row (CSR) numeric table.
//--
*/

#ifndef __CSR_NUMERIC_TABLE_H__
#define __CSR_NUMERIC_TABLE_H__

#include "services/base.h"
#include "data_management/data/numeric_table.h"

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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__CSRBLOCKDESCRIPTOR"></a>
 *  \brief %Base class that manages buffer memory for read/write operations required by CSR numeric tables.
 */
template<typename DataType = double>
class CSRBlockDescriptor
{
public:
    /** \private */
    CSRBlockDescriptor() : _values_ptr(0), _rows_ptr(0), _cols_ptr(0),
        _rows_buffer(0), _rows_capacity(0), _values_buffer(0), _values_capacity(0),
        _ncols(0), _nrows(0), _rowsOffset(0), _rwFlag(0) {}

    /** \private */
    ~CSRBlockDescriptor() { freeValuesBuffer(); freeRowsBuffer(); }

    /**
     *  Gets a pointer to the buffer
     *  \return Pointer to the block
     */
    inline DataType *getBlockValuesPtr() const { return _values_ptr; }
    inline size_t *getBlockColumnIndicesPtr() const { return _cols_ptr; }
    inline size_t *getBlockRowIndicesPtr() const { return _rows_ptr; }

    /**
     *  Returns the number of columns in the block
     *  \return Number of columns
     */
    inline size_t getNumberOfColumns() const { return _ncols; }

    /**
     *  Returns the number of rows in the block
     *  \return Number of rows
     */
    inline size_t getNumberOfRows() const { return _nrows; }

    /**
     *  Returns number of elements in values array.
     *  \return Number of elements in values array.
     */
    inline size_t getDataSize() const
    {
        return ((_nrows > 0) ? _rows_ptr[_nrows] - _rows_ptr[0] : 0);
    }
public:
    inline void setValuesPtr( DataType *ptr, size_t nValues )
    {
        _values_ptr = ptr;
        _nvalues    = nValues;
    }

    inline void setColumnIndicesPtr( size_t *ptr, size_t nValues )
    {
        _cols_ptr   = ptr;
        _nvalues    = nValues;
    }

    /**
     *  \param[in] ptr      Pointer to the buffer
     *  \param[in] nRows    Number of rows
     */
    inline void setRowIndicesPtr( size_t *ptr, size_t nRows )
    {
        _rows_ptr   = ptr;
        _nrows = nRows;
    }

    /**
     *  \param[in] nValues  Number of values
     */
    inline bool resizeValuesBuffer( size_t nValues )
    {
        size_t newSize = nValues * sizeof(DataType);
        if ( newSize > _values_capacity )
        {
            freeValuesBuffer();
            _values_buffer = (DataType *)daal::services::daal_malloc(newSize);
            if ( _values_buffer != 0 )
            {
                _values_capacity = newSize;
            }
            else
            {
                return false;
            }

        }

        _values_ptr = _values_buffer;

        return true;
    }

    /**
     *  \param[in] nRows    Number of rows
     */
    inline bool resizeRowsBuffer( size_t nRows )
    {
        _nrows = nRows;
        size_t newSize = (nRows + 1) * sizeof(size_t);
        if ( newSize > _rows_capacity )
        {
            freeRowsBuffer();
            _rows_buffer = (size_t *)daal::services::daal_malloc(newSize);
            if ( _rows_buffer != 0 )
            {
                _rows_capacity = newSize;
            }
            else
            {
                return false;
            }

        }

        _rows_ptr = _rows_buffer;

        return true;
    }

    inline void setDetails( size_t nColumns, size_t rowIdx, int rwFlag )
    {
        _ncols      = nColumns;
        _rowsOffset = rowIdx;
        _rwFlag     = rwFlag;
    }

    inline size_t getRowsOffset() const { return _rowsOffset; }
    inline size_t getRWFlag() const { return _rwFlag; }

protected:
    /**
     *  Frees the values buffer
     */
    void freeValuesBuffer()
    {
        if ( _values_capacity )
        {
            daal::services::daal_free( _values_buffer );
        }
        _values_buffer = 0;
        _values_capacity = 0;
    }

    /**
     *  Frees the rows buffer
     */
    void freeRowsBuffer()
    {
        if ( _rows_capacity )
        {
            daal::services::daal_free( _rows_buffer );
        }
        _rows_buffer = 0;
        _rows_capacity = 0;
    }

private:
    DataType *_values_ptr; /*<! Pointer to the buffer */
    size_t *_cols_ptr;   /*<! Pointer to the buffer */
    size_t *_rows_ptr;   /*<! Pointer to the buffer */
    size_t    _nrows;      /*<! Buffer size in bytes */
    size_t    _ncols;      /*<! Buffer size in bytes */
    size_t    _nvalues;      /*<! Buffer size in bytes */

    size_t _rowsOffset;    /*<! Buffer size in bytes */
    int    _rwFlag;        /*<! Buffer size in bytes */

    DataType *_values_buffer;   /*<! Pointer to the buffer */
    size_t    _values_capacity; /*<! Buffer size in bytes */

    size_t   *_rows_buffer;   /*<! Pointer to the buffer */
    size_t    _rows_capacity; /*<! Buffer size in bytes */
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__CSRNUMERICTABLEIFACE"></a>
 *  \brief Abstract class that defines the interface of CSR numeric tables
 */
class CSRNumericTableIface
{
public:
    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__CSRINDEXING"></a>
     * \brief Enumeration to specify the indexing scheme for access to data in the CSR layout
     */
    enum CSRIndexing
    {
        zeroBased = 0, /*!< 0-based indexing */
        oneBased  = 1  /*!< 1-based indexing */
    };

public:

    virtual ~CSRNumericTableIface() {}

    /**
     *  Returns number of elements in values array.
     *
     *  \return Number of elements in values array.
     */
    virtual size_t getDataSize() = 0;
    /**
     *  Gets a block of feature vectors in the CSR layout.
     *
     *  \param[in] vector_idx       Index of the first row to include into the block.
     *  \param[in] vector_num       Number of rows in the block.
     *  \param[in] rwflag           Flag specifying read/write access to the block of feature vectors.
     *  \param[out] block           The block of feature values.
     *
     *  \return Actual number of feature vectors returned by the method.
     */
    virtual services::Status getSparseBlock(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, CSRBlockDescriptor<double> &block) = 0;

    /**
     *  Gets a block of feature vectors in the CSR layout.
     *
     *  \param[in] vector_idx       Index of the first row to include into the block.
     *  \param[in] vector_num       Number of rows in the block.
     *  \param[in] rwflag           Flag specifying read/write access to the block of feature vectors.
     *  \param[out] block           The block of feature values.
     *
     *  \return Actual number of feature vectors returned by the method.
     */
    virtual services::Status getSparseBlock(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, CSRBlockDescriptor<float> &block) = 0;

    /**
     *  Gets a block of feature vectors in the CSR layout.
     *
     *  \param[in] vector_idx       Index of the first row to include into the block.
     *  \param[in] vector_num       Number of rows in the block.
     *  \param[in] rwflag           Flag specifying read/write access to the block of feature vectors.
     *  \param[out] block           The block of feature values.
     *
     *  \return Actual number of feature vectors returned by the method.
     */
    virtual services::Status getSparseBlock(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, CSRBlockDescriptor<int> &block) = 0;

    /**
     *  Releases a block of feature vectors in the CSR layout.
     *  \param[in] block           The block of feature values.
     */
    virtual services::Status releaseSparseBlock(CSRBlockDescriptor<double> &block) = 0;

    /**
     *  Releases a block of feature vectors in the CSR layout.
     *  \param[in] block           The block of feature values.
     */
    virtual services::Status releaseSparseBlock(CSRBlockDescriptor<float> &block) = 0;

    /**
     *  Releases a block of feature vectors in the CSR layout.
     *  \param[in] block           The block of feature values.
     */
    virtual services::Status releaseSparseBlock(CSRBlockDescriptor<int> &block) = 0;
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__CSRNUMERICTABLE"></a>
 *  \brief Class that provides methods to access data stored in the CSR layout.
 */
class CSRNumericTable : public NumericTable, public CSRNumericTableIface
{
public:
    DAAL_CAST_OPERATOR(CSRNumericTable)
    /**
     *  Constructor for an empty CSR Numeric Table
     */
    CSRNumericTable(): NumericTable(0, 0, DictionaryIface::equal), _ptr(0), _indexing(oneBased)
    {
        _layout = csrArray;
        this->_status |= setArrays<double>( 0, 0, 0 ); //data type doesn't matter
    }

    /**
     *  Constructor for a Numeric Table with user-allocated memory
     *  \tparam   DataType        Type of values in the Numeric Table
     *  \param[in]    ptr         Array of values in the CSR layout. Let ptr_size denote the size of an array ptr
     *  \param[in]    colIndices  Array of column indices in the CSR layout. Values of indices are determined by the index base
     *  \param[in]    rowOffsets  Array of row indices in the CSR layout. Size of the array is nrow+1. The first element is 0/1
     *                            in zero-/one-based indexing. The last element is ptr_size+0/1 in zero-/one-based indexing
     *  \param[in]    nColumns    Number of columns in the corresponding dense table
     *  \param[in]    nRows       Number of rows in the corresponding dense table
     *  \param[in]    indexing    Indexing scheme used to access data in the CSR layout
     *  Note: Present version of Intel(R) Data Analytics Acceleration Library supports 1-based indexing only
     */
    template<typename DataType>
    CSRNumericTable( DataType *const ptr, size_t *colIndices = 0, size_t *rowOffsets = 0,
                     size_t nColumns = 0, size_t nRows = 0, CSRIndexing indexing = oneBased ):
        NumericTable(nColumns, nRows, DictionaryIface::equal), _ptr(0), _indexing(indexing)
    {
        _layout = csrArray;
        this->_status |= setArrays<DataType>(ptr, colIndices, rowOffsets);

        _defaultFeature.setType<DataType>();
        this->_status |= _ddict->setAllFeatures( _defaultFeature );
    }

    virtual ~CSRNumericTable()
    {
        freeDataMemoryImpl();
    }

    virtual int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return SERIALIZATION_CSR_NT_ID;
    }

    /**
     *  Returns  pointers to a data set stored in the CSR layout
     *  \param[out]    ptr         Array of values in the CSR layout
     *  \param[out]    colIndices  Array of column indices in the CSR layout
     *  \param[out]    rowOffsets  Array of row indices in the CSR layout
     */
    template<typename DataType>
    services::Status getArrays(DataType **ptr, size_t **colIndices, size_t **rowOffsets) const
    {
        if(ptr) { *ptr = (DataType*)_ptr; }
        if (colIndices) { *colIndices = _colIndices; }
        if (rowOffsets) { *rowOffsets = _rowOffsets; }
        return services::Status();
    }

    /**
     *  Sets a pointer to a CSR data set
     *  \param[in]    ptr         Array of values in the CSR layout
     *  \param[in]    colIndices  Array of column indices in the CSR layout
     *  \param[in]    rowOffsets  Array of row indices in the CSR layout
     *  \param[in]    indexing    The indexing scheme for access to data in the CSR layout
     */
    template<typename DataType>
    services::Status setArrays(DataType *const ptr, size_t *colIndices, size_t *rowOffsets, CSRIndexing indexing = oneBased)
    {
        freeDataMemoryImpl();

        //if( ptr == 0 || colIndices == 0 || rowOffsets == 0 ) return services::Status(services::ErrorEmptyCSRNumericTable);

        _ptr = ptr;
        _colIndices = colIndices;
        _rowOffsets = rowOffsets;
        _indexing = indexing;

        if( ptr != 0 && colIndices != 0 && rowOffsets != 0 ) { _memStatus  = userAllocated; }
        return services::Status();
    }

    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    services::Status releaseBlockOfRows(BlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
         return releaseTBlock<double>(block);
    }
    services::Status releaseBlockOfRows(BlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
         return releaseTBlock<float>(block);
    }
    services::Status releaseBlockOfRows(BlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
         return releaseTBlock<int>(block);
    }

    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                ReadWriteMode rwflag, BlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
        return getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                ReadWriteMode rwflag, BlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
        return getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num,
                                ReadWriteMode rwflag, BlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
        return getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    services::Status releaseBlockOfColumnValues(BlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
         return releaseTFeature<double>(block);
    }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
         return releaseTFeature<float>(block);
    }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
         return releaseTFeature<int>(block);
    }


    services::Status getSparseBlock(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, CSRBlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
        return getSparseTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getSparseBlock(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, CSRBlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
        return getSparseTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getSparseBlock(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, CSRBlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
        return getSparseTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    services::Status releaseSparseBlock(CSRBlockDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
        return releaseSparseTBlock<double>(block);
    }
    services::Status releaseSparseBlock(CSRBlockDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
        return releaseSparseTBlock<float>(block);
    }
    services::Status releaseSparseBlock(CSRBlockDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
        return releaseSparseTBlock<int>(block);
    }

    /**
     *  Allocates memory for a data set
     *  \param[in]    dataSize     Number of non-zero values
     *  \param[in]    type         Memory type
     */
    services::Status allocateDataMemory(size_t dataSize, daal::MemType type = daal::dram)
    {
        freeDataMemoryImpl();

        size_t nrow = getNumberOfRows();

        if( nrow == 0 )
            return services::Status(services::ErrorIncorrectNumberOfObservations);

        NumericTableFeature &f = (*_ddict)[0];

        _ptr        =           daal::services::daal_malloc( dataSize   * f.typeSize     );
        _colIndices = (size_t *)daal::services::daal_malloc( dataSize   * sizeof(size_t) );
        _rowOffsets = (size_t *)daal::services::daal_malloc( (nrow + 1) * sizeof(size_t) );

        _memStatus = internallyAllocated;

        if( _ptr == 0 || _colIndices == 0 || _rowOffsets == 0 )
        {
            freeDataMemoryImpl();
            return services::Status(services::ErrorMemoryAllocationFailed);
        }

        _rowOffsets[0] = ((_indexing == oneBased) ? 1 : 0);
        return services::Status();
    }

    /**
     * Returns the indexing scheme for access to data in the CSR layout
     * \return  CSR layout indexing
     */
    CSRIndexing getCSRIndexing() const
    {
        return _indexing;
    }

    /**
     * \copydoc NumericTableIface::check
     */
    virtual services::Status check(const char *description, bool checkDataAllocation = true) const DAAL_C11_OVERRIDE
    {
        services::Status s;
        DAAL_CHECK_STATUS(s, data_management::NumericTable::check(description, checkDataAllocation));

        if( _indexing != oneBased )
        {
            return services::Status(services::Error::create(services::ErrorUnsupportedCSRIndexing, services::ArgumentName, description));
        }

        return services::Status();
    }

    /** \private */
    void serializeImpl  (InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<InputDataArchive, false>( arch );}

    /** \private */
    void deserializeImpl(OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<OutputDataArchive, true>( arch );}

protected:
    NumericTableFeature _defaultFeature;
    CSRIndexing _indexing;

    void   *_ptr;
    size_t *_colIndices;
    size_t *_rowOffsets;

    services::Status allocateDataMemoryImpl(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE
    {
        return services::Status(services::ErrorMethodNotSupported);
    }

    void freeDataMemoryImpl() DAAL_C11_OVERRIDE
    {
        if( getDataMemoryStatus() == internallyAllocated )
        {
            if( _ptr        != 0 ) { daal::services::daal_free(_ptr       ); }
            if( _colIndices != 0 ) { daal::services::daal_free(_colIndices); }
            if( _rowOffsets != 0 ) { daal::services::daal_free(_rowOffsets); }
        }

        _ptr = 0;
        _colIndices = 0;
        _rowOffsets = 0;

        _memStatus  = notAllocated;
    }


    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl( Archive *arch )
    {
        NumericTable::serialImpl<Archive, onDeserialize>( arch );

        size_t dataSize = 0;
        if( !onDeserialize )
        {
            dataSize = getDataSize();
        }
        arch->set( dataSize );

        if( onDeserialize )
        {
            allocateDataMemory( dataSize );
        }

        size_t nfeat = getNumberOfColumns();
        size_t nobs  = getNumberOfRows();

        if( nfeat > 0 )
        {
            NumericTableFeature &f = (*_ddict)[0];

            arch->set( (char *)_ptr, dataSize * f.typeSize );
            arch->set( _colIndices, dataSize );
            arch->set( _rowOffsets, nobs + 1   );
        }
    }


public:
    size_t getDataSize() DAAL_C11_OVERRIDE
    {
        size_t nobs  = getNumberOfRows();
        if( nobs > 0)
        {
            return _rowOffsets[nobs] - _rowOffsets[0];
        }
        else
        {
            return 0;
        }
    }

protected:

    template <typename T>
    services::Status getTBlock(size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> &block)
    {
        size_t ncols = getNumberOfColumns();
        size_t nobs  = getNumberOfRows();
        block.setDetails( 0, idx, rwFlag );

        if (idx >= nobs)
        {
            block.resizeBuffer( ncols, 0 );
            return services::Status();
        }

        NumericTableFeature &f = (*_ddict)[0];

        T *buffer;
        T *castingBuffer;
        char *location = (char *)_ptr + (_rowOffsets[idx] - 1) * f.typeSize;

        if( data_feature_utils::getIndexNumType<T>() == f.indexType )
        {
            castingBuffer = (T *)location;

            if( !block.resizeBuffer( ncols, nrows ) )
                return services::Status(services::ErrorMemoryAllocationFailed);
            buffer = block.getBlockPtr();
        }
        else
        {
            size_t sparseBlockSize = _rowOffsets[idx + nrows] - _rowOffsets[idx];

            if( !block.resizeBuffer( ncols, nrows, sparseBlockSize * sizeof(T) ) )
                return services::Status(services::ErrorMemoryAllocationFailed);
            buffer = block.getBlockPtr();

            castingBuffer = (T *)block.getAdditionalBufferPtr();

            data_feature_utils::getVectorUpCast(f.indexType,data_feature_utils::getInternalNumType<T>())
            ( sparseBlockSize, location, castingBuffer );
        }

        T *bufRowCursor       = castingBuffer;
        size_t *indicesCursor = _colIndices + _rowOffsets[idx] - 1;

        for( size_t i = 0; i < ncols * nrows; i++ ) { buffer[i] = (T)0; }

        for( size_t i = 0; i < nrows; i++ )
        {
            size_t sparseRowSize = _rowOffsets[idx + i + 1] - _rowOffsets[idx + i];

            for( size_t k = 0; k < sparseRowSize; k++ )
            {
                buffer[i * ncols + indicesCursor[k] - 1] = bufRowCursor[k];
            }

            bufRowCursor  += sparseRowSize;
            indicesCursor += sparseRowSize;
        }
        return services::Status();
    }

    template <typename T>
    services::Status releaseTBlock( BlockDescriptor<T> &block )
    {
        if (!(block.getRWFlag() & (int)writeOnly))
            block.setDetails( 0, 0, 0 );
        return services::Status();
    }

    template <typename T>
    services::Status getTFeature(size_t feat_idx, size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> &block)
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

        if( !block.resizeBuffer( 1, nrows ) )
            return services::Status(services::ErrorMemoryAllocationFailed);

        NumericTableFeature &f = (*_ddict)[0];

        char   *rowCursor     = (char *)_ptr + (_rowOffsets[idx] - 1) * f.typeSize;
        size_t *indicesCursor = _colIndices + (_rowOffsets[idx] - 1);

        T *buffer = block.getBlockPtr();

        for(size_t i = 0; i < nrows; i++)
        {
            buffer[i] = (T)0;

            size_t sparseRowSize = _rowOffsets[idx + i + 1] - _rowOffsets[idx + i];

            for(size_t k = 0; k < sparseRowSize; k++)
            {
                if( indicesCursor[k] - 1 == feat_idx )
                {
                    data_feature_utils::getVectorUpCast(f.indexType, data_feature_utils::getInternalNumType<T>())
                    ( 1, rowCursor + k * f.typeSize, buffer + i );
                }
            }

            rowCursor     += sparseRowSize * f.typeSize;
            indicesCursor += sparseRowSize;
        }
        return services::Status();
    }

    template <typename T>
    services::Status releaseTFeature( BlockDescriptor<T> &block )
    {
        if(block.getRWFlag() & (int)writeOnly)
            return services::Status();
        block.setDetails( 0, 0, 0 );
        return services::Status();
    }

    template <typename T>
    services::Status getSparseTBlock( size_t idx, size_t nrows, int rwFlag, CSRBlockDescriptor<T> &block )
    {
        size_t ncols = getNumberOfColumns();
        size_t nobs  = getNumberOfRows();
        block.setDetails( ncols, idx, rwFlag );

        if (idx >= nobs)
        {
            block.resizeValuesBuffer( 0 );
            return services::Status();
        }

        nrows = ( idx + nrows < nobs ) ? nrows : nobs - idx;

        NumericTableFeature &f = (*_ddict)[0];

        char *location = (char *)_ptr + (_rowOffsets[idx] - 1) * f.typeSize;

        size_t nValues = _rowOffsets[idx + nrows] - _rowOffsets[idx];

        if( data_feature_utils::getIndexNumType<T>() == f.indexType )
        {
            block.setValuesPtr( (T *)location, nValues );
        }
        else
        {
            if( !block.resizeValuesBuffer(nValues) ) { return  services::Status(); }

            data_feature_utils::getVectorUpCast(f.indexType, data_feature_utils::getInternalNumType<T>())
            ( nValues, location, block.getBlockValuesPtr() );
        }

        block.setColumnIndicesPtr( _colIndices + (_rowOffsets[idx] - 1), nValues );

        if( idx == 0 )
        {
            block.setRowIndicesPtr( _rowOffsets, nrows );
        }
        else
        {
            if( !block.resizeRowsBuffer(nrows) ) { return  services::Status(); }

            size_t *row_offsets = block.getBlockRowIndicesPtr();

            for(size_t i = 0; i < nrows + 1; i++)
            {
                row_offsets[i] = _rowOffsets[idx + i] - _rowOffsets[idx] + 1;
            }
        }
        return services::Status();
    }

    template <typename T>
    services::Status releaseSparseTBlock(CSRBlockDescriptor<T> &block)
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            NumericTableFeature &f = (*_ddict)[0];
            if (f.indexType != data_feature_utils::getIndexNumType<T>())
            {
                size_t nrows = block.getNumberOfRows();
                size_t idx   = block.getRowsOffset();
                size_t nValues = _rowOffsets[idx + nrows] - _rowOffsets[idx];

                char *ptr = (char *)block.getBlockValuesPtr();
                char *location = (char *)_ptr + (_rowOffsets[idx] - 1) * f.typeSize;

                data_feature_utils::getVectorDownCast(f.indexType, data_feature_utils::getInternalNumType<T>())
                        (nValues, ptr, location);
            }
        }
        block.setDetails( 0, 0, 0 );
        return services::Status();
    }

    virtual services::Status setNumberOfColumnsImpl(size_t ncol) DAAL_C11_OVERRIDE
    {
        _ddict->setNumberOfFeatures( ncol );
        _ddict->setAllFeatures( _defaultFeature );
        return services::Status();
    }

};
typedef services::SharedPtr<CSRNumericTable> CSRNumericTablePtr;
/** @} */
} // namespace interface1
using interface1::CSRNumericTableIface;
using interface1::CSRBlockDescriptor;
using interface1::CSRNumericTable;
using interface1::CSRNumericTablePtr;

}
} // namespace daal
#endif
