/* file: tensor.h */
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
//  Declaration and implementation of the base class for numeric n-cubes.
//--
*/


#ifndef __HOMOGEN_TENSOR_H__
#define __HOMOGEN_TENSOR_H__

#include "services/daal_defines.h"
#include "data_management/data/tensor.h"

namespace daal
{
namespace data_management
{
namespace interface1
{

/**
 *  <a name="DAAL-CLASS-HOMOGENTENSOR"></a>
 *  \brief Class that provides methods to access data stored as a contiguous array
 *  of homogeneous data in rows-major format.
 *  \tparam DataType Defines the underlying data type that describes a Tensor
 */
template<typename DataType = double>
class HomogenTensor : public Tensor
{
public:
    /** \private */
    HomogenTensor(size_t nDim, const size_t *dimSizes, DataType *data) : Tensor(nDim, dimSizes), _dimOffsets(0)
    {
        _ptr = data;
        _allocatedSize = 0;

        if( data )
        {
            _allocatedSize = getSize();
        }

        _dimOffsets = (size_t *)daal::services::daal_malloc(nDim * sizeof(size_t));

        if(!dimSizes)
        {
            this->_errors->add(services::ErrorNullParameterNotSupported);
            return;
        }

        for(size_t i = 0; i < nDim; i++)
        {
            _dimOffsets[i] = 1;
        }

        for(size_t i = 0; i < nDim; i++)
        {
            for(size_t j = 0; j < i; j++)
            {
                _dimOffsets[j] *= dimSizes[i];
            }
        }
    }

    /** \private */
    HomogenTensor(const services::Collection<size_t> &dims, DataType *data) : Tensor(dims), _dimOffsets(0)
    {
        _ptr = data;
        _allocatedSize = 0;

        if( data )
        {
            _allocatedSize = getSize();
        }

        size_t nDim = dims.size();
        _dimOffsets = (size_t *)daal::services::daal_malloc(nDim * sizeof(size_t));

        if(nDim == 0)
        {
            this->_errors->add(services::ErrorNullParameterNotSupported);
            return;
        }

        for(size_t i = 0; i < nDim; i++)
        {
            _dimOffsets[i] = 1;
        }

        for(size_t i = 0; i < nDim; i++)
        {
            for(size_t j = 0; j < i; j++)
            {
                _dimOffsets[j] *= dims[i];
            }
        }
    }

    /** \private */
    HomogenTensor(const services::Collection<size_t> &dims, AllocationFlag memoryAllocationFlag) : Tensor(dims), _dimOffsets(0),
        _allocatedSize(0), _ptr(0)
    {
        _dimOffsets = (size_t *)daal::services::daal_malloc(dims.size() * sizeof(size_t));

        for(size_t i = 0; i < dims.size(); i++)
        {
            _dimOffsets[i] = 1;
        }

        for(size_t i = 0; i < dims.size(); i++)
        {
            for(size_t j = 0; j < i; j++)
            {
                _dimOffsets[j] *= dims[i];
            }
        }

        if( memoryAllocationFlag == doAllocate )
        {
            allocateDataMemory();
        }
    }

    /** \private */
    HomogenTensor(const services::Collection<size_t> &dims, AllocationFlag memoryAllocationFlag, const DataType initValue):
        Tensor(dims), _dimOffsets(0), _allocatedSize(0), _ptr(0)
    {
        _dimOffsets = (size_t *)daal::services::daal_malloc(dims.size() * sizeof(size_t));

        for(size_t i = 0; i < dims.size(); i++)
        {
            _dimOffsets[i] = 1;
        }

        for(size_t i = 0; i < dims.size(); i++)
        {
            for(size_t j = 0; j < i; j++)
            {
                _dimOffsets[j] *= dims[i];
            }
        }

        if( memoryAllocationFlag == doAllocate )
        {
            allocateDataMemory();
            assign(initValue);
        }
    }

    /** \private */
    virtual ~HomogenTensor()
    {
        if( _dimOffsets ) { daal::services::daal_free(_dimOffsets); }
        freeDataMemory();
    }

public:
    DataType *getArray() const
    {
        return _ptr;
    }

    virtual void setDimensions(size_t nDim, const size_t *dimSizes) DAAL_C11_OVERRIDE
    {
        if( getNumberOfDimensions() != nDim)
        {
            if( _dimOffsets ) { daal::services::daal_free( _dimOffsets ); }
            _dimOffsets = (size_t *)daal::services::daal_malloc(nDim * sizeof(size_t));
            if(!_dimOffsets)
            {
                this->_errors->add(services::ErrorMemoryAllocationFailed);
                return;
            }
        }

        Tensor::setDimensions(nDim, dimSizes);

        if(!dimSizes)
        {
            this->_errors->add(services::ErrorNullParameterNotSupported);
            return;
        }

        for(size_t i = 0; i < nDim; i++)
        {
            _dimOffsets[i] = 1;
        }

        for(size_t i = 0; i < nDim; i++)
        {
            for(size_t j = 0; j < i; j++)
            {
                _dimOffsets[j] *= dimSizes[i];
            }
        }
    }

    virtual void allocateDataMemory() DAAL_C11_OVERRIDE
    {
        freeDataMemory();

        if( _memStatus != notAllocated )
        {
            /* Error is already reported by freeDataMemory() */
            return;
        }

        size_t size = getSize();

        _ptr = (DataType *)daal::services::daal_malloc( size * sizeof(DataType) );

        if( _ptr == 0 )
        {
            this->_errors->add(services::ErrorMemoryAllocationFailed);
            return;
        }

        _allocatedSize = getSize();
        _memStatus = internallyAllocated;
    }

    void assign(const DataType initValue)
    {
        size_t size = getSize();

        for(size_t i = 0; i < size; i++)
        {
            _ptr[i] = initValue;
        }
    }

    virtual void freeDataMemory() DAAL_C11_OVERRIDE
    {
        if( getDataMemoryStatus() == internallyAllocated && _allocatedSize > 0 )
        {
            daal::services::daal_free(_ptr);
        }

        _ptr = 0;
        _allocatedSize = 0;
        _memStatus = notAllocated;
    }

    void getSubtensor(size_t fixedDims, size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
                      ReadWriteMode rwflag, SubtensorDescriptor<double> &block,
                      TensorIface::DataLayout layout = TensorIface::defaultLayout) DAAL_C11_OVERRIDE
    {
        return getTSubtensor<double>(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, block);
    }
    void getSubtensor(size_t fixedDims, size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
                      ReadWriteMode rwflag, SubtensorDescriptor<float> &block,
                      TensorIface::DataLayout layout = TensorIface::defaultLayout) DAAL_C11_OVERRIDE
    {
        return getTSubtensor<float>(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, block);
    }
    void getSubtensor(size_t fixedDims, size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
                      ReadWriteMode rwflag, SubtensorDescriptor<int> &block,
                      TensorIface::DataLayout layout = TensorIface::defaultLayout) DAAL_C11_OVERRIDE
    {
        return getTSubtensor<int>(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, block);
    }

    void releaseSubtensor(SubtensorDescriptor<double> &block) DAAL_C11_OVERRIDE
    {
        releaseTSubtensor<double>(block);
    }
    void releaseSubtensor(SubtensorDescriptor<float> &block) DAAL_C11_OVERRIDE
    {
        releaseTSubtensor<float>(block);
    }
    void releaseSubtensor(SubtensorDescriptor<int> &block) DAAL_C11_OVERRIDE
    {
        releaseTSubtensor<int>(block);
    }

    virtual services::SharedPtr<Tensor> getSampleTensor(size_t firstDimIndex) DAAL_C11_OVERRIDE
    {
        services::Collection<size_t> newDims = getDimensions();
        if(!_ptr || newDims.size() == 0 || newDims[0] <= firstDimIndex) { return services::SharedPtr<Tensor>(); }
        newDims[0] = 1;
        return services::SharedPtr<Tensor>(new HomogenTensor<DataType>(newDims, _ptr + _dimOffsets[0]*firstDimIndex));
    }

    virtual int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return data_feature_utils::getIndexNumType<DataType>() + SERIALIZATION_HOMOGEN_TENSOR_ID;
    }

    void serializeImpl  (InputDataArchive  *archive) DAAL_C11_OVERRIDE
    {serialImpl<InputDataArchive, false>( archive );}

    void deserializeImpl(OutputDataArchive *archive) DAAL_C11_OVERRIDE
    {serialImpl<OutputDataArchive, true>( archive );}

protected:
    template<typename Archive, bool onDeserialize>
    void serialImpl( Archive *archive ) {}

private:

    template <typename T>
    void getTSubtensor( size_t fixedDims, size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, int rwFlag,
                        SubtensorDescriptor<T> &block )
    {
        size_t  nDim     = _dims.size();
        size_t *dimSizes = &(_dims[0]);
        size_t blockSize = block.setDetails( nDim, dimSizes, fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwFlag );

        size_t shift = 0;
        for( size_t i = 0; i < fixedDims; i++ )
        {
            shift += fixedDimNums[i] * _dimOffsets[i];
        }
        if( fixedDims != nDim )
        {
            shift += rangeDimIdx * _dimOffsets[fixedDims];
        }

        if( IsSameType<T, DataType>::value )
        {
            block.setPtr( (T *)(_ptr + shift) );
        }
        else
        {
            if( !block.resizeBuffer() )
            {
                this->_errors->add(services::ErrorMemoryAllocationFailed);
                return;
            }

            if( rwFlag & (int)readOnly )
            {
                data_feature_utils::vectorUpCast[data_feature_utils::getIndexNumType<DataType>()][data_feature_utils::getInternalNumType<T>()]
                ( blockSize, _ptr + shift, block.getPtr() );
            }
        }
    }

    template <typename T>
    void releaseTSubtensor( SubtensorDescriptor<T> &block )
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            if( !IsSameType<T, DataType>::value )
            {
                size_t nDim = getNumberOfDimensions();

                size_t blockSize = block.getSize();

                size_t fixedDims     = block.getFixedDims();
                size_t *fixedDimNums = block.getFixedDimNums();
                size_t rangeDimIdx   = block.getRangeDimIdx();

                size_t shift = 0;
                for( size_t i = 0; i < fixedDims; i++ )
                {
                    shift += fixedDimNums[i] * _dimOffsets[i];
                }
                if( fixedDims != nDim )
                {
                    shift += rangeDimIdx * _dimOffsets[fixedDims];
                }

                data_feature_utils::vectorDownCast[data_feature_utils::getIndexNumType<DataType>()][data_feature_utils::getInternalNumType<T>()]
                ( blockSize, block.getPtr(), _ptr + shift );
            }
        }
    }

private:
    DataType *_ptr;
    size_t   *_dimOffsets;
    size_t    _allocatedSize;

};

}
using interface1::HomogenTensor;

}
} // namespace daal

#endif
