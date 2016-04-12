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


#ifndef __TENSOR_H__
#define __TENSOR_H__

#include "services/error_handling.h"
#include "services/daal_memory.h"
#include "data_management/data/numeric_types.h"

namespace daal
{
namespace data_management
{

namespace interface1
{

/**
 *  <a name="DAAL-CLASS-SUBTENSORDESCRIPTOR"></a>
 *  \brief %Base class that manages buffer memory for read/write operations required by tensors.
 */
template<typename DataType> class SubtensorDescriptor;

/**
 *  <a name="DAAL-CLASS-TENSORIFACE"></a>
 *  \brief Abstract interface class for a data management component responsible for representation of data in the numeric format.
 *  This class declares the most general methods for data access.
 */
class TensorIface
{
public:
    /**
     * <a name="DAAL-ENUM-TENSORIFACE_MEMORYSTATUS"></a>
     * \brief Enumeration to specify the status of memory related to the Numeric Table
     */
    enum MemoryStatus
    {
        notAllocated        = 0, /*!< No memory allocated */
        userAllocated       = 1, /*!< Memory allocated on user side */
        internallyAllocated = 2  /*!< Memory allocated and managed by Tensor */
    };

    /**
     * <a name="DAAL-ENUM-TENSORIFACE_ALLOCATIONFLAG"></a>
     * \brief Enumeration to specify whether the Numeric Table must allocate memory
     */
    enum AllocationFlag
    {
        notAllocate = 0, /*!< Memory will not be allocated by Tensor */
        doAllocate  = 1  /*!< Memory will be allocated by Tensor when needed */
    };

    /**
     * <a name="DAAL-ENUM-TENSORIFACE_DATALAYOUT"></a>
     * \brief Enumeration to specify layout of Tensor data
     */
    enum DataLayout
    {
        defaultLayout = 0 /*!< Default memory layout for the Tensor */
    };

    virtual ~TensorIface()
    {}
    /**
     *  Sets the number of dimensions in the Tensor
     *
     *  \param[in] ndim     Number of dimensions
     *  \param[in] dimSizes Array with sizes for each dimension
     */
    virtual void setDimensions(size_t ndim, const size_t* dimSizes) = 0;

    /**
     *  Sets the number and size of dimensions in the Tensor
     *
     *  \param[in] dimensions Collection with sizes for each dimension
     */
    virtual void setDimensions(const services::Collection<size_t>& dimensions) = 0;

    /**
     *  Allocates memory for a data set
     */
    virtual void allocateDataMemory() = 0;

    /**
     *  Deallocates the memory allocated for a data set
     */
    virtual void freeDataMemory() = 0;
};

/**
 *  <a name="DAAL-CLASS-DENSETENSORIFACE"></a>
 *  \brief Abstract interface class for a data management component responsible for accessing data in the numeric format.
 *  This class declares specific methods to access Tensor data in a dense homogeneous form.
 */
class DenseTensorIface
{
public:
    virtual ~DenseTensorIface()
    {}
    /**
     *  Gets subtensor from the tensor
     *
     *  \param[in]  fixedDims    The number of first dimension with fixed values
     *  \param[in]  fixedDimNums Values at which dimensions are fixed
     *  \param[in]  rangeDimIdx  Values for the next dimension after fixed to get data from
     *  \param[in]  rangeDimNum  Range for dimension values to get data from
     *  \param[in]  rwflag       Flag specifying read/write access to the subtensor
     *  \param[out] subtensor    The subtensor descriptor.
     *  \param[in]  layout       Layout of the requested subtensor
     */
    virtual void getSubtensor(size_t fixedDims, size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        ReadWriteMode rwflag, SubtensorDescriptor<double>& subtensor, TensorIface::DataLayout layout = TensorIface::defaultLayout ) = 0;
    /**
     *  Gets subtensor from the tensor
     *
     *  \param[in]  fixedDims    The number of first dimension with fixed values
     *  \param[in]  fixedDimNums Values at which dimensions are fixed
     *  \param[in]  rangeDimIdx  Values for the next dimension after fixed to get data from
     *  \param[in]  rangeDimNum  Range for dimension values to get data from
     *  \param[in]  rwflag       Flag specifying read/write access to the subtensor
     *  \param[out] subtensor    The subtensor descriptor.
     *  \param[in]  layout       Layout of the requested subtensor
     */
    virtual void getSubtensor(size_t fixedDims, size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        ReadWriteMode rwflag, SubtensorDescriptor<float>& subtensor, TensorIface::DataLayout layout = TensorIface::defaultLayout ) = 0;
    /**
     *  Gets subtensor from the tensor
     *
     *  \param[in]  fixedDims    The number of first dimension with fixed values
     *  \param[in]  fixedDimNums Values at which dimensions are fixed
     *  \param[in]  rangeDimIdx  Values for the next dimension after fixed to get data from
     *  \param[in]  rangeDimNum  Range for dimension values to get data from
     *  \param[in]  rwflag       Flag specifying read/write access to the subtensor
     *  \param[out] subtensor    The subtensor descriptor.
     *  \param[in]  layout       Layout of the requested subtensor
     */
    virtual void getSubtensor(size_t fixedDims, size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum,
        ReadWriteMode rwflag, SubtensorDescriptor<int>& subtensor, TensorIface::DataLayout layout = TensorIface::defaultLayout ) = 0;

    /**
     *  Releases subtensor
     *
     *  \param[in] subtensor    The subtensor descriptor.
     */
    virtual void releaseSubtensor(SubtensorDescriptor<double>& subtensor) = 0;
    /**
     *  Releases subtensor
     *
     *  \param[in] subtensor    The subtensor descriptor.
     */
    virtual void releaseSubtensor(SubtensorDescriptor<float>& subtensor) = 0;
    /**
     *  Releases subtensor
     *
     *  \param[in] subtensor    The subtensor descriptor.
     */
    virtual void releaseSubtensor(SubtensorDescriptor<int>& subtensor) = 0;
};

/**
 *  <a name="DAAL-CLASS-TENSOR"></a>
 *  \brief Class for a data management component responsible for representation of data in the n-dimensions numeric format.
 *  This class implements the most general methods for data access.
 */
class Tensor : public SerializationIface, public TensorIface, public DenseTensorIface
{
public:
    /** \private */
    Tensor(size_t nDim, const size_t* dimSizes) : _errors(new services::KernelErrorCollection())
    {
        setDimensions(nDim, dimSizes);
        _memStatus    = notAllocated;
    }

    /** \private */
    Tensor(const services::Collection<size_t>& dims) : _errors(new services::KernelErrorCollection())
    {
        setDimensions(dims);
        _memStatus    = notAllocated;
    }

    /** \private */
    Tensor() : _errors(new services::KernelErrorCollection())
    {
        _memStatus    = notAllocated;
    }

    /** \private */
    virtual ~Tensor() {}

    /**
     *  Gets the status of the memory used by a data set connected with a Tensor
     */
    MemoryStatus getDataMemoryStatus() const { return _memStatus; }

    /**
     *  Gets the number of dimensions in the Tensor
     *
     *  \return Number of dimensions
     */
    size_t getNumberOfDimensions() const
    {
        return _dims.size();
    }

    /**
     *  Gets the size of the dimension in the Tensor
     *
     *  \param[in] dimIdx Index of dimension
     *
     *  \return Dimension size
     */
    size_t getDimensionSize(size_t dimIdx) const
    {
        if(getNumberOfDimensions() > dimIdx) return _dims[dimIdx];
        return 0;
    }

    /**
     *  Gets the size of dimensions in the Tensor
     *
     *  \return Collection with sizes for each dimension
     */
    const services::Collection<size_t>& getDimensions() const
    {
        return _dims;
    }

    virtual void setDimensions(const services::Collection<size_t>& dims) DAAL_C11_OVERRIDE
    {
        _dims.clear();
        _dims.resize(dims.size());

        for(size_t i = 0; i < dims.size(); i++)
        {
            _dims.push_back(dims[i]);
        }
    }

    virtual void setDimensions(size_t nDim, const size_t* dimSizes) DAAL_C11_OVERRIDE
    {
        _dims.clear();
        _dims.resize(nDim);

        if(!dimSizes)
        {
            this->_errors->add(services::ErrorNullParameterNotSupported);
            return;
        }

        for(size_t i=0; i<nDim; i++)
        {
            _dims.push_back(dimSizes[i]);
        }
    }

    /**
     *  Returns errors during the computation
     *  \return Errors during the computation
     */
    services::SharedPtr<services::KernelErrorCollection> getErrors()
    {
        return _errors;
    }

    size_t getSize()
    {
        size_t nDim = getNumberOfDimensions();
        if( nDim==0 ) return 0;

        size_t size = 1;

        for(size_t i=0; i<nDim; i++)
        {
            size *= _dims[i];
        }

        return size;
    }

    /**
     * Checks the correctness of dimension sizes of this tensor
     * \param[in] dims Collection with required tensor dimension sizes
     * \return Error if the dimension sizes of this tensor are not equal to the required dimension sizes.
     *         Empty error otherwise.
     */
    services::SharedPtr<services::Error> check(const services::Collection<size_t> *dims = NULL) const
    {
        using namespace daal::services;

        SharedPtr<Error> error;
        if (dims)
        {
            /* Here if collection of the required dimension sizes is provided */
            if (getNumberOfDimensions() != dims->size())
            { error = SharedPtr<Error>(new Error(ErrorIncorrectNumberOfDimensionsInTensor)); return error; }

            for (size_t d = 0; d < dims->size(); d++)
            {
                if (getDimensionSize(d) != (*dims)[d])
                {
                    error = SharedPtr<Error>(new Error(ErrorIncorrectSizeOfDimensionInTensor));
                    error->addIntDetail(Dimension, (int)d);
                    return error;
                }
            }
        }
        else
        {
            /* Here if collection of the required dimension sizes is not provided */
            /* Check that the tensor is not empty */
            size_t nDims = getNumberOfDimensions();
            if (nDims == 0)
            { error = SharedPtr<Error>(new Error(ErrorIncorrectNumberOfDimensionsInTensor)); return error; }

            for (size_t d = 0; d < nDims; d++)
            {
                if (getDimensionSize(d) == 0)
                {
                    error = SharedPtr<Error>(new Error(ErrorIncorrectSizeOfDimensionInTensor));
                    error->addIntDetail(Dimension, (int)d);
                    return error;
                }
            }
        }

        return error;
    }

    /**
     *  Returns new tensor with first dimension limited to one point
     *  \param[in] firstDimIndex Index of the point in the first dimention
     */
    virtual services::SharedPtr<Tensor> getSampleTensor(size_t firstDimIndex) = 0;

protected:
    MemoryStatus  _memStatus;

    services::Collection<size_t> _dims;

    services::SharedPtr<services::KernelErrorCollection> _errors;
};

/**
 *  <a name="DAAL-CLASS-SUBTENSORDESCRIPTOR"></a>
 *  \brief Class with descriptor of the subtensor retrieved from Tensor getSubTensor function
 */
template<typename DataType>
class SubtensorDescriptor
{
public:
    /** \private */
    SubtensorDescriptor() :
        _ptr(0), _buffer(0), _capacity(0),
        _tensorNDims(0), _nFixedDims(0), _rangeDimIdx(0), _dimNums(0),
        _subtensorSize(0), _rwFlag(0) {}

    /** \private */
    ~SubtensorDescriptor()
    {
        freeBuffer();
        if( _dimNums )
        {
            daal::services::daal_free( _dimNums );
        }
    }

    /**
     *   Gets a pointer to the buffer for the subtensor
     *  \return Pointer to the subtensor
     */
    inline DataType* getPtr() const { return _ptr; }

    /**
     *  Returns the number of dimensions of the subtensor
     *  \return Number of columns
     */
    inline size_t getNumberOfDims() const { return _tensorNDims-_nFixedDims; }

    /**
     *  Returns the array with sizes of dimensions of the subtensor
     *  \return Number of rows
     */
    inline size_t* getSubtensorDimSizes() const { return _dimNums+_nFixedDims; }

public:
    inline void setPtr( DataType* ptr )
    {
        _ptr   = ptr;
    }

    inline bool resizeBuffer()
    {
        if ( _subtensorSize > _capacity )
        {
            freeBuffer();

            _buffer = (DataType*)daal::services::daal_malloc( _subtensorSize*sizeof(DataType) );

            if ( _buffer != 0 )
            {
                _capacity = _subtensorSize;
            }
            else
            {
                return false;
            }

        }

        _ptr = _buffer;

        return true;
    }

    inline size_t setDetails( size_t tensorNDims, size_t *tensorDimNums,
        size_t nFixedDims, size_t *fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, int rwFlag )
    {
        _rwFlag = rwFlag;

        if( _tensorNDims != tensorNDims )
        {
            if( _tensorNDims != 0 )
            {
                daal::services::daal_free( _dimNums );
            }

            _dimNums = (size_t*)daal::services::daal_malloc( tensorNDims * sizeof(size_t) );
            if( !_dimNums )
            {
                _tensorNDims = 0;
                return 0;
            }

            _tensorNDims = tensorNDims;
        }

        _nFixedDims = nFixedDims;
        for( size_t i = 0; i < _nFixedDims; i++ )
        {
            _dimNums[i] = fixedDimNums[i];
        }

        _subtensorSize = 1;

        if( _nFixedDims != _tensorNDims )
        {
            _rangeDimIdx = rangeDimIdx;
            _dimNums[_nFixedDims] = rangeDimNum;
            _subtensorSize *= rangeDimNum;
        }

        for( size_t i = _nFixedDims+1; i < _tensorNDims; i++ )
        {
            _dimNums[i] = tensorDimNums[i];
            _subtensorSize *= tensorDimNums[i];
        }

        return _subtensorSize;
    }

    inline size_t  getSize()         const { return _subtensorSize; }
    inline size_t  getFixedDims()    const { return _nFixedDims; }
    inline size_t* getFixedDimNums() const { return _dimNums; }
    inline size_t  getRangeDimIdx()  const { return _rangeDimIdx; }
    inline size_t  getRangeDimNum()  const
    {
        if( _nFixedDims != _tensorNDims )
        {
            return _dimNums[_nFixedDims];
        }
        return 1;
    }

    inline size_t  getRWFlag() const { return _rwFlag; }

protected:
    /**
     *  Frees the buffer
     */
    void freeBuffer()
    {
        if ( _capacity )
        {
            daal::services::daal_free( _buffer );
        }
        _buffer = 0;
        _capacity = 0;
    }

private:
    DataType *_ptr;      /*<! Pointer to the buffer */
    DataType *_buffer;   /*<! Pointer to the buffer */
    size_t    _capacity; /*<! Buffer size in bytes */

    size_t _tensorNDims;
    size_t _nFixedDims;
    size_t _rangeDimIdx;
    size_t *_dimNums;

    size_t _subtensorSize;

    int    _rwFlag;        /*<! Buffer size in bytes */
};

}

using interface1::Tensor;
using interface1::SubtensorDescriptor;

}
} // namespace daal

#endif
