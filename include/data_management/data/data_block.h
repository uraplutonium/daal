/* file: data_block.h */
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
//  Implementation of the DataBlock type
//--
*/

#ifndef __DAAL_DATABLOCK_H__
#define __DAAL_DATABLOCK_H__

#include "services/base.h"

namespace daal
{
namespace data_management
{

namespace interface1
{
/**
 * @ingroup serialization
 * @{
 */
/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATABLOCKIFACE"></a>
 * \brief Abstract interface class for a data management component responsible for a pointer to a byte array and its size.
 * This class declares the most general methods for data access.
 */
class DataBlockIface : public Base
{
public:
    virtual ~DataBlockIface() {}
    /**
    * Returns a pointer to a byte array stored in DataBlock
    * \return Pointer to the byte array stored in DataBlock
    */
    virtual byte *getPtr() = 0;
    /**
     * Returns the size of a byte array stored in DataBlock
     * \return Size of the byte array stored in DataBlock
     */
    virtual size_t getSize() = 0;
    /**
     * Sets a pointer to a byte array
     * \param[in] ptr Pointer to the byte array
     */
    virtual void setPtr(byte *ptr) = 0;
    /**
     * Sets the size of a byte array
     * \param[in] size Size of the byte array
     */
    virtual void setSize(size_t size) = 0;
};
/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATABLOCK"></a>
 * \brief Class that stores a pointer to a byte array and its size. Not responsible for memory management
 */
class DAAL_EXPORT DataBlock : public DataBlockIface
{
public:
    /**
     * Default constructor. Creates an empty DataBlock of zero size with a zero pointer to a byte array
     */
    DataBlock() : _ptr(0), _size(0)
    {}
    /**
     * Constructor. Creates DataBlock with a user-defined byte array
     * \param ptr Pointer to the byte array
     * \param size Size of the byte array
     */
    DataBlock(byte * ptr, size_t size) : _ptr(ptr), _size(size)
    {}
    /**
     * Constructor. Creates an empty DataBlock of a predefined size
     * \param size Size of the byte array
     */
    DataBlock(size_t size) : _ptr(0), _size(size)
    {}
    /**
     * Copy constructor. Copies a pointer and the size stored in another DataBlock
     * \param block Reference to DataBlock
     */
    DataBlock(const DataBlock &block)
    {
       _ptr = block._ptr;
       _size = block._size;
    }

    virtual ~DataBlock() {}

    virtual byte *getPtr() DAAL_C11_OVERRIDE
    {
        return _ptr;
    }

    virtual size_t getSize() DAAL_C11_OVERRIDE
    {
        return _size;
    }

    virtual void setPtr(byte *ptr) DAAL_C11_OVERRIDE
    {
        _ptr = ptr;
    }

    virtual void setSize(size_t size) DAAL_C11_OVERRIDE
    {
        _size = size;
    }

private:
    byte * _ptr;
    size_t _size;
};
/** @} */
} // namespace interface1
using interface1::DataBlock;
using interface1::DataBlockIface;
}
}

#endif
