/* file: data_serialize.h */
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
//  Declaration and implementation of the serialization class
//--
*/

#ifndef __DAAL_SERIALIZE_H__
#define __DAAL_SERIALIZE_H__

#include "services/base.h"
#include "services/daal_memory.h"

namespace daal
{
namespace data_management
{

namespace interface1
{

class InputDataArchive;
class OutputDataArchive;

/**
 *  <a name="DAAL-CLASS-SERIALIZATIONIFACE"></a>
 *  \brief Abstract interface class that defines the interface for serialization and deserialization.
 */

class DAAL_EXPORT SerializationIface : public Base
{
public:
    virtual ~SerializationIface() {}

    /**
     *  Performs serialization
     *  \param[in]  archive  Storage for a serialized object or data structure
     */
    void serialize(interface1::InputDataArchive &archive);

    /**
     *  Performs deserialization
     *  \param[in]  archive  Storage for a deserialized object or data structure
     */
    void deserialize(interface1::OutputDataArchive &archive);

    virtual int getSerializationTag() = 0;

    /**
     *  Interfaces for the implementation of serialization
     *  \param[in]  archive  Storage for a serialized object or data structure
     */
    virtual void serializeImpl(interface1::InputDataArchive *archive) = 0;

    /**
     *  Interfaces for the implementation of deserialization
     *  \param[in]  archive  Storage for a deserialized object or data structure
     */
    virtual void deserializeImpl(interface1::OutputDataArchive *archive) = 0;

    /**
     *  Allocates memory of the given size for serializable object
     *  \param[in] sz number of bytes to be allocated
     *  \return pointer to the allocated memory
     */
    static void* operator new(std::size_t sz)
    {
        return daal::services::daal_malloc(sz);
    }

    /**
     *  Allocates memory of the given size for array of serializable objects
     *  \param[in] sz number of bytes to be allocated
     *  \return pointer to the allocated memory
     */
    static void* operator new[](std::size_t sz)
    {
        return daal::services::daal_malloc(sz);
    }

    /**
    *  Placement new for the serializable object
    *  \param[in] sz     number of bytes to be allocated
    *  \param[in] where  pointer to the memory
    *  \return pointer to the allocated memory
    */
    static void *operator new(std::size_t sz, void *where)
    {
        return where;
    }

    /**
     *  Placement new for the serializable object
     *  \param[in] sz     number of bytes to be allocated
     *  \param[in] where  pointer to the memory
     *  \return pointer to the allocated memory
     */
    static void *operator new[](std::size_t sz, void *where)
    {
        return where;
    }

    /**
     *  Frees memory allocated for serializable object
     *  \param[in] ptr pointer to the allocated memory
     *  \param[in] sz  number of bytes to be freed
     */
    static void operator delete(void* ptr, std::size_t sz)
    {
        daal::services::daal_free(ptr);
    }

    /**
     *  Frees memory allocated for array of serializable objects
     *  \param[in] ptr pointer to the allocated memory
     *  \param[in] sz  number of bytes to be freed
     */
    static void operator delete[](void* ptr, std::size_t sz)
    {
        daal::services::daal_free(ptr);
    }
};
} // namespace interface1
using interface1::SerializationIface;

}
}

#endif
