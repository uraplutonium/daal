/* file: data_collection.h */
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

#ifndef __DATA_COLLECTION_H__
#define __DATA_COLLECTION_H__

#include "services/daal_defines.h"
#include "data_management/data/data_serialize.h"
#include "data_management/data/data_archive.h"
#include "services/daal_shared_ptr.h"
#include "services/collection.h"

namespace daal
{
namespace data_management
{

namespace interface1
{

/**
 *  <a name="DAAL-CLASS-DATACOLLECTION"></a>
 *  \brief Class that provides functionality of Collection container for objects derived from
 *  SerializationIface interface and implements SerializationIface itself
 */
class DataCollection : public SerializationIface, public services::Collection<services::SharedPtr<SerializationIface> >
{
public:

    /** Default constructor */
    DataCollection() : services::Collection<services::SharedPtr<SerializationIface> >() {}

    /**
     *  Constructor with a defined number of elements
     *  \param[in]  n  Number of elements
     */
    DataCollection(size_t n) : services::Collection<services::SharedPtr<SerializationIface> >(n) {}

    virtual ~DataCollection() {};

    virtual int getSerializationTag()
    {
        return SERIALIZATION_DATACOLLECTION_ID;
    }

    void serializeImpl(interface1::InputDataArchive *arch)
    {
        serialImpl<interface1::InputDataArchive, false>( arch );
    }

    void deserializeImpl(interface1::OutputDataArchive *arch)
    {
        serialImpl<interface1::OutputDataArchive, true>( arch );
    }

    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        size_t size = _size;

        arch->set(size);

        if( onDeserialize )
        {
            resize(size);
        }

        _size = size;

        for(size_t i = 0; i < _size; i++)
        {
            arch->setSharedPtrObj(_array[i]);
        }
    }
};

/**
 *  <a name="DAAL-CLASS-KEYVALUEDATACOLLECTION"></a>
 *  \brief Class that provides functionality of a key-value container for objects derived from the
 *  SerializationIface interface with a key of the size_t type
 */
class KeyValueDataCollection : public SerializationIface
{
public:

    /** Default constructor */
    KeyValueDataCollection()
    {
        nullPtr = new services::SharedPtr<SerializationIface>();
    }

    KeyValueDataCollection(services::Collection<size_t> &_keys, DataCollection &_values)
    {
        for(size_t i = 0; i < _keys.size(); i++)
        {
            keys.push_back(_keys[i]);
            values.push_back(_values[i]);
        }
        nullPtr = new services::SharedPtr<SerializationIface>();
    }


    virtual ~KeyValueDataCollection()
    {
        delete nullPtr;
    }

    /**
     *  Returns a reference to SharedPtr for a stored object with a given key if an object with such key was registered
     *  \param[in]  k  Key value
     *  \return Reference to SharedPtr of the SerializationIface type
     */
    const services::SharedPtr<SerializationIface> &operator[] (size_t k) const
    {
        size_t i;
        for( i = 0; i < keys.size(); i++ )
        {
            if( keys[i] == k )
            {
                return values[i];
            }
        }
        return *nullPtr;
    }

    /**
     *  Creates an empty SharedPtr and stores it under a requested key and returns a reference for this value
     *  \param[in]  k  Key value
     *  \return Reference to SharedPtr of the SerializationIface type
     */
    services::SharedPtr<SerializationIface> &operator[] (size_t k)
    {
        size_t i;
        for( i = 0; i < keys.size(); i++ )
        {
            if( keys[i] == k )
            {
                return values[i];
            }
        }
        keys.push_back(k);
        values.push_back( services::SharedPtr<SerializationIface>() );
        return values[i];
    }

    /**
     *  Returns a reference to SharedPtr for a stored key with a given index
     *  \param[in]  idx  Index of the requested key
     *  \return Reference to SharedPtr of the size_t type
     */
    size_t getKeyByIndex(int idx)
    {
        return keys[idx];
    }

    /**
     *  Returns a reference to SharedPtr for a stored object with a given index
     *  \param[in]  idx  Index of the requested object
     *  \return Reference to SharedPtr of the SerializationIface type
     */
    services::SharedPtr<SerializationIface> &getValueByIndex(int idx)
    {
        return values[idx];
    }

    /**
     *  Returns the number of stored objects
     *  \return Number of stored objects
     */
    size_t size()
    {
        return keys.size();
    }

    /**
     *  Removes all elements from a container
     */
    void clear()
    {
        keys.clear();
        values.clear();
    }

    /** \private */
    virtual int getSerializationTag()
    {
        return SERIALIZATION_KEYVALUEDATACOLLECTION_ID;
    }

    /** \private */
    void serializeImpl(interface1::InputDataArchive  *arch)
    {
        serialImpl<interface1::InputDataArchive, false>( arch );
    }

    /** \private */
    void deserializeImpl(interface1::OutputDataArchive *arch)
    {
        serialImpl<interface1::OutputDataArchive, true>( arch );
    }

    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl( Archive *arch )
    {
        arch->setObj(&values);

        size_t size = values.size();

        if( onDeserialize )
        {
            keys.resize(size);
        }

        for(size_t i = 0; i < size; i++)
        {
            if( onDeserialize )
            {
                keys.push_back(0);
            }
            arch->set(keys[i]);
        }
    }

protected:
    services::Collection<size_t> keys;
    DataCollection values;
    services::SharedPtr<SerializationIface> *nullPtr;
};
} // namespace interface1
using interface1::DataCollection;
using interface1::KeyValueDataCollection;

}
}

#endif
