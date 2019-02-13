/* file: daal_factory_impl.cpp */
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
//  Implementation of dictionary utils.
//--
*/

#include "data_archive.h"
#include "homogen_numeric_table.h"
#include "aos_numeric_table.h"
#include "soa_numeric_table.h"
#include "csr_numeric_table.h"
#include "merged_numeric_table.h"
#include "row_merged_numeric_table.h"
#include "symmetric_matrix.h"
#include "matrix.h"
#include "data_collection.h"
#include "homogen_tensor.h"
#include "service_mkl_tensor.h"
#include "serialization_utils.h"

#include "memory_block.h"

namespace daal
{
namespace data_management
{

namespace interface1
{
template class BlockDescriptor<int>;
template class BlockDescriptor<float>;
template class BlockDescriptor<double>;
}

#undef __DAAL_CREATOR_ARGUMENTS
#define __DAAL_CREATOR_ARGUMENTS(...) __VA_ARGS__

#undef __DAAL_ADD_TYPE
#define __DAAL_ADD_TYPE(leftPart, rightPart)                    \
    {                                                           \
        registerObject(new leftPart float          rightPart);  \
        registerObject(new leftPart double         rightPart);  \
        registerObject(new leftPart int            rightPart);  \
        registerObject(new leftPart unsigned int   rightPart);  \
        registerObject(new leftPart DAAL_INT64     rightPart);  \
        registerObject(new leftPart DAAL_UINT64    rightPart);  \
        registerObject(new leftPart char           rightPart);  \
        registerObject(new leftPart unsigned char  rightPart);  \
        registerObject(new leftPart short          rightPart);  \
        registerObject(new leftPart unsigned short rightPart);  \
    }

#undef __DAAL_REGISTER_TEMPLATED_OBJECT
#define __DAAL_REGISTER_TEMPLATED_OBJECT(CreatorName, ObjectName, ...)                                                  \
    {                                                                                                                   \
        __DAAL_ADD_TYPE(__DAAL_CREATOR_ARGUMENTS(CreatorName<ObjectName<__VA_ARGS__), __DAAL_CREATOR_ARGUMENTS(> >()))  \
    }

namespace interface1
{

struct FactoryEntry
{
    typedef services::SharedPtr<const AbstractCreator> ValueType;
    int key;
    ValueType value;

    FactoryEntry(int _key = -1) : key(_key){}
    FactoryEntry(int _key, const ValueType& _value) : key(_key), value(_value){}
    FactoryEntry(const FactoryEntry& o) : key(o.key), value(o.value){}
    FactoryEntry &operator=(const FactoryEntry &o)
    {
        key = o.key;
        value = o.value;
        return *this;
    }
};

class FactoryImpl
{
public:
    FactoryImpl(){}
    int find(int id) const
    {
        for(size_t i = 0; i < _map.size(); i++)
        {
            if(_map[i].key == id)
                return (int)i;
        }
        return -1;
    }
    void add(const AbstractCreator* creator, bool bOwned)
    {
        if(bOwned)
            _map.push_back(FactoryEntry(creator->getTag(), FactoryEntry::ValueType(creator)));
        else
            _map.push_back(FactoryEntry(creator->getTag(), FactoryEntry::ValueType(creator, daal::services::EmptyDeleter())));
    }
    const AbstractCreator* at(size_t index) const { return _map[index].value.get();  }

protected:
    services::Collection<FactoryEntry> _map;
};

}
using interface1::FactoryImpl;

class DefaultCreator : public AbstractCreator
{
public:
    DefaultCreator(const SerializationDesc* desc) : _desc(desc){}

    virtual SerializationIface *create() const { return (*_desc->creator())(); }
    virtual int getTag() const { return _desc->tag(); }

private:
    const SerializationDesc* _desc;
};

Factory::Factory() : _impl(nullptr)
{
    _impl = new FactoryImpl();
    for(auto ptr = SerializationDesc::first(); ptr; ptr = ptr->next())
        _impl->add(new DefaultCreator(ptr), true);

    __DAAL_REGISTER_TEMPLATED_OBJECT(Creator, HomogenNumericTable, );
    __DAAL_REGISTER_TEMPLATED_OBJECT(Creator, Matrix, );
    __DAAL_REGISTER_TEMPLATED_OBJECT(Creator, HomogenTensor, );

    __DAAL_REGISTER_TEMPLATED_OBJECT(Creator, PackedSymmetricMatrix,  NumericTableIface::upperPackedSymmetricMatrix, );
    __DAAL_REGISTER_TEMPLATED_OBJECT(Creator, PackedSymmetricMatrix,  NumericTableIface::lowerPackedSymmetricMatrix, );
    __DAAL_REGISTER_TEMPLATED_OBJECT(Creator, PackedTriangularMatrix, NumericTableIface::upperPackedTriangularMatrix, );
    __DAAL_REGISTER_TEMPLATED_OBJECT(Creator, PackedTriangularMatrix, NumericTableIface::lowerPackedTriangularMatrix, );

    registerObject(new Creator<CSRNumericTable>());
    registerObject(new Creator<AOSNumericTable>());
    registerObject(new Creator<SOANumericTable>());
    registerObject(new Creator<MergedNumericTable>());
    registerObject(new Creator<RowMergedNumericTable>());
    registerObject(new Creator<NumericTableDictionary>());
    registerObject(new Creator<data_management::DataCollection >());
    registerObject(new Creator<data_management::KeyValueDataCollection >());

    registerObject(new Creator<daal::internal::MklTensor<float> >());
    registerObject(new Creator<daal::internal::MklTensor<double> >());

    registerObject(new Creator<algorithms::OptionalArgument >());
    registerObject(new Creator<data_management::MemoryBlock >());
}

Factory::~Factory()
{
    delete _impl;
}

void Factory::registerObject(AbstractCreator *creator)
{
    _impl->add(creator, true);
}

Factory &Factory::instance()
{
    static Factory obj;
    return obj;
}

SerializationIface *Factory::createObject(int objectId)
{
    int pos = _impl->find(objectId);
    if(pos == -1)
        return NULL;
    return _impl->at(pos)->create();
}

Factory::Factory(const Factory &) {}
Factory &Factory::operator = (const Factory &factory) { return (*this); }

void SerializationIface::serialize(InputDataArchive &archive)
{
    archive.segmentHeader( getSerializationTag() );
    serializeImpl( &archive );
    archive.segmentFooter();
}

void SerializationIface::deserialize(OutputDataArchive &archive)
{
    int tag = archive.segmentHeader();
    deserializeImpl( &archive );
    archive.segmentFooter();
}

class PtrHolder
{
public:
    PtrHolder() : _first(nullptr){}
    void setFirst(SerializationDesc* p) { _first = p; }
    const SerializationDesc* first() const { return _first; }

private:
    SerializationDesc* _first = nullptr;
};

static PtrHolder& getPtrHolder()
{
    static PtrHolder holder[1];
    return holder[0];
}

SerializationDesc::SerializationDesc(SerializationDesc::creatorFunc func, int tag) : _f(func), _tag(tag), _next(nullptr)
{
    _next = getPtrHolder().first();
    getPtrHolder().setFirst(this);
}

const SerializationDesc* SerializationDesc::first()
{
    return getPtrHolder().first();
}

} // namespace data_management
} // namespace daal
