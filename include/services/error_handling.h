/** file error_handling.h */
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
//  Handling errors in Intel(R) DAAL.
//--
*/

#ifndef __ERROR_HANDLING__
#define __ERROR_HANDLING__

#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
#include <exception>
#endif

#include "daal_string.h"
#include "error_indexes.h"
#include "error_id.h"
#include "services/collection.h"

namespace daal
{
namespace services
{
namespace interface1
{

#if (!defined(DAAL_NOTHROW_EXCEPTIONS))

/**
 * <a name="DAAL-ENUM-SERVICES__EXCEPTION"></a>
 * \brief Class that represents an exception
 */
struct Exception : std::exception
{
public:
    /**
     * Constructs an exception with a description
     * \param[in] description    Description of the exception
     */
    Exception(const char *description) : _description(description) {};

    /**
     * Return description of the exception
     * \return Description of the exception
     */
    virtual const char *what() const throw() { return _description; };

#ifndef cxx11
    /**
     * Destructor of an the exception
     */
    virtual ~Exception() throw() {}
#endif

    /**
     * Returns exception with description
     * \return Exception with description
     */
    static Exception getException(const String &description)
    {
        String d(description);
        return Exception(d.c_str());
    }

    /**
    * Returns exception with description
    * \return Exception with description
    */
    static Exception getException(const char *description)
    {
        return Exception(description);
    }

private:
    const char *_description;
};

#endif
/**
 * <a name="DAAL-ENUM-SERVICES__MESSAGECOLLECTION"></a>
 * \brief Class that represents an Message collection
 * \tparam IDType Type of message in message collection
 */
template<class IDType>
class MessageCollection : public Collection<SharedPtr<Message<IDType> > >
{
public:
    /**
     * Constructs a message collection
     * \param[in] noMessageFound Index retuned when no corresponding element is found in the collection
     */
    MessageCollection(IDType noMessageFound) : _noMessageFound(noMessageFound) {};

    /**
     * Finds message for error by error ID
     * \param[in] id Error identifier
     * \return Pointer to message
     */
    services::SharedPtr<Message<IDType> > find(IDType id)
    {
        bool found = false;
        size_t index = 0;

        if(this->size() == 0) { return find(_noMessageFound); }

        for(size_t i = 0; i < this->size() && found == false; i++)
        {
            if((*this)[i]->id() == id)
            {
                found = true;
                index = i;
            }
        }

        if(found) { return (*this)[index]; }
        else { return find(_noMessageFound); }
    }

    /**
     * Destructor of a message collection
     */
    virtual ~MessageCollection() {};

protected:
    const IDType _noMessageFound;
};

/**
 * <a name="DAAL-ENUM-SERVICES__ERRORMESSAGECOLLECTION"></a>
 * \brief Class that represents an error message collection
 */
class DAAL_EXPORT ErrorMessageCollection : public MessageCollection<ErrorID>
{
public:
    /**
     * Constructs an error message collection
     */
    ErrorMessageCollection() : MessageCollection<ErrorID>(NoErrorMessageFound)
    {
        parseResourceFile();
    };

    /**
     * Destructor of an error message collection
     */
    ~ErrorMessageCollection() {};

protected:
    void parseResourceFile();
};

/**
 * <a name="DAAL-ENUM-SERVICES__ERRORDETAILCOLLECTION"></a>
 * \brief Class that represents an error detail collection
 */
class DAAL_EXPORT ErrorDetailCollection : public MessageCollection<ErrorDetailID>
{
public:
    /**
     * Construct an error detail collection
     */
    ErrorDetailCollection() :
        MessageCollection<ErrorDetailID>(NoErrorMessageDetailFound)
    {
        parseResourceFile();
    };

    /**
     * Destructor of an error detail collection
     */
    ~ErrorDetailCollection() {};

protected:
    void parseResourceFile();
};

} // namespace interface1
#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
using interface1::Exception;
#endif
using interface1::MessageCollection;
using interface1::ErrorMessageCollection;
using interface1::ErrorDetailCollection;

namespace interface1
{

/**
 * <a name="DAAL-ENUM-SERVICES__ERROR"></a>
 * \brief Class that represents an error
 */
class DAAL_EXPORT Error
{
public:
    /**
     * Constructs an error from an identifier
     * \param[in] id    Identifier of the error
     */
    Error(const ErrorID &id = NoErrorMessageFound) : _id(id),
        _intDetails(Collection<SharedPtr<ErrorDetail<int> > >()),
        _doubleDetails(Collection<SharedPtr<ErrorDetail<double> > >()),
        _stringDetails(Collection<SharedPtr<ErrorDetail<String> > >())
    {}

    /**
     * Copy constructor. Constructs an error from a copy of the content of another error
     * \param[in] e    Another error to be used as a source with which to initialize the contents of this error
     */
    Error(Error &e) : _id(e.id()), _intDetails(e.intDetails()), _doubleDetails(e.doubleDetails()), _stringDetails(e.stringDetails()) {}

    /** Destructor */
    ~Error() {}

    /**
     * Returns a description of the error
     * \return Identifier of this error
     */
    ErrorID id() { return _id; }

    /**
     * Sets an identifier of the error
     * \param[in] id    Identifier of the error
     */
    void setId(ErrorID id)
    {
        _id = id;
    }

    /**
     * Returns the word description of the error
     * \return Description of the error
     */
    const char *description();

    /**
     * Returns a collection of integer details associated with this error
     * \return Collection of integer details associated with this error
     */
    Collection<SharedPtr<ErrorDetail<int> > > intDetails() { return _intDetails; }

    /**
     * Returns a collection of floating-point details associated with this error
     * \return Collection of floating-point details associated with this error
     */
    Collection<SharedPtr<ErrorDetail<double> > > doubleDetails() { return _doubleDetails; }

    /**
     * Returns a collection of string details associated with this error
     * \return Collection of string details associated with this error
     */
    Collection<SharedPtr<ErrorDetail<String> > > stringDetails() { return _stringDetails; }

    /**
     * Adds an integer detail into a collection of details associated with this error
     * \param[in] id    Identifier of the detail
     * \param[in] value Value of the detail
     */
    void addIntDetail(const ErrorDetailID &id, const int &value)
    {
        _intDetails.push_back(services::SharedPtr<ErrorDetail<int> >(new ErrorDetail<int>(id, value)));
    }

    /**
     * Adds a floating-point detail into a collection of details associated with this error
     * \param[in] id    Identifier of the detail
     * \param[in] value Value of the detail
     */
    void addDoubleDetail(const ErrorDetailID &id, const double &value)
    {
        _doubleDetails.push_back(services::SharedPtr<ErrorDetail<double> >(new ErrorDetail<double>(id, value)));
    }

    /**
     * Adds a string detail into a collection of details associated with this error
     * \param[in] id    Identifier of the detail
     * \param[in] value Value of the detail
     */
    void addStringDetail(const ErrorDetailID &id, const String &value)
    {
        _stringDetails.push_back(services::SharedPtr<ErrorDetail<String> >(new ErrorDetail<String>(id, value)));
    }

private:
    ErrorID _id;
    Collection<SharedPtr<ErrorDetail<int> > > _intDetails;
    Collection<SharedPtr<ErrorDetail<double> > > _doubleDetails;
    Collection<SharedPtr<ErrorDetail<String> > > _stringDetails;
};


/**
 * <a name="DAAL-ENUM-SERVICES__KERNELERRORCOLLECTION"></a>
 * \brief Class that represents a kernel error collection (collection that cannot throw exceptions)
 */
class DAAL_EXPORT KernelErrorCollection : public Collection<SharedPtr<Error> >
{
public:
    /**
     * Constructs a kernel error collection
     */
    KernelErrorCollection() : _description(0) {};

    /**
     * Copy constructor of a kernel error collection
     * \param[in] other Kernel error collection that will be copied
     */
    KernelErrorCollection(KernelErrorCollection &other) : _description(0) {};

    /**
     * Constructs a kernel error collection from two collections
     * \param[in] first   First part of kernel error collection
     * \param[in] second  Second part of kernel error collection
     */
    KernelErrorCollection(daal::services::SharedPtr<KernelErrorCollection> first,
                          daal::services::SharedPtr<KernelErrorCollection> second) : _description(0)
    {
        resize(first->capacity() + second->capacity());
        for(size_t i = 0; i < first->size(); i++)
        {
            this->push_back((*first)[i]);
        }
        for(size_t i = 0; i < second->size(); i++)
        {
            this->push_back((*second)[i]);
        }
    };

    /**
     * Adds error id to kernel error collection
     * \param[in] id Identifier of error
     */
    void add(const ErrorID &id)
    {
        push_back(services::SharedPtr<Error>(new Error(id)));
    }

    /**
     * Adds error to kernel error collection
     * \param[in] e Error to be added to kernel error collection
     */
    void add(const services::SharedPtr<Error> &e)
    {
        push_back(e);
    }

    /**
     * Adds error kernel collection
     * \param[in] e Error collection that will be added
     */
    void add(const services::SharedPtr<KernelErrorCollection> &e)
    {
        for(size_t i = 0; i < e->size(); i++)
        {
            push_back((*e)[i]);
        }
    }

    /**
     * Returns true if kernel error collection is empty
     * \return True if kernel error collection is empty
     */
    bool isEmpty()
    {
        return (size() == 0);
    }

    /**
     * Destructor of a kernel error collection
     */
    virtual ~KernelErrorCollection() { if(_description) { daal_free(_description); } }

    /**
     * Returns description of all errors from collection
     * \return Description of all errors from collection
     */
    char *getDescription();

private:
    char *_description;
};

/**
 * <a name="DAAL-ENUM-SERVICES__ERRORCOLLECTION"></a>
 * \brief Class that represents an error collection
 */
class DAAL_EXPORT ErrorCollection
{
public:
    /**
     * Constructs an error collection
     */
    ErrorCollection() : _errors(new KernelErrorCollection()) {};

    /**
     * Copy constructor for error collection
     * \param[in] other Error collection that will be copied
     */
    ErrorCollection(const ErrorCollection &other) : _errors(other.getErrors()) {};

    /**
     * Copy constructor for error collection
     * \param[in] other Error collection that will be copied
     */
    ErrorCollection(services::SharedPtr<ErrorCollection> other) : _errors(other->getErrors()) {};

    /**
     * Constructs error collection from two error collections
     * \param[in] first   First part of error collection
     * \param[in] second  Second part of error collection
     */
    ErrorCollection(daal::services::SharedPtr<ErrorCollection> first,
                    daal::services::SharedPtr<KernelErrorCollection> second) : _errors(first->getErrors())
    {
        _errors->add(second);
    };

    /**
     * Constructs error collection from three error collections
     * \param[in] first   First part of error collection
     * \param[in] second  Second part of error collection
     * \param[in] third   Third part of error collection
     */
    ErrorCollection(services::SharedPtr<ErrorCollection> first,
                    services::SharedPtr<KernelErrorCollection> second,
                    services::SharedPtr<KernelErrorCollection> third) : _errors(first->getErrors())
    {
        _errors->add(second);
        _errors->add(third);
    };

    /**
     * Adds error to error collection and throws exception if DAAL_NOTHROW_EXCEPTIONS not defined
     * \param[in] id Error identifier
     */
    void add(const ErrorID &id)
    {
        _errors->add(id);

#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
        throw Exception::getException(getDescription());
#endif
    }


    /**
     * Adds error to error collection and throws exception if DAAL_NOTHROW_EXCEPTIONS not defined
     * \param[in] e Error
     */
    void add(const services::SharedPtr<Error> &e)
    {
        _errors->add(e);
#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
        throw Exception::getException(getDescription());
#endif
    }

    /**
     * Adds error collection to another error collection and throw exception if DAAL_NOTHROW_EXCEPTIONS not defined
     * \param[in] e Error collection
     */
    void add(const services::SharedPtr<ErrorCollection> &e)
    {
        for(size_t i = 0; i < e->size(); i++)
        {
            _errors->add((*(e->getErrors()))[i]);
        }
#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
        throw Exception::getException(getDescription());
#endif
    }

    /**
     * Returns size of an error collection
     * \return Size of an error collection
     */
    size_t size()
    {
        return _errors->size();
    }

    /**
     * Returns true if kernel error collection is empty
     * \return True if kernel error collection is empty
     */
    bool isEmpty()
    {
        return _errors->isEmpty();
    }

    /**
     * Destructor of error collection
     */
    virtual ~ErrorCollection() {}

    /**
     * Returns kernel error collection
     * \return True if kernel error collection is empty
     */
    const services::SharedPtr<KernelErrorCollection> &getErrors() const
    {
        return _errors;
    }

    /**
     * Returns description of all errors from collection
     * \return Description of all errors from collection
     */
    char *getDescription() const { return _errors->getDescription(); }

private:
    services::SharedPtr<KernelErrorCollection> _errors;
};

} // namespace interface1
using interface1::Error;
using interface1::KernelErrorCollection;
using interface1::ErrorCollection;

}
};
#endif
