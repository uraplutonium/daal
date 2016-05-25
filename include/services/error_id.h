/** file error_id.h */
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
//  Data types for error handling in Intel(R) DAAL.
//--
*/

#ifndef __ERROR_ID__
#define __ERROR_ID__

#include "error_indexes.h"
#include "services/daal_defines.h"
#include "daal_memory.h"

namespace daal
{
namespace services
{

namespace interface1
{

/**
* <a name="DAAL-ENUM-SERVICES__ERRORDETAILBASE"></a>
* \brief Base for error detail classes
*/
class DAAL_EXPORT ErrorDetailBase
{
public:
    DAAL_NEW_DELETE();

    /**
    * Constructs error detail from error identifier
    * \param[in] id    Error identifier
    */
    ErrorDetailBase(const ErrorDetailID &id) : _id(id), _next(NULL){}

    /**
    * Destructor
    */
    virtual ~ErrorDetailBase(){}

    /**
    * Returns identifier of an error detail
    * \return identifier of an error detail
    */
    ErrorDetailID id() const { return _id; }

    /**
    * Returns copy of this object
    * \return copy of this object
    */
    virtual ErrorDetailBase* clone() const = 0;

    /**
    * Adds description of the error detail to the given string
    * \param[in] str String to add descrition to
    */
    virtual void describe(char* str) const = 0;

    /**
    * Access to the pointer of the next detail
    * \return pointer to the next detail
    */
    const ErrorDetailBase* next() const { return _next; }

protected:
    /**
    * Access to the pointer of the next detail
    * \return pointer to the next detail
    */
    ErrorDetailBase* next() { return _next; }

    /**
    * Set pointer of the next detail
    * \param[in] ptr Pointer of the next detail
    */
    void addNext(ErrorDetailBase* ptr) { _next = ptr; }

private:
    const ErrorDetailID _id;
    ErrorDetailBase* _next;
    friend class Error;
};

/**
 * <a name="DAAL-ENUM-SERVICES__ERRORDETAIL"></a>
 * \brief Class that represents error detail
 * \tparam Type of value in an error detail
 */
template<typename T>
class ErrorDetail : public ErrorDetailBase
{
public:
    DAAL_NEW_DELETE();

    /**
     * Constructs error detail from error identifier and value
     * \param[in] id    Error identifier
     * \param[in] value Value of error detail
     */
    ErrorDetail(const ErrorDetailID &id, const T &value) : ErrorDetailBase(id), _value(value) {}

    /**
    * Destructor
    */
    ~ErrorDetail(){}

    /**
     * Returns value of an error detail
     * \return value of an error detail
     */
    T value() const { return _value; }

    /**
    * Returns copy of this object
    * \return copy of this object
    */
    virtual ErrorDetailBase* clone() const { return new ErrorDetail<T>(id(), value());  }

    /**
    * Adds description of the error detail to the given string
    * \param[in] str String to add descrition to
    */
    virtual void describe(char* str) const;

private:
    const T _value;
};

/**
 * <a name="DAAL-ENUM-SERVICES__SINGLEERROR"></a>
 * \brief Class that represents single error
 */
class SingleError
{
public:
    /**
     * Constructs a single error
     */
    SingleError() : _isInitialized(false), _errorID(UnknownError) {};

    /**
     * Constructs a single error from error identifier
     * \param[in] errorID Error identifier
     */
    SingleError(const ErrorID &errorID) : _isInitialized(true), _errorID(errorID) {};

    /**
     * Returns true if error was initialized
     * \return true if error was initialized
     */
    bool isInitialized() const { return _isInitialized; }

    /**
     * Returns ErrorID of an single error
     * \return ErrorID of an single error
     */
    ErrorID getErrorID() const { return _errorID; }

    /**
     * Sets ErrorID for an single error
     * \param[in] errorID Error identifier to be set
     */
    void setErrorID(const ErrorID &errorID)
    {
        _errorID = errorID;
        _isInitialized = true;
    }

private:
    bool _isInitialized;
    ErrorID _errorID;
};
} // namespace interface1
using interface1::ErrorDetailBase;
using interface1::ErrorDetail;
using interface1::SingleError;

}
}
#endif
