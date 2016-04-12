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

namespace daal
{
namespace services
{

namespace interface1
{
/**
 * <a name="DAAL-ENUM-SERVICES__MESSAGE"></a>
 * \brief Class that represents Message
 * \tparam IDType Type of message
 */
template<class IDType>
class Message
{
public:
    /**
     * Constructs Message from identifier and description
     * \param[in] id Error identifier
     * \param[in] description Description for message
     */
    Message(const IDType &id, const char *description) : _id(id), _description(description) {};

    /**
     * Destructor of Message class
     */
    virtual ~Message() {};

    /**
     * Returns identifier of a message
     * \return identifier of a message
     */
    const IDType id() { return _id; }

    /**
     * Returns description of a message
     * \return description of a message
     */
    const char *description() { return _description; }

private:
    IDType _id;
    const char *_description;
};

/**
 * <a name="DAAL-ENUM-SERVICES__ERRORDETAIL"></a>
 * \brief Class that represents error detail
 * \tparam Type of value in an error detail
 */
template<typename T>
class ErrorDetail
{
public:
    /**
     * Constructs error detail from error identifier and value
     * \param[in] id    Error identifier
     * \param[in] value Value of error detail
     */
    ErrorDetail(const ErrorDetailID &id, const T &value) : _id(id), _value(value) {};

    /**
     * Returns identifier of an error detail
     * \return identifier of an error detail
     */
    ErrorDetailID id() const { return _id; }

    /**
     * Returns value of an error detail
     * \return value of an error detail
     */
    T value() const { return _value; }

private:
    const ErrorDetailID _id;
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
    bool isInitialized() { return _isInitialized; }

    /**
     * Returns ErrorID of an single error
     * \return ErrorID of an single error
     */
    ErrorID getErrorID() { return _errorID; }

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
using interface1::Message;
using interface1::ErrorDetail;
using interface1::SingleError;

}
}
#endif
