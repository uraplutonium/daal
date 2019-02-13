/** file daal_string.cpp */
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

#include <cstring>

#include "services/daal_string.h"
#include "service_defines.h"
#include "service_service.h"

namespace daal
{
namespace services
{

DAAL_EXPORT const int String::__DAAL_STR_MAX_SIZE = DAAL_MAX_STRING_SIZE;

void String::initialize(const char *str, const size_t length)
{
    if(length)
    {
        _c_str = (char *)daal::services::daal_malloc(sizeof(char) * (length + 1));
        if (!_c_str) { return; }

        daal::internal::Service<>::serv_strncpy_s(_c_str, length + 1, str, length + 1);
    }
}

void String::reset()
{
    if (_c_str) { daal_free(_c_str); }
}

String::String() : _c_str(0) { }

String::String(size_t length, char placeholder) : _c_str(0)
{
    if (length)
    {
        _c_str = (char *)daal::services::daal_malloc(sizeof(char) * (length + 1));
        if (!_c_str) { return; }

        for (size_t i = 0; i < length; i++)
        {
            _c_str[i] = placeholder;
        }
        _c_str[length] = '\0';
    }
}

String::String(const char *begin, const char *end) : _c_str(0)
{
    initialize(begin, end - begin);
}

String::String(const char *str, size_t capacity) : _c_str(0)
{
    size_t strLength = 0;
    if(str)
    {
        strLength = strnlen(str, String::__DAAL_STR_MAX_SIZE);
    }
    initialize(str, strLength);
};

String::String(const String &str) : _c_str(0)
{
    initialize(str.c_str(), str.length());
};

String::~String()
{
    reset();
}

String &String::operator = (const String &other)
{
    if (this != &other)
    {
        reset();
        initialize(other.c_str(), other.length());
    }
    return *this;
}

size_t String::length() const
{
    if(_c_str)
    {
        return strnlen(_c_str, String::__DAAL_STR_MAX_SIZE);
    }
    return 0;
}

void String::add(const String &str)
{
    size_t prevLength = length();
    char *prevStr = (char *)daal::services::daal_malloc(sizeof(char) * (prevLength + 1));
    daal::internal::Service<>::serv_strncpy_s(prevStr, prevLength + 1, _c_str, prevLength + 1);

    size_t newLength = prevLength + str.length() + 1;
    if(_c_str) { daal_free(_c_str); }
    _c_str = (char *)daal::services::daal_malloc(sizeof(char) * (newLength + 1));

    daal::internal::Service<>::serv_strncpy_s(_c_str, prevLength + 1, prevStr, prevLength + 1);
    daal::internal::Service<>::serv_strncat_s(_c_str, newLength, str.c_str(), newLength - prevLength);

    if(prevStr) { daal_free(prevStr); }
}

String &String::operator+ (const String &str)
{
    add(str);
    return *this;
}

char String::operator[] (size_t index) const
{
    return _c_str[index];
}

char String::get(size_t index) const
{
    return _c_str[index];

}

const char *String::c_str() const
{
    return _c_str;
}

} // namespace services
} // namespace daal
