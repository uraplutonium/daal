/* file: service_service.cpp */
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
//  Implementation of service functions
//--
*/

#include "service_service.h"

float daal::services::daal_string_to_float(const char * nptr, char ** endptr)
{
    return daal::internal::Service<>::serv_string_to_float(nptr, endptr);
}

double daal::services::daal_string_to_double(const char * nptr, char ** endptr)
{
    return daal::internal::Service<>::serv_string_to_double(nptr, endptr);
}

int daal::services::daal_int_to_string(char * buffer, size_t n, int value)
{
    return daal::internal::Service<>::serv_int_to_string(buffer, n, value);
}

int daal::services::daal_double_to_string(char * buffer, size_t n, double value)
{
    return daal::internal::Service<>::serv_double_to_string(buffer, n, value);
}
