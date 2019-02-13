/* file: minmax_parameter.cpp */
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
//  Implementation of minmax algorithm and types methods.
//--
*/

#include "minmax_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace minmax
{
namespace interface1
{

/** Constructs min-max normalization parameters */
DAAL_EXPORT ParameterBase::ParameterBase(double lowerBound, double upperBound,
    const SharedPtr<low_order_moments::BatchImpl> &moments) :
    lowerBound(lowerBound), upperBound(upperBound), moments(moments) { }

/**
 * Check the correctness of the %ParameterBase object
 */
DAAL_EXPORT Status ParameterBase::check() const
{
    DAAL_CHECK(moments, ErrorNullParameterNotSupported);
    DAAL_CHECK(lowerBound < upperBound, ErrorLowerBoundGreaterThanOrEqualToUpperBound);
    return Status();
}

}// namespace interface1
}// namespace minmax
}// namespace normalization
}// namespace algorithms
}// namespace daal
