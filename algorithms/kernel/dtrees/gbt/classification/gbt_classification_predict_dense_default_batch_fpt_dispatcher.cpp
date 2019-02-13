/* file: gbt_classification_predict_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of gradient boosted trees algorithm container -- a class
//  that contains fast gradient boosted trees prediction kernels
//  for supported architectures.
//--
*/

#include "gbt_classification_predict_container.h"

namespace daal
{
namespace algorithms
{
namespace interface1
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(gbt::classification::prediction::BatchContainer, batch,\
    DAAL_FPTYPE, gbt::classification::prediction::defaultDense)
}

namespace gbt
{
namespace classification
{
namespace prediction
{
namespace interface1
{
template <>
Batch<DAAL_FPTYPE, gbt::classification::prediction::defaultDense>::Batch(size_t nClasses)
{
    _par = new ParameterType(nClasses);
    initialize();
};

using BatchType = Batch<DAAL_FPTYPE, gbt::classification::prediction::defaultDense>;
template <>
Batch<DAAL_FPTYPE, gbt::classification::prediction::defaultDense>::Batch(const BatchType &other) : classifier::prediction::Batch(other), input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}
}
}
}
}

}
} // namespace daal
