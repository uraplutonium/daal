/* file: abs_layer_backward_fpt.cpp */
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
//  Implementation of abs calculation algorithm and types methods.
//--
*/

#include "abs_layer_backward_types.h"
#include "abs_layer_types.h"

#include "service_mkl_tensor.h"
#include "tensor.h"
#include "service_tensor.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace abs
{
namespace backward
{
namespace interface1
{
/**
* Allocates memory to store the result of the backward abs layer
* \param[in] input     Pointer to an object containing the input data
* \param[in] parameter %Parameter of the layer
* \param[in] method    Computation method
*/
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const layers::Parameter *param = static_cast<const layers::Parameter * >(parameter);
    if (!param->propagateGradient) { return services::Status(); }
    const Input *in = static_cast<const Input *>(input);

    services::Status s;
    data_management::TensorPtr valueTable = in->get(auxData);
    if (!valueTable) return services::Status(services::ErrorNullInputNumericTable);

    if (!get(layers::backward::gradient))
    {
        auto inTensor = in->get(layers::backward::inputGradient);
        data_management::HomogenTensor<algorithmFPType> *inHomo = dynamic_cast<data_management::HomogenTensor<algorithmFPType>*>( inTensor.get() );
        internal       ::MklTensor    <algorithmFPType> *inMkl  = dynamic_cast<internal       ::MklTensor    <algorithmFPType>*>( inTensor.get() );

        if(inHomo || inMkl)
        {
            set(layers::backward::gradient, inTensor);
        }
        else
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::backward::gradient, valueTable->getDimensions());
        }
    }
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace backward
}// namespace abs
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
