/* file: spatial_maximum_pooling2d_layer_forward_fpt.cpp */
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
//  Implementation of spatial pooling2d calculation algorithm and types methods.
//--
*/

#include "spatial_maximum_pooling2d_layer_types.h"
#include "spatial_maximum_pooling2d_layer_forward_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace spatial_maximum_pooling2d
{
namespace forward
{
namespace interface1
{
/**
 * Allocates memory to store the result of the forward spatial pyramid maximum 2D pooling layer
 * \param[in] input Pointer to an object containing the input data
 * \param[in] parameter %Parameter of the forward spatial pyramid maximum 2D pooling layer
 * \param[in] method Computation method for the layer
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status s = spatial_pooling2d::forward::Result::allocate<algorithmFPType>(input, parameter, method);
    DAAL_CHECK_STATUS_VAR(s);
    const Input *in = static_cast<const Input *>(input);
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    if(!algParameter->predictionStage)
    {
        const services::Collection<size_t> &inputDims = in->get(layers::forward::data)->getDimensions();
        services::Collection<size_t> valueDims = computeValueDimensions(inputDims, algParameter);

        set(auxSelectedIndices, data_management::HomogenTensor<int>::create(valueDims, data_management::Tensor::doAllocate, &s));
        set(auxInputDimensions, createAuxInputDimensions(inputDims));
    }
    return s;
}
template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace forward
}// namespace spatial_maximum_pooling2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
