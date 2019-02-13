/* file: average_pooling3d_layer_backward.cpp */
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
//  Implementation of average_pooling3d calculation algorithm and types methods.
//--
*/

#include "average_pooling3d_layer_backward_types.h"
#include "average_pooling3d_layer_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace average_pooling3d
{
namespace backward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_AVERAGE_POOLING3D_BACKWARD_RESULT_ID);
/**
 * Default constructor
 */
Input::Input() {}
Input::Input(const Input& other) : super(other) {}

/**
 * Returns an input object for backward average 3D pooling layer
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr Input::get(LayerDataId id) const
{
    layers::LayerDataPtr inputData = get(layers::backward::inputFromForward);
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>((*inputData)[id]);
}

/**
 * Sets an input object for the backward average 3D pooling layer
 * \param[in] id  Identifier of the input object
 * \param[in] ptr Pointer to the object
 */
void Input::set(LayerDataId id, const data_management::NumericTablePtr &ptr)
{
    layers::LayerDataPtr inputData = get(layers::backward::inputFromForward);
    (*inputData)[id] = ptr;
}

/**
 * Checks an input object for the backward average 3D pooling layer
 * \param[in] parameter Algorithm parameter
 * \param[in] method    Computation method
 */
services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *param = static_cast<const Parameter *>(parameter);
    if (!param->propagateGradient) { return services::Status(); }

    services::Status s;
    DAAL_CHECK_STATUS(s, pooling3d::backward::Input::check(parameter, method));

    data_management::NumericTablePtr auxInputDimensions = get(average_pooling3d::auxInputDimensions);
    const services::Collection<size_t> &inputGradDims = get(layers::backward::inputGradient)->getDimensions();

    DAAL_CHECK_EX(inputGradDims.size() >= 3, services::ErrorIncorrectNumberOfDimensionsInTensor, services::ParameterName, inputGradientStr());

    return data_management::checkNumericTable(auxInputDimensions.get(), auxInputDimensionsStr(), data_management::packed_mask, 0, inputGradDims.size(), 1);
}

data_management::NumericTablePtr Input::getAuxInputDimensions() const
{
    return get(auxInputDimensions);
}

/**
* Default constructor
*/
Result::Result() {}

/**
 * Checks the result of the backward average 3D pooling layer
 * \param[in] input     %Input object for the layer
 * \param[in] parameter %Parameter of the layer
 * \param[in] method    Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *param = static_cast<const Parameter *>(parameter);
    if (!param->propagateGradient) { return services::Status(); }

    return pooling3d::backward::Result::check(input, parameter, method);
}

}// namespace interface1
}// namespace backward
}// namespace average_pooling3d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
