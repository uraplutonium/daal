/* file: concat_layer_backward.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of concat calculation algorithm and types methods.
//--
*/

#include "concat_layer_backward_types.h"
#include "concat_layer_types.h"
#include "serialization_utils.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace concat
{
namespace backward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_CONCAT_BACKWARD_RESULT_ID);
/** \brief Default constructor */
Input::Input() {};
Input::Input(const Input& other) : super(other) {}

/**
* Returns input object of the backward concat layer
* \param[in] id    Identifier of the input object
* \return          %Input LayerData that corresponds to the given identifier
*/
data_management::NumericTablePtr Input::get(layers::concat::LayerDataId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>
           ((*get(layers::backward::inputFromForward))[id]);
}

/**
* Sets input for the backward concat layer
* \param[in] id      Identifier of the input object
* \param[in] value   Pointer to the object
*/
void Input::set(layers::concat::LayerDataId id, const data_management::NumericTablePtr &value)
{
    services::SharedPtr<LayerData> layerData = get(layers::backward::inputFromForward);
    (*layerData)[id] = value;
}

/**
* Checks an input object for the layer algorithm
* \param[in] par     %Parameter of algorithm
* \param[in] method  Computation method of the algorithm
*/
services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *algParameter = static_cast<const Parameter *>(par);
    if (!algParameter->propagateGradient) { return services::Status(); }

    size_t concatDimension = algParameter->concatDimension;

    services::Status s;
    services::SharedPtr<data_management::Tensor> inputGradientTensor = get(layers::backward::inputGradient);
    DAAL_CHECK_STATUS(s, data_management::checkTensor(inputGradientTensor.get(), inputGradientStr()));
    services::Collection<size_t> inputGradientDims = inputGradientTensor->getDimensions();
    if (concatDimension > inputGradientDims.size() - 1) return services::Status(services::ErrorIncorrectParameter);

    if(Argument::size() != 2) return services::Status(services::ErrorIncorrectNumberOfInputNumericTables);

    services::SharedPtr<LayerData> layerData = get(layers::backward::inputFromForward);
    if (!layerData) return services::Status(services::ErrorNullLayerData);


    data_management::NumericTablePtr dimsNT = get(auxInputDimensions);
    if (dimsNT.get() == 0) return services::Status(services::ErrorNullNumericTable);
    if (dimsNT->getNumberOfRows() != 1) return services::Status(services::ErrorIncorrectNumberOfRowsInInputNumericTable);
    if (dimsNT->getNumberOfColumns() == 0) return services::Status(services::ErrorIncorrectNumberOfColumnsInInputNumericTable);

    size_t inputSize = dimsNT->getNumberOfColumns();

    data_management::BlockDescriptor<int> block;
    dimsNT->getBlockOfRows(0, 1, data_management::readOnly, block);
    int *auxDims = block.getBlockPtr();

    size_t sum = 0;
    for (size_t i = 0; i < inputSize; i++)
    {
        sum += auxDims[i];
    }

    if (inputGradientDims[concatDimension] != sum) return services::Status(services::ErrorIncorrectNumberOfRowsInInputNumericTable);
    inputGradientDims[concatDimension] = sum;

    DAAL_CHECK_STATUS(s, data_management::checkTensor(inputGradientTensor.get(), inputGradientStr(), &inputGradientDims));

    dimsNT->releaseBlockOfRows(block);
    return s;
}

    /** \brief Default constructor */
Result::Result() : layers::backward::Result() {};

/**
* Returns result object of the backward concat layer
* \param[in] id       Identifier of the result object
* \param[in] index    Index of the result object
* \return             %Input ResultLayerData that corresponds to the given identifier
*/
services::SharedPtr<data_management::Tensor> Result::get(layers::backward::ResultLayerDataId id, size_t index) const
{
    services::SharedPtr<LayerData> layerData = get(id);
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[index]);
}

/**
 * Sets result for the backward concat layer
 * \param[in] id       Identifier of the result object
 * \param[in] value    Pointer to the object
 * \param[in] index    Index of the result object
 */
void Result::set(layers::backward::ResultLayerDataId id, const services::SharedPtr<data_management::Tensor> &value, size_t index)
{
    services::SharedPtr<LayerData> layerData = get(id);
    (*layerData)[index] = value;
}

/**
 * Returns resulting gradient of the backward concat layer
 * \param[in] index Index of the tensor with gradient
 * \return Resulting gradient that corresponds to the given index
 */
services::SharedPtr<data_management::Tensor> Result::getGradient(size_t index) const
{
    return get(layers::backward::resultLayerData, index);
}

/**
 * Checks the result of the backward concat layer
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *parameter = static_cast<const Parameter *>(par);
    if (!parameter->propagateGradient) { return services::Status(); }

    if(Argument::size() != 4) return services::Status(services::ErrorIncorrectNumberOfInputNumericTables);

    const Input *algInput = static_cast<const Input *>(input);
    size_t concatDimension = parameter->concatDimension;
    services::SharedPtr<LayerData> layerData = get(layers::backward::resultLayerData);
    if (!layerData) return services::Status(services::ErrorNullLayerData);

    size_t nInputs = layerData->size();
    if (nInputs == 0) return services::Status(services::ErrorIncorrectSizeOfLayerData);

    services::SharedPtr<data_management::Tensor> inputGradientTensor = algInput->get(layers::backward::inputGradient);

    services::Collection<size_t> dims = inputGradientTensor->getDimensions();

    services::Status s;
    size_t sum = 0;
    for (size_t i = 0; i < nInputs; i++)
    {
        services::SharedPtr<data_management::Tensor> layerDataTensor = get(layers::backward::resultLayerData, i);
        dims[concatDimension] = layerDataTensor->getDimensionSize(concatDimension);
        sum += dims[concatDimension];

        DAAL_CHECK_STATUS(s, data_management::checkTensor(layerDataTensor.get(), resultLayerDataStr(), &dims));
    }
    if (sum != inputGradientTensor->getDimensionSize(concatDimension))
    {
        return services::Status(services::Error::create(services::ErrorIncorrectSizeOfDimensionInTensor, services::ArgumentName, inputGradientStr()));
    }
    return s;
}

/**
 * Returns the layout of the result object for the layer algorithm
 * \return Layout of the result object for the layer algorithm
 */
LayerResultLayout Result::getLayout() const  { return collectionResult; }

size_t Result::getElem(data_management::NumericTablePtr nt, size_t index) const
{
    data_management::BlockDescriptor<int> block;
    nt->getBlockOfRows(0, 1, data_management::readOnly, block);
    int *dataArray = block.getBlockPtr();
    nt->releaseBlockOfRows(block);
    return (size_t)dataArray[index];
}

}// namespace interface1
}// namespace backward
}// namespace concat
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
