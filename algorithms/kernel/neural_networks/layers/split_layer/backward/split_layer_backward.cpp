/* file: split_layer_backward.cpp */
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
//  Implementation of split calculation algorithm and types methods.
//--
*/

#include "split_layer_backward_types.h"
#include "split_layer_types.h"
#include "serialization_utils.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace split
{
namespace backward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_SPLIT_BACKWARD_RESULT_ID);
/** \brief Default constructor */
Input::Input()
{
    set(inputGradientCollection, services::SharedPtr<LayerData>(new LayerData()));
}

/**
 * Returns a tensor with a given index from the collection of input tensors
 * \param[in] id    Identifier of the collection of input tensors
 * \param[in] index Index of the tensor to be returned
 * \return          Pointer to the table with the input tensor
 */
services::SharedPtr<data_management::Tensor> Input::get(InputLayerDataId id, size_t index) const
{
    services::SharedPtr<layers::LayerData> layerData = get(id);
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[index]);
}

/**
 * Returns input Tensor of the layer algorithm
 * \param[in] id    Identifier of the input tensor
 * \return          %Input tensor that corresponds to the given identifier
 */
services::SharedPtr<LayerData> Input::get(InputLayerDataId id) const
{
    return services::staticPointerCast<LayerData, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for the backward split layer
 * \param[in] id     Identifier of the input object
 * \param[in] value  Pointer to the input object
 * \param[in] index  Index of the tensor to be set
 */
void Input::set(InputLayerDataId id, const services::SharedPtr<data_management::Tensor> &value, size_t index)
{
    services::SharedPtr<layers::LayerData> layerData = get(id);
    (*layerData)[index] = value;
}

/**
* Sets input for the layer algorithm
* \param[in] id    Identifier of the input object
* \param[in] ptr   Pointer to the object
*/
void Input::set(InputLayerDataId id, const services::SharedPtr<LayerData> &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Adds tensor with input gradient to the input object of the backward split layer
 * \param[in] igTensor  Tensor with input gradient
 * \param[in] index     Index of the tensor with input gradient
 *
 * \return Status of computations
 */
services::Status Input::addInputGradient(const services::SharedPtr<data_management::Tensor> &igTensor, size_t index)
{
    services::SharedPtr<LayerData> layerData = get(inputGradientCollection);

    size_t nInputs = layerData->size();
    (*(layerData))[nInputs] = igTensor;
    set(inputGradientCollection, layerData);

    return services::Status();
}

/**
 * Sets input structure retrieved from the result of the forward layer
 * \param[in] result Result of the forward layer
 */
services::Status Input::setInputFromForward(services::SharedPtr<layers::forward::Result> result)
{
    return services::Status();
}

/**
 * Checks an input object of the backward split layer
 * \param[in] par     Algorithm parameter
 * \param[in] method  Computation method
 *
 * \return Status of computations
 */
services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *parameter = static_cast<const Parameter *>(par);
    if (!parameter->propagateGradient) { return services::Status(); }

    DAAL_CHECK(Argument::size() == 2, services::ErrorIncorrectNumberOfInputNumericTables);

    services::SharedPtr<LayerData> layerData = get(inputGradientCollection);
    size_t nInputs = parameter->nInputs;

    DAAL_CHECK(layerData->size() == nInputs, services::ErrorIncorrectSizeOfLayerData);
    services::SharedPtr<data_management::Tensor> inputTensor0 = get(inputGradientCollection, 0);

    services::Status s = data_management::checkTensor(inputTensor0.get(), inputGradientCollectionStr());
    if(!s)
    {
        const size_t inx = s.getCollection()->size();
        services::KernelErrorCollectionPtr errors = s.getCollection()->getErrors();
        (*errors)[inx - 1]->addIntDetail(services::ArgumentName, (int)0);
        return s;
    }

    services::Collection<size_t> dims0 = inputTensor0->getDimensions();

    for (size_t i = 1; i < nInputs; i++)
    {
        s = data_management::checkTensor(get(inputGradientCollection, i).get(), inputGradientCollectionStr(), &dims0);
        if (!s)
        {
            const size_t inx = s.getCollection()->size();
            services::KernelErrorCollectionPtr errors = s.getCollection()->getErrors();
            (*errors)[inx - 1]->addIntDetail(services::ArgumentName, (int)i);
            return s;
        }
    }

    return s;
}

/**
* Returns the layout of the input object for the layer algorithm
* \return Layout of the input object for the layer algorithm
*/
LayerInputLayout Input::getLayout() const { return collectionInput; }

    /** \brief Default constructor */
Result::Result() : layers::backward::Result() {};

/**
 * Checks the result of the backward split layer
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 *
 * \return Status of computations
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *parameter = static_cast<const Parameter *>(par);
    if (!parameter->propagateGradient) { return services::Status(); }

    const Input *algInput = static_cast<const Input *>(input);

    DAAL_CHECK(Argument::size() == 4, services::ErrorIncorrectNumberOfInputNumericTables);

    services::SharedPtr<data_management::Tensor> inputTensor = algInput->get(inputGradientCollection, 0);
    services::Collection<size_t> dims = inputTensor->getDimensions();

    return data_management::checkTensor(get(layers::backward::gradient).get(), gradientStr(), &dims);
}

}// namespace interface1
}// namespace backward
}// namespace split
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
