/* file: concat_layer_backward_types.h */
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
//  Implementation of the backward concat layer
//--
*/

#ifndef __NEURAL_NETWORKS__CONCAT_LAYER_BACKWARD_TYPES_H__
#define __NEURAL_NETWORKS__CONCAT_LAYER_BACKWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_backward_types.h"
#include "algorithms/neural_networks/layers/concat/concat_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * \brief Contains classes for the concat layer
 */
namespace concat
{
/**
 * \brief Contains classes for the backward concat layer
 */
namespace backward
{
/**
* \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
*/
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONCAT__BACKWARD__INPUT"></a>
 * \brief %Input parameters for the backward concat layer
 */
class Input : public layers::backward::Input
{
public:
    /** \brief Default constructor */
    Input() {};

    virtual ~Input() {}

    /**
     * Returns an input object for the backward concat layer
     */
    using layers::backward::Input::get;

    /**
     * Sets an input object for the backward concat layer
     */
    using layers::backward::Input::set;

    /**
    * Returns input object of the backward concat layer
    * \param[in] id    Identifier of the input object
    * \return          %Input LayerData that corresponds to the given identifier
    */
    services::SharedPtr<data_management::NumericTable> get(layers::concat::LayerDataId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>
               ((*get(layers::backward::inputFromForward))[id]);
    }

    /**
    * Sets input for the backward concat layer
    * \param[in] id      Identifier of the input object
    * \param[in] value   Pointer to the object
    */
    void set(layers::concat::LayerDataId id, const services::SharedPtr<data_management::NumericTable> &value)
    {
        services::SharedPtr<LayerData> layerData = get(layers::backward::inputFromForward);
        (*layerData)[id] = value;
    }

    /**
    * Checks an input object for the layer algorithm
    * \param[in] par     %Parameter of algorithm
    * \param[in] method  Computation method of the algorithm
    */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        const Parameter *algParameter = static_cast<const Parameter *>(par);
        size_t concatDimension = algParameter->concatDimension;

        services::SharedPtr<data_management::Tensor> inputGradientTensor = get(layers::backward::inputGradient);
        services::SharedPtr<services::Error> error = checkTensor(inputGradientTensor, "inputGradient");
        if (error) { this->_errors->add(error); return; }
        services::Collection<size_t> inputGradientDims = inputGradientTensor->getDimensions();
        if (concatDimension > inputGradientDims.size() - 1) {this->_errors->add(services::ErrorIncorrectParameter); return; }

        if(Argument::size() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        services::SharedPtr<LayerData> layerData = get(layers::backward::inputFromForward);
        if (!layerData) { this->_errors->add(services::ErrorNullLayerData); return; }


        services::SharedPtr<data_management::NumericTable> dimsNT = get(auxInputDimensions);
        if (dimsNT->getNumberOfRows() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }
        if (dimsNT->getNumberOfColumns() == 0) { this->_errors->add(services::ErrorIncorrectNumberOfColumnsInInputNumericTable); return; }

        size_t inputSize = dimsNT->getNumberOfColumns();

        data_management::BlockDescriptor<int> block;
        dimsNT->getBlockOfRows(0, 1, data_management::readOnly, block);
        int *auxDims = block.getBlockPtr();

        size_t sum = 0;
        for (size_t i = 0; i < inputSize; i++)
        {
            sum += auxDims[i];
        }

        if (inputGradientDims[concatDimension] != sum) { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }
        inputGradientDims[concatDimension] = sum;

        error = checkTensor(inputGradientTensor, "inputGradient", &inputGradientDims);
        if (error) { this->_errors->add(error); return; }
        dimsNT->releaseBlockOfRows(block);
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONCAT__BACKWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method of the backward concat layer
 */
class Result : public layers::backward::Result
{
public:
    /** \brief Default constructor */
    Result() : layers::backward::Result() {};
    virtual ~Result() {};

    /**
     * Returns the result of the backward concat layer
     */
    using layers::backward::Result::get;

    /**
     * Sets the result of the backward concat layer
     */
    using layers::backward::Result::set;

    /**
    * Returns result object of the backward concat layer
    * \param[in] id       Identifier of the result object
    * \param[in] index    Index of the result object
    * \return             %Input ResultLayerData that corresponds to the given identifier
    */
    services::SharedPtr<data_management::Tensor> get(layers::backward::ResultLayerDataId id, size_t index) const
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
    void set(layers::backward::ResultLayerDataId id, const services::SharedPtr<data_management::Tensor> &value, size_t index)
    {
        services::SharedPtr<LayerData> layerData = get(id);
        (*layerData)[index] = value;
    }

    /**
     * Checks the result of the backward concat layer
     * \param[in] input   %Input object for the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 4) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        const Input *algInput = static_cast<const Input *>(input);
        const Parameter *parameter = static_cast<const Parameter *>(par);
        size_t concatDimension = parameter->concatDimension;
        services::SharedPtr<LayerData> layerData = get(layers::backward::resultLayerData);
        if (!layerData) { this->_errors->add(services::ErrorNullLayerData); return; }

        size_t nInputs = layerData->size();
        if (nInputs == 0) { this->_errors->add(services::ErrorIncorrectSizeOfLayerData); return; }

        services::SharedPtr<data_management::Tensor> inputGradientTensor = algInput->get(layers::backward::inputGradient);

        services::Collection<size_t> dims = inputGradientTensor->getDimensions();

        size_t sum = 0;
        for (size_t i = 0; i < nInputs; i++)
        {
            services::SharedPtr<data_management::Tensor> layerDataTensor = get(layers::backward::resultLayerData, i);
            dims[concatDimension] = layerDataTensor->getDimensionSize(concatDimension);
            sum += dims[concatDimension];
            services::SharedPtr<services::Error> error = checkTensor(layerDataTensor, "resultLayerData", &dims);
            if (error) { this->_errors->add(error); return; }
        }
        if (sum != inputGradientTensor->getDimensionSize(concatDimension))
        {
            services::SharedPtr<services::Error> error = services::SharedPtr<services::Error>(new services::Error(
                                                                                                  services::ErrorIncorrectSizeOfDimensionInTensor));
            error->addStringDetail(services::ArgumentName, "inputGradient");
            this->_errors->add(error);
            return;
        }
    }

    /**
    * Allocates memory to store the result of the backward concat layer
     * \param[in] input     Pointer to an object containing the input data
     * \param[in] method    Computation method for the algorithm
     * \param[in] parameter %Parameter of the backward concat layer
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const Input *in = static_cast<const Input * >(input);
        const Parameter *par = static_cast<const Parameter *>(parameter);

        size_t concatDimension = par->concatDimension;

        size_t nOutputs = (in->get(layers::concat::auxInputDimensions))->getNumberOfColumns();

        services::SharedPtr<LayerData> resultCollection = services::SharedPtr<LayerData>(new LayerData());

        services::Collection<size_t> dimsCollection = in->get(layers::backward::inputGradient)->getDimensions();

        for(size_t i = 0; i < nOutputs; i++)
        {
            services::SharedPtr<data_management::NumericTable> dimsTable = in->get(layers::concat::auxInputDimensions);

            dimsCollection[concatDimension] = getElem(dimsTable, i);
            (*resultCollection)[i] = services::SharedPtr<data_management::Tensor>(new data_management::HomogenTensor<algorithmFPType>(
                                                                                      dimsCollection, data_management::Tensor::doAllocate));
        }
        Argument::set(layers::backward::resultLayerData, services::staticPointerCast<data_management::SerializationIface, LayerData>
                      (resultCollection));
    }

    /**
    * Returns the serialization tag of the result
    * \return     Serialization tag of the result
    */
    int getSerializationTag() { return SERIALIZATION_NEURAL_NETWORKS_LAYERS_CONCAT_BACKWARD_RESULT_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for the serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes the object
    *  \param[in]  arch  Storage for the deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

    /**
     * Returns the layout of the result object for the layer algorithm
     * \return Layout of the result object for the layer algorithm
     */
    virtual LayerResultLayout getLayout() const { return collectionResult; }

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }

    size_t getElem(services::SharedPtr<data_management::NumericTable> nt, size_t index) const
    {
        data_management::BlockDescriptor<int> block;
        nt->getBlockOfRows(0, 1, data_management::readOnly, block);
        int *dataArray = block.getBlockPtr();
        nt->releaseBlockOfRows(block);
        return (size_t)dataArray[index];
    }
};
} // namespace interface1
using interface1::Input;
using interface1::Result;
} // namespace backward

} // namespace concat
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
