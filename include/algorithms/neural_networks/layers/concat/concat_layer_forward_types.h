/* file: concat_layer_forward_types.h */
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
//  Implementation of the forward concat layer
//--
*/

#ifndef __NEURAL_NETWORKS__CONCAT_LAYER_FORWARD_TYPES_H__
#define __NEURAL_NETWORKS__CONCAT_LAYER_FORWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_forward_types.h"
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
 * \brief Contains classes for the forward concat layer
 */
namespace forward
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONCAT__FORWARD__INPUT"></a>
 * \brief %Input objects for the forward concat layer
 */
class Input : public layers::forward::Input
{
public:
    /** \brief Default constructor */
    Input() {};

    /**
    * Gets the input of the forward concat layer
    */
    using layers::forward::Input::get;

    /**
    * Sets the input of the forward concat layer
    */
    using layers::forward::Input::set;

    virtual ~Input() {}

    /**
    * Returns input Tensor of the forward concat layer
    * \param[in] id       Identifier of the input object
    * \param[in] index    Index of the input object
    * \return             %Input tensor that corresponds to the given identifier
    */
    services::SharedPtr<data_management::Tensor> get(layers::forward::InputLayerDataId id, size_t index) const
    {
        services::SharedPtr<LayerData> layerData = get(id);
        return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[index]);
    }

    /**
    * Returns input LayerData of the forward concat layer
    * \param[in] id    Identifier of the input object
    * \return          %Input InputLayerData that corresponds to the given identifier
    */
    services::SharedPtr<LayerData> get(layers::forward::InputLayerDataId id) const
    {
        return services::staticPointerCast<LayerData, data_management::SerializationIface>(Argument::get(id));
    }

    /**
    * Sets input for the forward concat layer
    * \param[in] id      Identifier of the input object
    * \param[in] value   Pointer to the object
    */
    void set(layers::forward::InputLayerDataId id, const services::SharedPtr<LayerData> &value)
    {
        Argument::set(id, value);
    }

    /**
    * Sets input for the forward concat layer
    * \param[in] id      Identifier of the input object
    * \param[in] value   Pointer to the object
    * \param[in] index   Index of the input object
    */
    void set(layers::forward::InputLayerDataId id, const services::SharedPtr<data_management::Tensor> &value, size_t index)
    {
        services::SharedPtr<LayerData> layerData = get(id);
        (*layerData)[index] = value;
    }

    /**
    * Checks an input object for the forward concat layer
    * \param[in] par     %Parameter of algorithm
    * \param[in] method  Computation method of the algorithm
    */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        const Parameter *parameter = static_cast<const Parameter *>(par);
        size_t concatDimension = parameter->concatDimension;
        services::SharedPtr<LayerData> layerData = get(layers::forward::inputLayerData);
        if (!layerData) { this->_errors->add(services::ErrorNullLayerData); return; }

        size_t nInputs = layerData->size();
        if (nInputs == 0) { this->_errors->add(services::ErrorIncorrectSizeOfLayerData); return; }
        services::SharedPtr<data_management::Tensor> layerDataTensor0 = get(layers::forward::inputLayerData, 0);

        if (!data_management::checkTensor(layerDataTensor0.get(), this->_errors.get(), strInputLayerData())) { return; }

        services::Collection<size_t> dims = layerDataTensor0->getDimensions();

        if (concatDimension > dims.size() - 1) {this->_errors->add(services::ErrorIncorrectParameter); return; }

        for (size_t i = 1; i < nInputs; i++)
        {
            services::SharedPtr<data_management::Tensor> layerDataTensor = get(layers::forward::inputLayerData, i);
            dims[concatDimension] = layerDataTensor->getDimensionSize(concatDimension);

            if (!data_management::checkTensor(layerDataTensor.get(), this->_errors.get(), strInputLayerData(), &dims)) { return; }
        }
    }

    /**
    * Returns the layout of the input object for the layer algorithm
    * \return Layout of the input object for the layer algorithm
    */
    LayerInputLayout getLayout() DAAL_C11_OVERRIDE { return collectionInput; }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONCAT__FORWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method of the forward concat layer
 */
class Result : public layers::forward::Result
{
public:
    /** \brief Default constructor */
    Result() : layers::forward::Result() {};
    virtual ~Result() {};

    /**
     * Returns the result of the forward concat layer
     */
    using layers::forward::Result::get;

    /**
    * Sets the result of the forward concat layer
    */
    using layers::forward::Result::set;

    /**
    * Sets the result of the forward concat layer
    * \param[in] id      Identifier of the result
    * \param[in] value   Pointer to the result
    */
    void set(LayerDataId id, const services::SharedPtr<data_management::NumericTable> &value)
    {
        (*get(layers::forward::resultForBackward))[id] = value;
    }

    /**
    * Returns input object of the forward concat layer
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    services::SharedPtr<data_management::NumericTable> get(layers::concat::LayerDataId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>
               ((*get(layers::forward::resultForBackward))[id]);
    }

    /**
    * Returns collection of dimensions of concat layer output
    * \param[in] inputSize   Collection of input tensors dimensions
    * \param[in] par         Parameters of the algorithm
    * \param[in] method      Method of the algorithm
    * \return    Collection of dimensions of concat layer output
    */
    virtual const services::Collection<size_t> getValueSize(const services::Collection<size_t> &inputSize,
                                                            const daal::algorithms::Parameter *par, const int method) const DAAL_C11_OVERRIDE
    {
        return services::Collection<size_t>();
    }

    /**
    * Returns collection of dimensions of concat layer output
    * \param[in] inputSize   Collection of input tensors dimensions
    * \param[in] parameter   Parameters of the algorithm
    * \param[in] method      Method of the algorithm
    * \return    Collection of dimensions of concat layer output
    */
    services::Collection<size_t> getValueSize(const services::Collection< services::Collection<size_t> > &inputSize,
                                              const daal::algorithms::Parameter *parameter, const int method) DAAL_C11_OVERRIDE
    {
        const Parameter *par = static_cast<const Parameter *>(parameter);

        size_t nInputs = inputSize.size();
        size_t concatDimension = par->concatDimension;

        size_t sum = 0;
        for (size_t i = 0; i < nInputs; i++)
        {
            sum += inputSize[i][concatDimension];
        }

        services::Collection<size_t> dimsCollection = inputSize[0];
        dimsCollection[concatDimension] = sum;

        return dimsCollection;
    }


    /**
    * Allocates memory to store the result of the forward concat layer
    * \param[in] input     Pointer to an object containing the input data
    * \param[in] parameter %Parameter of the algorithm
    * \param[in] method    Computation method for the algorithm
    */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const Input *in = static_cast<const Input * >(input);
        const Parameter *par = static_cast<const Parameter *>(parameter);

        services::SharedPtr<data_management::Tensor> valueTable = in->get(layers::forward::inputLayerData, 0);

        size_t nInputs = in->get(layers::forward::inputLayerData)->size();
        size_t concatDimension = par->concatDimension;

        size_t sum = 0;
        for (size_t i = 0; i < nInputs; i++)
        {
            size_t dim = (in->get(layers::forward::inputLayerData, i))->getDimensionSize(concatDimension);
            sum += dim;
        }
        if(!valueTable) { this->_errors->add(services::ErrorNullInputNumericTable); return; }

        services::Collection<size_t> dimsCollection = valueTable->getDimensions();
        dimsCollection[concatDimension] = sum;

        if (!get(layers::forward::value))
        {
            set(layers::forward::value, services::SharedPtr<data_management::Tensor>(
                          new data_management::HomogenTensor<algorithmFPType>(dimsCollection,
                                                                              data_management::Tensor::doAllocate)));
        }
        set(layers::forward::resultForBackward, services::SharedPtr<LayerData>(new LayerData()));

        services::SharedPtr<data_management::HomogenNumericTable<size_t> > auxDimTable(new data_management::HomogenNumericTable<size_t>
                                                                                       (nInputs, 1, data_management::NumericTable::doAllocate));
        size_t *auxDimArray = auxDimTable->getArray();

        for (size_t i = 0; i < nInputs; i++)
        {
            size_t dim = (in->get(layers::forward::inputLayerData, i))->getDimensionSize(concatDimension);
            auxDimArray[i] = dim;
        }

        set(layers::concat::auxInputDimensions, auxDimTable);
    }

    /**
    * Checks the result object for the layer algorithm
    * \param[in] input         %Input of the algorithm
    * \param[in] parameter     %Parameter of algorithm
    * \param[in] method        Computation method of the algorithm
    */
    virtual void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
                       int method) const DAAL_C11_OVERRIDE
    {
        const Input *algInput = static_cast<const Input *>(input);
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);
        size_t concatDimension = algParameter->concatDimension;

        if(Argument::size() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        services::SharedPtr<LayerData> layerData = get(layers::forward::resultForBackward);
        if (!layerData) { this->_errors->add(services::ErrorNullLayerData); return; }

        services::SharedPtr<LayerData> inputLayerData = algInput->get(layers::forward::inputLayerData);
        size_t inputSize = inputLayerData->size();
        services::SharedPtr<data_management::NumericTable> dimsNT = get(auxInputDimensions);
        if (dimsNT->getNumberOfColumns() != inputSize) { this->_errors->add(services::ErrorIncorrectNumberOfColumnsInOutputNumericTable); return; }
        if (dimsNT->getNumberOfRows() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfRowsInOutputNumericTable); return; }

        size_t sum = 0;
        for (size_t i = 0; i < inputSize; i++)
        {
            services::SharedPtr<data_management::Tensor> inputTensor = algInput->get(layers::forward::inputLayerData, i);
            size_t dim = inputTensor->getDimensionSize(concatDimension);

            sum += dim;
        }

        services::SharedPtr<data_management::Tensor> valueTensor = get(layers::forward::value);
        services::Collection<size_t> dims = algInput->get(layers::forward::inputLayerData, 0)->getDimensions();
        dims[concatDimension] = sum;

        if (!data_management::checkTensor(get(layers::forward::value).get(), this->_errors.get(), strValue(), &dims)) { return; }
    }

    /**
    * Returns the serialization tag of the result
    * \return     Serialization tag of the result
    */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_LAYERS_CONCAT_FORWARD_RESULT_ID; }

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

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
} // namespace interface1
using interface1::Input;
using interface1::Result;
} // namespace forward
} // namespace concat
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
