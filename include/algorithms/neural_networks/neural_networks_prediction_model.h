/* file: neural_networks_prediction_model.h */
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
//  Implementation of neural network.
//--
*/

#ifndef __NEURAL_NETWORK_PREDICTION_MODEL_H__
#define __NEURAL_NETWORK_PREDICTION_MODEL_H__

#include "algorithms/algorithm.h"

#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/layer_types.h"

namespace daal
{
namespace algorithms
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__MODEL"></a>
 * \brief Contains classes for training and prediction using neural network
 */
/**
 * \brief Contains classes for training and prediction using neural network
 */
namespace neural_networks
{
namespace prediction
{
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__PARAMETER"></a>
 *  \brief Class representing the parameters of neural network prediction
 */
class Parameter : public daal::algorithms::Parameter
{};

/**
* <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__PREDICTION__MODEL"></a>
* \brief Class Model object for the prediction stage of neural network algorithm
*/
class Model : public daal::algorithms::Model
{
public:
    /** Default constructor */
    Model() : _forwardLayers(new neural_networks::ForwardLayers),
        _nextLayers(new services::Collection<layers::NextLayers>) {}

    /**
     * Constructs model object for the prediction stage of neural network
     * from the list of forward stages of the layers and the list of connections between the layers
     * \param[in] forwardLayers  List of forward stages of the layers
     * \param[in] nextLayers     List of next layers for each layer with corresponding index
     */
    Model(const services::SharedPtr<neural_networks::ForwardLayers> &forwardLayers,
          const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayers) :
        _forwardLayers(forwardLayers), _nextLayers(nextLayers) {}

    /** Copy constructor */
    Model(const Model &model) :
        _forwardLayers(model.getLayers()), _nextLayers(model.getNextLayers()), _weightsAndBiases(model.getWeightsAndBiases()) {}

    /**
     * Constructs model object for the prediction stage of neural network from a collection of layer descriptors
     * \param[in] layerDescriptors  Collection of layer descriptors of every inserted layer
     */
    Model(const services::Collection<layers::LayerDescriptor> &layerDescriptors) :
        _forwardLayers(new neural_networks::ForwardLayers),
        _nextLayers(new services::Collection<layers::NextLayers>)
    {
        for(size_t i = 0; i < layerDescriptors.size(); i++)
        {
            insertLayer(layerDescriptors[i]);
        }
    }

    /** \brief Destructor */
    virtual ~Model() {}

    /**
     * Insert a layer to a certain position in neural network
     * \param[in] layerDescriptor  %LayerDescriptor of the inserted layer
     */
    void insertLayer(const layers::LayerDescriptor &layerDescriptor)
    {
        _forwardLayers->insert(layerDescriptor.index, layerDescriptor.layer->forwardLayer);
        _nextLayers->insert(layerDescriptor.index, layerDescriptor.nextLayers);
    }

    /**
     * Sets list of forward stages of the layers and the list of connections between the layers
     * \param[in] forwardLayers  List of forward stages of the layers
     * \param[in] nextLayers     List of next layers for each layer with corresponding index
     */
    void setLayers(const services::SharedPtr<neural_networks::ForwardLayers> &forwardLayers,
                   const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayers)
    {
        _forwardLayers = forwardLayers;
        _nextLayers = nextLayers;
    }

    /**
     * Returns the list of forward stages of the layers
     * \return List of forward stages of the layers
     */
    const services::SharedPtr<neural_networks::ForwardLayers> getLayers() const
    {
        return _forwardLayers;
    }

    /**
     * Returns the forward stage of a layer with certain index in the network
     * \param[in] index  Index of the layer in the network
     * \return Forward stage of a layer with certain index in the network
     */
    const services::SharedPtr<layers::forward::LayerIface> getLayer(const size_t index) const
    {
        return _forwardLayers->get(index);
    }

    /**
     * Returns list of connections between layers
     * \return          List of next layers for each layer with corresponding index
     */
    const services::SharedPtr<services::Collection<layers::NextLayers> > getNextLayers() const
    {
        return _nextLayers;
    }

    /**
     * Sets table containing all neural network weights and biases
     * \param[in] weightsAndBiases          Table containing all neural network weights and biases
     */
    void setWeightsAndBiases(const services::SharedPtr<data_management::NumericTable> &weightsAndBiases)
    {
        _weightsAndBiases = weightsAndBiases;
    }

    /**
     * Returns table containing all neural network weights and biases
     * \return          Table containing all neural network weights and biases
     */
    const services::SharedPtr<data_management::NumericTable> getWeightsAndBiases() const
    {
        return _weightsAndBiases;
    }

    /**
     * Returns the serialization tag of the neural network model
     * \return         Serialization tag of the neural network model
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_PREDICTION_MODEL_ID; }
    /**
    *  Serializes an object
    *  \param[in]  arch  Storage for a serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive   *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes an object
    *  \param[in]  arch  Storage for a deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        // Model::serialImpl<Archive, onDeserialize>(arch);

        // arch->setSharedPtrObj(_forwardLayers);
        // arch->setSharedPtrObj(_nextLayers);
    }

private:
    services::SharedPtr<neural_networks::ForwardLayers> _forwardLayers; /*!< List of forward layers of the network */
    services::SharedPtr<services::Collection<layers::NextLayers> > _nextLayers; /*!< List of edges connecting the layers in the network */

    services::SharedPtr<data_management::NumericTable> _weightsAndBiases; /*!< Weights and biases of all the layers in the network */
};
} // namespace interface1
using interface1::Model;
using interface1::Parameter;
} // namespace prediction
} // namespace neural_networks
} // namespace algorithms
} //namespace daal

#endif
