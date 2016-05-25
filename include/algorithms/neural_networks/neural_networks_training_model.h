/* file: neural_networks_training_model.h */
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

#ifndef __NEURAL_NETWORK_TRAINING_MODEL_H__
#define __NEURAL_NETWORK_TRAINING_MODEL_H__

#include "algorithms/algorithm.h"

#include "data_management/data/data_dictionary.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/layer_types.h"
#include "algorithms/neural_networks/layers/softmax/softmax_layer_forward.h"
#include "algorithms/neural_networks/neural_networks_prediction_model.h"
#include "algorithms/neural_networks/neural_networks_training_input.h"

#include "algorithms/optimization_solver/iterative_solver/iterative_solver_batch.h"
#include "algorithms/optimization_solver/sgd/sgd_batch.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for training and prediction using neural network
 */
namespace neural_networks
{
namespace training
{
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__PARAMETER"></a>
 *  \brief Class representing the parameters of neural network
 */
class Parameter : public daal::algorithms::Parameter
{
public:
    /**
     * Constructs the parameters of neural network algorithm
     * \param[in] batchSize_                  Size of the batch to be processed by the neural network
     * \param[in] optimizationSolver_         Optimization solver used in the neural network
     */
    Parameter(size_t batchSize_ = 1,
              services::SharedPtr<optimization_solver::iterative_solver::Batch > optimizationSolver_ =
                  services::SharedPtr<optimization_solver::iterative_solver::Batch>(new optimization_solver::sgd::Batch<float>())) :
        batchSize(batchSize_), optimizationSolver(optimizationSolver_) {};

    size_t batchSize; /*!< Size of the batch to be processed by the neural network. */

    services::SharedPtr<optimization_solver::iterative_solver::Batch>  optimizationSolver; /*!< Optimization solver used in the neural network*/
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__MODEL"></a>
 *  \brief Class representing the model of neural network
 */
class Model : public daal::algorithms::Model
{
public:
    /** \brief Constructor */
    Model() :
        _forwardLayers(new ForwardLayers()),
        _backwardLayers(new BackwardLayers()),
        _nextLayers(new services::Collection<layers::NextLayers>()),
        _errors() {}

    /** \brief Copy constructor */
    Model(const Model &model) :
        _forwardLayers(model.getForwardLayers()),
        _backwardLayers(model.getBackwardLayers()),
        _nextLayers(model.getNextLayers()),
        _errors(model.getErrors()) {}

    /** \brief Destructor */
    virtual ~Model() {};

    /**
     * Insert a layer to a certain position in neural network
     * \param[in] layerDescriptor  %LayerDescriptor of the inserted layer
     */
    void insertLayer(const layers::LayerDescriptor &layerDescriptor)
    {
        _forwardLayers->insert(layerDescriptor.index, layerDescriptor.layer->forwardLayer);
        _backwardLayers->insert(layerDescriptor.index, layerDescriptor.layer->backwardLayer);
        _nextLayers->insert(layerDescriptor.index, layerDescriptor.nextLayers);
    }

    /**
     * Insert a collection of layers to a certain position in neural network
     * \param[in] dataSize          Dimensionality of the training data
     * \param[in] layerDescriptors  Collection of %LayerDescriptor of every inserted layer
     * \param[in] parameter         Parameters of the training
     */
    template<typename modelFPType>
    void initialize(const services::Collection<size_t> &dataSize, const services::Collection<layers::LayerDescriptor> &layerDescriptors,
                    const Parameter *parameter)
    {
        for(size_t i = 0; i < layerDescriptors.size(); i++)
        {
            insertLayer(layerDescriptors[i]);
        }
        allocate<modelFPType>(dataSize, parameter);
    }

    /**
     * Returns list of forward layers
     * \return          List of forward layers
     */
    const services::SharedPtr<ForwardLayers> getForwardLayers() const
    {
        return _forwardLayers;
    }

    /**
     * Returns the forward stage of a layer with certain index in the network
     * \param[in] index  Index of the layer in the network
     * \return Forward stage of a layer with certain index in the network
     */
    const services::SharedPtr<layers::forward::LayerIface> getForwardLayer(const size_t index) const
    {
        return _forwardLayers->get(index);
    }

    /**
     * Returns list of backward layers
     * \return          List of backward layers
     */
    const services::SharedPtr<BackwardLayers> getBackwardLayers() const
    {
        return _backwardLayers;
    }

    /**
     * Returns the backward stage of a layer with certain index in the network
     * \param[in] index  Index of the layer in the network
     * \return Backward stage of a layer with certain index in the network
     */
    const services::SharedPtr<layers::backward::LayerIface> getBackwardLayer(const size_t index) const
    {
        services::SharedPtr<layers::backward::LayerIface> layer = _backwardLayers->get(index);
        return layer;
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
     * Returns list of forward layers and their parameters organised in the prediction::Model
     * \return          List of forward layers and their parameters organised in the prediction::Model
     */
    template<typename modelFPType>
    const services::SharedPtr<prediction::Model> getPredictionModel()
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        size_t nLayers = _forwardLayers->size();

        Collection<size_t> sampleSize(_sampleSize);
        sampleSize[0] = 1;

        /* Copy forward layers */
        SharedPtr<ForwardLayers> _predictionForwardLayers(new ForwardLayers(nLayers));
        SharedPtr<Collection<NextLayers> > _predictionNextLayers(new Collection<NextLayers>(nLayers));
        for (size_t i = 0; i < nLayers; i++)
        {
            (*_predictionForwardLayers)[i] = _forwardLayers->get(i)->clone();
            (*_predictionNextLayers)[i] = _nextLayers->get(i);
        }
        SharedPtr<softmax::forward::Batch<modelFPType> > lastSoftmax(new softmax::forward::Batch<modelFPType>());
        lastSoftmax->parameter.dimension = 1;
        (*_predictionForwardLayers)[nLayers-1] = lastSoftmax;

        /* Clear layers' inputs if needed */
        for (size_t i = 0; i < nLayers; i++)
        {
            layers::forward::Input *forwardInput = _predictionForwardLayers->get(i)->getLayerInput();
            if (forwardInput->getLayout() == collectionInput)
            {
                forwardInput->set(forward::inputLayerData, SharedPtr<LayerData>());
            }
        }

        /* Connect layers results and next layers inputs */
        _predictionForwardLayers->get(0)->getLayerInput()->set(forward::data,
                                                     SharedPtr<Tensor>(new HomogenTensor<modelFPType>(sampleSize, Tensor::doAllocate)));
        for (size_t i = 0; i < nLayers; i++)
        {
            _predictionForwardLayers->get(i)->allocateResult();
            connectLayerResultAndNextLayerInput(i, _predictionForwardLayers, _predictionNextLayers);
        }

        prediction::Model *predictionModel = new prediction::Model(_predictionForwardLayers, _predictionNextLayers);

        SharedPtr<HomogenNumericTable<modelFPType> > weightsAndBiasesTable = tensorsToTable<modelFPType>(_weightsAndBiasesTensors);
        setWeightsAndBiasesFromTable<modelFPType>(weightsAndBiasesTable.get(), _predictionForwardLayers);
        predictionModel->setWeightsAndBiases(weightsAndBiasesTable);

        return services::SharedPtr<prediction::Model>(predictionModel);
    }

    /**
     * Returns weights and biases storage status
     * \return Weights and biases storage status.
     * True if weights and biases of all layers stored in one numeric table. False otherwise.
     */
    bool getWeightsAndBiasesStorageStatus()
    {
        return _storeWeightsInTable;
    }

    /**
     * Returns the weights and biases of all forward layers of neural network as numeric table
     * \return   Weights and biases container
     */
    template<typename modelFPType>
    services::SharedPtr<data_management::NumericTable> getAllWeightsAndBiases() const
    {
        if (_storeWeightsInTable)
        {
            return _weightsAndBiasesTable;
        }
        else
        {
            return tensorsToTable<modelFPType>(_weightsAndBiasesTensors);
        }
    }

    /**
     * Returns the weights and biases of all forward layers of neural network as numeric table
     * \return   Weights and biases container
     */
    template<typename modelFPType>
    services::SharedPtr<data_management::NumericTable> getAllWeightsAndBiasesDerivatives() const
    {
        if (_storeWeightsInTable)
        {
            return _weightsAndBiasesDerivativesTable;
        }
        else
        {
            return tensorsToTable<modelFPType>(_weightsAndBiasesDerivativesTensors);
        }
    }

    /**
     * Sets the weights and biases of all forward layers of neural network from numeric table
     * \param[in] table Numeric table that stores weights and biases values for all layers
     */
    template<typename modelFPType>
    void setAllWeightsAndBiases(const services::SharedPtr<data_management::NumericTable> &table)
    {
        using namespace data_management;

        if (_storeWeightsInTable)
        {
            if (table.get() == _weightsAndBiasesTable.get()) { return; }

            BlockDescriptor<modelFPType> srcBlock, dstBlock;
            table->getBlockOfRows(0, 1, readOnly, srcBlock);
            _weightsAndBiasesTable->getBlockOfRows(0, 1, writeOnly, dstBlock);
            modelFPType *srcArray = srcBlock.getBlockPtr();
            modelFPType *dstArray = dstBlock.getBlockPtr();
            size_t blockSize = _weightsAndBiasesTable->getNumberOfColumns() * sizeof(modelFPType);
            services::daal_memcpy_s(dstArray, blockSize, srcArray, blockSize);
            _weightsAndBiasesTable->releaseBlockOfRows(dstBlock);
            table->releaseBlockOfRows(srcBlock);
        }
        else
        {
            tableToTensors<modelFPType>(table, _weightsAndBiasesTensors, 0, 2 * _forwardLayers->size());
        }
    }

    /**
     * Returns the weights and biases of the forward layer of neural network as numeric table
     * \param[in] idx Index of the forward layer
     * \return   Weights and biases container
     */
    template<typename modelFPType>
    services::SharedPtr<data_management::NumericTable> getWeightsAndBiases(size_t idx) const
    {
        using namespace data_management;

        if (idx > _forwardLayers->size() || ((_weightsSize[idx] + _biasesSize[idx]) == 0))
        {
            return services::SharedPtr<NumericTable>();
        }

        return tensorsToTable<modelFPType>(_weightsAndBiasesTensors, 2*idx, 2);
    }

    /**
     * Sets the weights and biases of the forward layer of neural network from numeric table
     * \param[in] idx   Index of the forward layer
     * \param[in] table Numeric table that stores weights and biases values for the layer
     */
    template<typename modelFPType>
    void setWeightsAndBiases(size_t idx, const services::SharedPtr<data_management::NumericTable> &table)
    {
        using namespace data_management;

        if (idx > _forwardLayers->size()) { return; }

        size_t weightsAndBiasesSize = _weightsSize[idx] + _biasesSize[idx];
        if (weightsAndBiasesSize == 0) { return; }

        if (_storeWeightsInTable)
        {
            BlockDescriptor<modelFPType> srcBlock, dstBlock;
            table->getBlockOfRows(0, 1, readOnly, srcBlock);
            _weightsAndBiasesTable->getBlockOfRows(0, 1, writeOnly, dstBlock);
            modelFPType *wbFullArray = dstBlock.getBlockPtr() + _weightsOffsets[idx];
            modelFPType *wbArray = srcBlock.getBlockPtr();

            size_t blockSize = weightsAndBiasesSize * sizeof(modelFPType);
            services::daal_memcpy_s(wbFullArray, blockSize, wbArray, blockSize);
            _weightsAndBiasesTable->releaseBlockOfRows(dstBlock);
            table->releaseBlockOfRows(srcBlock);
        }
        else
        {
            tableToTensors<modelFPType>(table, _weightsAndBiasesTensors, 2*idx, 2);
        }
    }

    /**
     * Returns the weights and biases derivatives of the backward layer of neural network as numeric table
     * \param[in] idx Index of the backward layer
     * \return   Weights and biases derivatives container
     */
    template<typename modelFPType>
    services::SharedPtr<data_management::NumericTable> getWeightsAndBiasesDerivatives(size_t idx) const
    {
        using namespace data_management;

        if (idx > _backwardLayers->size() || ((_weightsSize[idx] + _biasesSize[idx]) == 0))
        {
            return services::SharedPtr<NumericTable>();
        }

        return tensorsToTable<modelFPType>(_weightsAndBiasesDerivativesTensors, 2*idx, 2);
    }

    /**
     * Sets the error collection to the Model
     * \param[in] errors  Collection of errors
     */
    void setErrors(services::ErrorCollection &errors) { _errors = errors; }

    /**
     * Returns the errors of the Model
     * \return   Collection of errors
     */
    const services::ErrorCollection &getErrors() const { return _errors; }

    /**
     * Allocates the buffers needed for the training using neural network
     * \param[in] dataSize         Size of the input data for the training
     * \param[in] parameter        Parameters of the training
     */
    template<typename modelFPType>
    void allocate(const services::Collection<size_t> &dataSize, const daal::algorithms::Parameter *parameter)
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        Parameter *par = const_cast<Parameter *>(static_cast<const Parameter *>(parameter));

        if (_sampleSize.size() > 0) { _sampleSize.clear(); }
        _sampleSize = dataSize;
        _sampleSize[0] = par->batchSize;

        _forwardLayers->get(0)->getLayerInput()->set(forward::data,
                                                     SharedPtr<Tensor>(new HomogenTensor<modelFPType>(_sampleSize, Tensor::doAllocate)));

        computeWeightsAndBiasesSizes();

        allocateAndSetWeightsAndBiases<modelFPType>();

        size_t nLayers = _forwardLayers->size();
        for (size_t i = 0; i < nLayers; i++)
        {
            setBackwardLayerInputsAndResults(_forwardLayers->get(i), _backwardLayers->get(i));
        }

        allocateAndSetWeightsAndBiasesDerivatives<modelFPType>();
    }

    /**
     * Returns the serialization tag of the neural network model
     * \return         Serialization tag of the neural network model
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_TRAINING_MODEL_ID; }

    /**
    *  Serializes an object
    *  \param[in]  arch  Storage for a serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
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
        // arch->setSharedPtrObj(_backwardLayers);
        // arch->setSharedPtrObj(_nextLayers);
    }

    size_t getCollectionSize(const services::Collection<size_t> &collection)
    {
        if(collection.size() == 0) { return 0; }

        size_t size = 1;
        for(size_t i = 0; i < collection.size(); i++)
        {
            size *= collection[i];
        }
        return size;
    }

    void computeWeightsAndBiasesSizes()
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        _weightsAndBiasesSize = 0;
        _storeWeightsInTable = true;

        size_t nLayers = _forwardLayers->size();
        size_t weightsSize, biasesSize;

        for (size_t i = 0; i < nLayers; i++)
        {
            SharedPtr<forward::LayerIface> forwardLayer = _forwardLayers->get(i);
            SharedPtr<backward::LayerIface> backwardLayer = _backwardLayers->get(i);
            forward::Input *forwardInput = forwardLayer->getLayerInput();
            SharedPtr<backward::Result> backwardResult = backwardLayer->getLayerResult();
            layers::Parameter *parameter = forwardLayer->getLayerParameter();

            /* Check if weights, biases or derivative are allocated by user */
            if (forwardInput->get(forward::weights) || forwardInput->get(forward::biases) ||
                backwardResult->get(backward::weightDerivatives) || backwardResult->get(backward::biasDerivatives))
            {
                _storeWeightsInTable = false;
            }

            _weightsDimsCollection.push_back(forwardInput->getWeightsSizes(parameter));
            _biasesDimsCollection .push_back(forwardInput->getBiasesSizes(parameter));
            weightsSize = getCollectionSize(_weightsDimsCollection[i]);
            biasesSize  = getCollectionSize(_biasesDimsCollection[i]);

            _weightsOffsets.push_back(_weightsAndBiasesSize);
            _weightsAndBiasesSize += weightsSize;
            _weightsSize.push_back(weightsSize);
            _biasesOffsets .push_back(_weightsAndBiasesSize);
            _weightsAndBiasesSize += biasesSize;
            _biasesSize .push_back(biasesSize);

            forwardLayer->allocateResult();
            connectLayerResultAndNextLayerInput(i, _forwardLayers, _nextLayers);
        }
    }

    template<typename modelFPType>
    void allocateAndSetWeightsAndBiases()
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        size_t nLayers = _forwardLayers->size();
        if (_storeWeightsInTable)
        {
            HomogenNumericTable<modelFPType> *weightsAndBiasesTablePtr = allocateParameterTable<modelFPType>(_weightsAndBiasesSize, (size_t)1);
            _weightsAndBiasesTable = SharedPtr<NumericTable>(weightsAndBiasesTablePtr);

            setWeightsAndBiasesFromTable<modelFPType>(weightsAndBiasesTablePtr, _forwardLayers);
        }
        else
        {
            for(size_t i = 0; i < nLayers; i++)
            {
                SharedPtr<forward::LayerIface> forwardLayer = _forwardLayers->get(i);

                forwardLayer->allocateInput();
            }
        }
        _weightsAndBiasesTensors.clear();
        for(size_t i = 0; i < nLayers; i++)
        {
            SharedPtr<forward::LayerIface> forwardLayer = _forwardLayers->get(i);
            forwardLayer->initializeInput();
            forward::Input *input = forwardLayer->getLayerInput();
            _weightsAndBiasesTensors.push_back(input->get(forward::weights));
            _weightsAndBiasesTensors.push_back(input->get(forward::biases));
            forwardLayer->getLayerResult()->setResultForBackward(input);
        }
    }

    template<typename modelFPType>
    void setWeightsAndBiasesFromTable(data_management::HomogenNumericTable<modelFPType> *weightsAndBiasesTable,
            services::SharedPtr<ForwardLayers> &forwardLayers)
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        size_t nLayers = forwardLayers->size();
        modelFPType *weightsAndBiasesArray = weightsAndBiasesTable->getArray();

        for(size_t i = 0; i < nLayers; i++)
        {
            SharedPtr<forward::LayerIface> forwardLayer = forwardLayers->get(i);
            forward::Input *input = forwardLayer->getLayerInput();

            SharedPtr<Tensor> weights(new HomogenTensor<modelFPType>(_weightsDimsCollection[i], weightsAndBiasesArray + _weightsOffsets[i]));
            SharedPtr<Tensor> biases(new HomogenTensor<modelFPType>(_biasesDimsCollection[i], weightsAndBiasesArray + _biasesOffsets[i]));
            input->set(forward::weights, weights);
            input->set(forward::biases, biases);
        }
    }

    template<typename modelFPType>
    void allocateAndSetWeightsAndBiasesDerivatives()
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        size_t nLayers = _backwardLayers->size();

        if (_storeWeightsInTable)
        {
            HomogenNumericTable<modelFPType> *weightsAndBiasesDerivativesTablePtr =
                allocateParameterTable<modelFPType>(_weightsAndBiasesSize, (size_t)1);
            _weightsAndBiasesDerivativesTable = SharedPtr<NumericTable>(weightsAndBiasesDerivativesTablePtr);

            modelFPType *weightsAndBiasesDerivativesArray = weightsAndBiasesDerivativesTablePtr->getArray();

            for(size_t i = 0; i < nLayers; i++)
            {
                SharedPtr<backward::Result> backwardResult = _backwardLayers->get(i)->getLayerResult();

                SharedPtr<Tensor> weightDerivatives(
                    new HomogenTensor<modelFPType>(_weightsDimsCollection[i], weightsAndBiasesDerivativesArray + _weightsOffsets[i]));
                SharedPtr<Tensor> biasDerivatives(
                    new HomogenTensor<modelFPType>(_biasesDimsCollection[i], weightsAndBiasesDerivativesArray + _biasesOffsets[i]));

                backwardResult->set(backward::weightDerivatives, weightDerivatives);
                backwardResult->set(backward::biasDerivatives, biasDerivatives);
            }
        }
        else
        {
            for(size_t i = 0; i < nLayers; i++)
            {
                _backwardLayers->get(i)->allocateResult();
            }
        }

        _weightsAndBiasesDerivativesTensors.clear();
        for(size_t i = 0; i < nLayers; i++)
        {
            SharedPtr<backward::Result> backwardResult = _backwardLayers->get(i)->getLayerResult();
            _weightsAndBiasesDerivativesTensors.push_back(backwardResult->get(backward::weightDerivatives));
            _weightsAndBiasesDerivativesTensors.push_back(backwardResult->get(backward::biasDerivatives));
        }
    }

    void connectLayerResultAndNextLayerInput(size_t layerId, services::SharedPtr<ForwardLayers> &forwardLayers,
                                             const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayers)
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        services::SharedPtr<layers::forward::LayerIface> &forwardLayer = forwardLayers->get(layerId);
        const NextLayers &next = nextLayers->get(layerId);
        SharedPtr<forward::Result> forwardResult = forwardLayer->getLayerResult();
        LayerResultLayout resultLayout = forwardResult->getLayout();
        if(resultLayout == tensorResult)
        {
            SharedPtr<Tensor> valueTensor = forwardResult->get(forward::value);

            for(size_t j = 0; j < next.size(); j++)
            {
                setNextLayerInput(forwardLayers, next[j], valueTensor);
            }
        }
        else if(resultLayout == collectionResult)
        {
            // split
            SharedPtr<LayerData> valueCollection = forwardResult->get(forward::resultForBackward);

            for(size_t j = 0; j < next.size(); j++)
            {
                setNextLayerInput(forwardLayers, next[j], staticPointerCast<Tensor, SerializationIface>( (*valueCollection)[j] ));
            }
        }
    }

    void setBackwardLayerInputsAndResults(const services::SharedPtr<layers::forward::LayerIface> &forwardLayer,
                                         services::SharedPtr<layers::backward::LayerIface> &backwardLayer)
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        layers::forward::Input *forwardInput = forwardLayer->getLayerInput();
        services::SharedPtr<layers::forward::Result> forwardResult = forwardLayer->getLayerResult();
        LayerResultLayout resultLayout = forwardResult->getLayout();
        LayerInputLayout inputLayout = forwardInput->getLayout();

        layers::backward::Input *backwardInput = backwardLayer->getLayerInput();
        services::SharedPtr<layers::backward::Result> backwardResult = backwardLayer->getLayerResult();

        backwardInput->set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));
        if(resultLayout == tensorResult)
        {
            backwardInput->set(backward::inputGradient, forwardResult->get(forward::value));
            if(inputLayout == tensorInput)
            {
                backwardResult->set(backward::gradient, forwardInput->get(forward::data));
            }
            else if(inputLayout == collectionInput)
            {
                // merge
                backwardResult->set(backward::resultLayerData, forwardInput->get(forward::inputLayerData));
            }
        }
        else if(resultLayout == collectionResult)
        {
            // split
            backwardResult->set(backward::gradient, forwardInput->get(forward::data));
        }
    }

    void setNextLayerInput(services::SharedPtr<ForwardLayers> &forwardLayers, size_t layerId,
        const services::SharedPtr<data_management::Tensor> &tensor)
    {
        using namespace services;
        using namespace layers;

        forward::Input *nextForwardInput = forwardLayers->get(layerId)->getLayerInput();

        if(nextForwardInput->getLayout() == tensorInput)
        {
            nextForwardInput->set(forward::data, tensor);
        }
        else
        {
            SharedPtr<LayerData> layerData = nextForwardInput->get(forward::inputLayerData);
            if( !(layerData) )
            {
                layerData = SharedPtr<LayerData>(new LayerData());
            }

            size_t n = layerData->size();
            (*(layerData))[n] = tensor;
            nextForwardInput->set(forward::inputLayerData, layerData);
        }
    }

    template<typename modelFPType>
    data_management::HomogenNumericTable<modelFPType> *allocateParameterTable(size_t nColumns, size_t nRows) const
    {
        using namespace data_management;

        NumericTableFeature feature;
        services::SharedPtr<NumericTableDictionary > dictionary(new NumericTableDictionary(nColumns, true));
        dictionary->setAllFeatures(feature);
        HomogenNumericTable<modelFPType> *table = new HomogenNumericTable<modelFPType>(dictionary);
        table->setNumberOfRows(nRows);
        table->allocateDataMemory();

        return table;
    }

    template<typename modelFPType>
    services::SharedPtr<data_management::HomogenNumericTable<modelFPType> > tensorsToTable(
        const services::Collection<services::SharedPtr<data_management::Tensor> > &tensors) const
    {
        return tensorsToTable<modelFPType>(tensors, 0, tensors.size());
    }

    template<typename modelFPType>
    services::SharedPtr<data_management::HomogenNumericTable<modelFPType> > tensorsToTable(
        const services::Collection<services::SharedPtr<data_management::Tensor> > &tensors, size_t startTensor, size_t nTensors) const
    {
        using namespace data_management;
        size_t tableSize = 0;
        for (size_t i = startTensor; i < startTensor + nTensors; i++)
        {
            if (tensors[i])
            {
                tableSize += tensors[i]->getSize();
            }
        }
        if (tableSize == 0)
        {
            return services::SharedPtr<HomogenNumericTable<modelFPType> >();
        }

        services::SharedPtr<HomogenNumericTable<modelFPType> > table(allocateParameterTable<modelFPType>(tableSize, (size_t)1));
        modelFPType *tableArray = table->getArray();

        size_t tableOffset = 0;
        SubtensorDescriptor<modelFPType> subtensor;
        for (size_t i = startTensor; i < startTensor + nTensors; i++)
        {
            if (!tensors[i]) { continue; }
            size_t tensorSize = tensors[i]->getSize();
            if (tensorSize == 0) { continue; }
            const services::Collection<size_t> &dims = tensors[i]->getDimensions();
            size_t firstDimension = dims[0];
            tensors[i]->getSubtensor(0, 0, 0, firstDimension, readOnly, subtensor);
            modelFPType *tensorArray = subtensor.getPtr();

            services::daal_memcpy_s(tableArray + tableOffset, tensorSize * sizeof(modelFPType), tensorArray, tensorSize * sizeof(modelFPType));
            tableOffset += tensorSize;

            tensors[i]->releaseSubtensor(subtensor);
        }
        return table;
    }

    template<typename modelFPType>
    void tableToTensors(const services::SharedPtr<data_management::NumericTable> &table,
        services::Collection<services::SharedPtr<data_management::Tensor> > &tensors, size_t startTensor, size_t nTensors)
    {
        using namespace data_management;

        BlockDescriptor<modelFPType> block;
        table->getBlockOfRows(0, table->getNumberOfRows(), readOnly, block);
        modelFPType *tableArray = block.getBlockPtr();

        size_t tableOffset = 0;
        SubtensorDescriptor<modelFPType> subtensor;
        for (size_t i = startTensor; i < startTensor + nTensors; i++)
        {
            if (!tensors[i]) { continue; }
            if (tensors[i]->getSize() == 0) { continue; }
            const services::Collection<size_t> &dims = tensors[i]->getDimensions();
            size_t firstDimension = dims[0];
            tensors[i]->getSubtensor(0, 0, 0, firstDimension, writeOnly, subtensor);
            size_t tensorSize = subtensor.getSize();
            modelFPType *tensorArray = subtensor.getPtr();

            services::daal_memcpy_s(tensorArray, tensorSize * sizeof(modelFPType), tableArray + tableOffset, tensorSize * sizeof(modelFPType));
            tableOffset += tensorSize;

            tensors[i]->releaseSubtensor(subtensor);
        }
        table->releaseBlockOfRows(block);
    }

private:
    services::Collection<size_t> _sampleSize;
    services::SharedPtr<ForwardLayers> _forwardLayers;   /*!< List of forward  layers of the network */
    services::SharedPtr<BackwardLayers> _backwardLayers; /*!< List of backward layers of the network */
    services::SharedPtr<services::Collection<layers::NextLayers> > _nextLayers; /*!< List of edges connecting the layers in the network */
    mutable services::ErrorCollection _errors; /*!< Collection of the errors */

    bool _storeWeightsInTable;      /*!< Flag. True if weights and biases of all the layers are stored in one numeric table */
    size_t _weightsAndBiasesSize;   /*!< Full number of elements in weights and biases of all the layers in the network */
    services::Collection<services::Collection<size_t> > _weightsDimsCollection;  /*!< Collection of weights tensors dimensions of all layers */
    services::Collection<services::Collection<size_t> > _biasesDimsCollection;   /*!< Collection of biases  tensors dimensions of all layers */
    services::Collection<size_t> _weightsSize;       /*!< Collection of number of elements in weights tensors for each layer */
    services::Collection<size_t> _biasesSize;        /*!< Collection of number of elements in biases  tensors for each layer */
    services::Collection<size_t> _weightsOffsets;       /*!< Collection of the offsets of data blocks that contain weights tensors for each layer */
    services::Collection<size_t> _biasesOffsets;        /*!< Collection of the offsets of data blocks that contain biases  tensors for each layer */
    services::SharedPtr<data_management::NumericTable> _weightsAndBiasesTable;            /*!< Weights and biases of all the layers in the network */
    services::SharedPtr<data_management::NumericTable> _weightsAndBiasesDerivativesTable; /*!< Weight and biases derivatives of the layers */
    services::Collection<services::SharedPtr<data_management::Tensor> > _weightsAndBiasesTensors;
    services::Collection<services::SharedPtr<data_management::Tensor> > _weightsAndBiasesDerivativesTensors;
};
} // namespace interface1
using interface1::Parameter;
using interface1::Model;

} // namespace training
} // namespace neural_networks
} // namespace algorithms
} //namespace daal
#endif
