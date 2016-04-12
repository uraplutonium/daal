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
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/layer_types.h"
#include "algorithms/neural_networks/neural_networks_prediction_model.h"
#include "algorithms/neural_networks/neural_networks_training_input.h"

#include "algorithms/optimization_solver/optimization_solver_batch.h"
#include "algorithms/optimization_solver/sgd/sgd_batch.h"
#include "algorithms/optimization_solver/objective_function/objective_function_batch.h"
#include "algorithms/optimization_solver/objective_function/mse_batch.h"

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
template <typename modelFPType>
class Parameter : public daal::algorithms::Parameter
{
public:
    /**
     * Constructs the parameters of neural network algorithm
     * \param[in] batchSize_                  Size of the batch to be processed by the neural network
     * \param[in] nIterations_                Maximal number of iterations of the algorithm
     * \param[in] optimizationSolver_         Optimization solver used in the neural network
     * \param[in] objectiveFunction_          Objective function used in the neural network
     */
    Parameter(size_t batchSize_ = 1,
              size_t nIterations_ = 1000,
              services::SharedPtr<optimization_solver::sgd::Batch<modelFPType> > optimizationSolver_ =
                  services::SharedPtr<optimization_solver::sgd::Batch<modelFPType> >(
                      new optimization_solver::sgd::Batch<modelFPType>
                      (services::SharedPtr<optimization_solver::sum_of_functions::Batch>(new optimization_solver::mse::Batch<modelFPType>(1)))),
              services::SharedPtr<optimization_solver::mse::Batch<modelFPType> > objectiveFunction_ =
                  services::SharedPtr<optimization_solver::mse::Batch<modelFPType> >(new optimization_solver::mse::Batch<modelFPType>(1))) :
        batchSize(batchSize_), nIterations(nIterations_), optimizationSolver(optimizationSolver_), objectiveFunction(objectiveFunction_) {};

    size_t batchSize; /*!< Size of the batch to be processed by the neural network. */
    size_t nIterations; /*!< Maximal number of iterations of the algorithm. */

    services::SharedPtr<optimization_solver::sgd::Batch<modelFPType> > optimizationSolver; /*!< Optimization solver used in the neural network*/
    services::SharedPtr<optimization_solver::mse::Batch<modelFPType> > objectiveFunction; /*!< Objective function used in the neural network. */
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
        _errors(),
        _parameters(new services::Collection<services::SharedPtr<layers::Parameter> >()) {};

    /** \brief Copy constructor */
    Model(const Model &model) :
        _forwardLayers(model.getForwardLayers()),
        _backwardLayers(model.getBackwardLayers()),
        _nextLayers(model.getNextLayers()),
        _errors(model.getErrors()),
        _parameters(model.getParameters()) {};

    /** \brief Destructor */
    virtual ~Model() {};

    /**
     * Insert a layer to a certain position in neural network
     * \param[in] layerDescriptor  %LayerDescriptor of the inserted layer
     */
    void insertLayer(const layers::LayerDescriptor &layerDescriptor)
    {
        layers::Parameter *par = const_cast<layers::Parameter *>(layerDescriptor.layer->cloneLayerParameter());
        _parameters->insert(layerDescriptor.index, services::SharedPtr<layers::Parameter>(par));
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
                    const Parameter<modelFPType> *parameter)
    {
        for(size_t i = 0; i < layerDescriptors.size(); i++)
        {
            insertLayer(layerDescriptors[i]);
        }
        allocate<modelFPType>(dataSize, parameter);
    }

    /**
     * Sets list of forward layers
     * \param[in] forwardLayers          List of forward layers
     */
    void setForwardLayers(const services::SharedPtr<ForwardLayers> &forwardLayers)
    {
        _forwardLayers = forwardLayers;
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
     * Sets list of backward layers
     * \param[in] backwardLayers          List of backward layers
     */
    void setBackwardLayers(const services::SharedPtr<BackwardLayers> &backwardLayers)
    {
        _backwardLayers = backwardLayers;
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
     * Sets list of connections between layers
     * \param[in] nextLayers          List of next layers for each layer with corresponding index
     */
    void setNextLayers(const services::SharedPtr<services::Collection<layers::NextLayers> > &nextLayers)
    {
        _nextLayers = nextLayers;
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
     * Sets table containing all layer parameters
     * \param[in] parameters          Table containing all layer parameters
     */
    void setParameters(const services::SharedPtr<services::Collection<services::SharedPtr<layers::Parameter> > > &parameters)
    {
        _parameters = parameters;
    }

    /**
     * Returns table containing all layer parameters
     * \return          Table containing all layer parameters
     */
    const services::SharedPtr<services::Collection<services::SharedPtr<layers::Parameter> > > getParameters() const
    {
        return _parameters;
    }

    /**
     * Returns list of forward layers and their parameters organised in the prediction::Model
     * \return          List of forward layers and their parameters organised in the prediction::Model
     */
    const services::SharedPtr<prediction::Model> getPredictionModel() const
    {
        prediction::Model *predictionModel = new prediction::Model(_forwardLayers, _nextLayers);
        predictionModel->setWeightsAndBiases(_weightsAndBiasesTable);
        predictionModel->setParameters(_parameters);
        return services::SharedPtr<prediction::Model>(predictionModel);
    }

    /**
     * Returns the weights and biases container
     * \return   Weights and biases container
     */
    const services::SharedPtr<data_management::NumericTable> getWeightsAndBiases() const
    {
        return _weightsAndBiasesTable;
    }

    /**
     * Returns the weights and biases derivatives container
     * \return   Weights and biases derivatives container
     */
    const services::SharedPtr<data_management::NumericTable> &getWeightsAndBiasesDerivatives() const { return _weightsAndBiasesDerivativesTable; }

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

        Parameter<modelFPType> *par = const_cast<Parameter<modelFPType> *>(static_cast<const Parameter<modelFPType> *>(parameter));
        size_t nLayers = _forwardLayers->size();

        // allocation for batchSize = 1
        Collection<size_t> sampleSize;
        sampleSize.push_back(1);
        for(size_t i = 1; i < dataSize.size(); i++) { sampleSize.push_back(dataSize[i]); }

        size_t weightsSize = 0, biasesSize = 0;
        Collection<Collection<size_t> > weightsCollection, biasesCollection;

        computeWeightsAndBiasesSize<modelFPType>(nLayers, sampleSize, weightsSize, biasesSize, weightsCollection, biasesCollection);

        size_t parameterSize = weightsSize + biasesSize;

        _weightsAndBiasesTable = SharedPtr<NumericTable>(allocateParameterTable<modelFPType>(parameterSize, (const size_t)1));
        _weightsAndBiasesDerivativesTable = SharedPtr<NumericTable>(allocateParameterTable<modelFPType>(parameterSize, (const size_t)1));

        modelFPType *weightsAndBiasesArray =
            (staticPointerCast<HomogenNumericTable<modelFPType>, NumericTable>(_weightsAndBiasesTable))->getArray();
        modelFPType *weightsAndBiasesDerivativesArray =
            (staticPointerCast<HomogenNumericTable<modelFPType>, NumericTable>(_weightsAndBiasesDerivativesTable))->getArray();

        setForwardBuffers<modelFPType>(nLayers, weightsSize, weightsCollection, biasesCollection,
                                       weightsAndBiasesArray, weightsAndBiasesDerivativesArray);

        setObjectiveFunctionArguments<modelFPType>(nLayers, par);

        setBackwardBuffers<modelFPType>(nLayers);
    }

    /**
     * Returns the serialization tag of the neural network model
     * \return         Serialization tag of the neural network model
     */

    int getSerializationTag() { return SERIALIZATION_NEURAL_NETWORKS_TRAINING_MODEL_ID; }
    /**
    *  Serializes an object
    *  \param[in]  arch  Storage for a serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch)
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes an object
    *  \param[in]  arch  Storage for a deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch)
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

    template<typename modelFPType>
    void setObjectiveFunctionArguments(const size_t nLayers, Parameter<modelFPType> *parameter)
    {
        using namespace services;
        using namespace data_management;
        using namespace optimization_solver;

        SharedPtr<Tensor> forwardResult = _forwardLayers->get(nLayers - 1)->getLayerResult()->get(layers::forward::value);
        SharedPtr<HomogenNumericTable<modelFPType> > forwardResultTable = tensorToTable<modelFPType>(forwardResult);

        SharedPtr<Tensor> groundTruthTensor(new HomogenTensor<modelFPType>(forwardResult->getDimensions(), Tensor::notAllocate));
        SharedPtr<HomogenNumericTable<modelFPType> > groundTruthTable = tensorToTable<modelFPType>(groundTruthTensor);
        SharedPtr<HomogenNumericTable<modelFPType> > dataTable(new HomogenNumericTable<modelFPType>(NULL, 0,
                                                                                                    forwardResultTable->getNumberOfColumns()));

        parameter->objectiveFunction->input.set(mse::dependentVariables, groundTruthTable);
        parameter->objectiveFunction->input.set(mse::argument, forwardResultTable);
        parameter->objectiveFunction->input.set(mse::data, dataTable);
        parameter->objectiveFunction->parameter.resultsToCompute = objective_function::gradient;
        parameter->objectiveFunction->allocate();

        SharedPtr<HomogenNumericTable<modelFPType> > gradientTable =
            staticPointerCast<HomogenNumericTable<modelFPType>, NumericTable> (
                parameter->objectiveFunction->getResult()->get(objective_function::resultCollection, objective_function::gradientIdx));

        SharedPtr<HomogenTensor<modelFPType> > gradientTensor = tableToTensor<modelFPType>(gradientTable, forwardResult->getDimensions());

        _backwardLayers->get(nLayers - 1)->getLayerInput()->set(layers::backward::inputGradient, gradientTensor);
    }

    template<typename modelFPType>
    void computeWeightsAndBiasesSize(const size_t nLayers, const services::Collection<size_t> &sampleSize,
                                     size_t &weightsSize, size_t &biasesSize,
                                     services::Collection<services::Collection<size_t> > &weightsCollection,
                                     services::Collection<services::Collection<size_t> > &biasesCollection)
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        _forwardLayers->get(0)->getLayerInput()->set(forward::data,
                                                     SharedPtr<Tensor>(new HomogenTensor<modelFPType>(sampleSize, Tensor::notAllocate)));
        for(size_t i = 0; i < nLayers; i++)
        {
            SharedPtr<forward::LayerIface> forwardLayer = _forwardLayers->get(i);
            forward::Input *input = forwardLayer->getLayerInput();
            layers::Parameter *param = forwardLayer->getLayerParameter();
            SharedPtr<forward::Result> result = forwardLayer->getLayerResult();

            weightsCollection.push_back(input->getWeightsSizes(param));
            biasesCollection.push_back(input->getBiasesSizes(param));

            weightsSize += getCollectionSize(weightsCollection[i]);
            biasesSize  += getCollectionSize(biasesCollection[i]);

            NextLayers next = _nextLayers->get(i);
            LayerResultLayout resultLayout = result->getLayout();

            setForwardLayerResult<modelFPType>(forwardLayer, input, param, result, resultLayout);
            setNextForwardLayerInput<modelFPType>(forwardLayer, input, param, result, next, resultLayout);
        }
    }

    template<typename modelFPType>
    void setForwardLayerResult(services::SharedPtr<layers::forward::LayerIface> forwardLayer, layers::forward::Input *input,
                               layers::Parameter *param, services::SharedPtr<layers::forward::Result> result, layers::LayerResultLayout resultLayout)
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        LayerInputLayout inputLayout = input->getLayout();
        const int method = forwardLayer->getMethod();

        if(resultLayout == tensorResult)
        {
            Collection<size_t> valueSize;

            if(inputLayout == tensorInput)
            {
                const Collection<size_t> sampleSize = input->get(forward::data)->getDimensions();
                valueSize = result->getValueSize(sampleSize, param, method);

            }
            else if(inputLayout == collectionInput)
            {
                // merge
                Collection< Collection<size_t> > dimsCollection;
                SharedPtr<LayerData> layerData = input->get(forward::inputLayerData);
                size_t nInputs = layerData->size();

                for(size_t k = 0; k < nInputs; k++)
                {
                    dimsCollection.push_back( (staticPointerCast<Tensor, SerializationIface>( (*layerData)[k] ))->getDimensions() );
                }
                valueSize = result->getValueSize(dimsCollection, param, method);
                result->set(forward::resultForBackward, SharedPtr<LayerData>(new LayerData()));
            }
            result->set(forward::value, SharedPtr<Tensor>(new HomogenTensor<modelFPType>(valueSize, Tensor::notAllocate)));
        }
        else if(resultLayout == collectionResult)
        {
            // split
            const Collection<size_t> sampleSize = input->get(forward::data)->getDimensions();
            const Collection<Collection<size_t> > valueCollectionSize = result->getValueCollectionSize(sampleSize, param, method);

            size_t nOutputs = valueCollectionSize.size();

            SharedPtr<LayerData> resultCollection = SharedPtr<LayerData>(new LayerData());

            for(size_t i = 0; i < nOutputs; i++)
            {
                (*resultCollection)[i] = SharedPtr<Tensor>(new HomogenTensor<modelFPType>( sampleSize, Tensor::notAllocate));
            }
            result->set(forward::resultForBackward, resultCollection);

        }
    }

    template<typename modelFPType>
    void setNextForwardLayerInput(services::SharedPtr<layers::forward::LayerIface> forwardLayer, layers::forward::Input *input,
                                  layers::Parameter *param, services::SharedPtr<layers::forward::Result> result, layers::NextLayers next,
                                  layers::LayerResultLayout resultLayout)
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        if(resultLayout == tensorResult)
        {
            SharedPtr<Tensor> valueTensor = result->get(forward::value);

            for(size_t j = 0; j < next.size(); j++)
            {
                setNextLayerInput<modelFPType>(next[j], valueTensor);
            }
        }
        else if(resultLayout == collectionResult)
        {
            // split
            services::SharedPtr<LayerData> valueCollection = result->get(forward::resultForBackward);

            for(size_t j = 0; j < next.size(); j++)
            {
                setNextLayerInput<modelFPType>(next[j], staticPointerCast<Tensor, SerializationIface>( (*valueCollection)[j] ));
            }
        }
    }

    template<typename modelFPType>
    void setNextLayerInput(size_t nLayer, services::SharedPtr<data_management::Tensor> tensor)
    {
        using namespace services;
        using namespace layers;

        forward::Input *nextForwardInput = _forwardLayers->get(nLayer)->getLayerInput();

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
    void setForwardBuffers(const size_t nLayers, const size_t weightsSize,
                           const services::Collection<services::Collection<size_t> > weightsCollection,
                           const services::Collection<services::Collection<size_t> > biasesCollection,
                           modelFPType *weightsAndBiasesArray, modelFPType *weightsAndBiasesDerivativesArray)
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        modelFPType *weightsArray = weightsAndBiasesArray;
        modelFPType *biasesArray = weightsAndBiasesArray + weightsSize;

        modelFPType *weightDerivativesArray = weightsAndBiasesDerivativesArray;
        modelFPType *biasDerivativesArray = weightsAndBiasesDerivativesArray + weightsSize;

        size_t weightsUsed = 0, biasesUsed = 0;
        for(size_t i = 0; i < nLayers; i++)
        {
            SharedPtr<forward::LayerIface> forwardLayer = _forwardLayers->get(i);
            forward::Input *input = forwardLayer->getLayerInput();

            SharedPtr<backward::LayerIface> backwardLayer = _backwardLayers->get(i);
            SharedPtr<backward::Result> backwardResult = backwardLayer->getLayerResult();

            SharedPtr<HomogenTensor<modelFPType> > weights(new HomogenTensor<modelFPType>(weightsCollection[i], weightsArray + weightsUsed));
            SharedPtr<HomogenTensor<modelFPType> > biases(new HomogenTensor<modelFPType>(biasesCollection[i], biasesArray + biasesUsed));

            SharedPtr<HomogenTensor<modelFPType> > weightDerivatives(new HomogenTensor<modelFPType>(weightsCollection[i],
                                                                                                    weightDerivativesArray + weightsUsed));
            SharedPtr<HomogenTensor<modelFPType> > biasDerivatives(new HomogenTensor<modelFPType>(biasesCollection[i],
                                                                                                  biasDerivativesArray + biasesUsed));

            weightsUsed += getCollectionSize(weightsCollection[i]);
            biasesUsed  += getCollectionSize(biasesCollection[i]);

            input->set(forward::weights, weights);
            input->set(forward::biases, biases);

            backwardResult->set(backward::weightDerivatives, weightDerivatives);
            backwardResult->set(backward::biasDerivatives, biasDerivatives);

            forwardLayer->allocateInput();
            forwardLayer->initializeInput();
            forwardLayer->allocateResult();

            NextLayers next = _nextLayers->get(i);

            layers::Parameter *param = forwardLayer->getLayerParameter();
            SharedPtr<forward::Result> result = forwardLayer->getLayerResult();

            LayerResultLayout resultLayout = result->getLayout();

            if(i < nLayers - 1)
            {
                forward::Input *nextForwardInput = _forwardLayers->get(i + 1)->getLayerInput();

                if(nextForwardInput->getLayout() == collectionInput)
                {
                    nextForwardInput->set(forward::inputLayerData, SharedPtr<LayerData>());
                }
            }
            setNextForwardLayerInput<modelFPType>(forwardLayer, input, param, result, next, resultLayout);
        }
    }

    template<typename modelFPType>
    void setBackwardBuffers(const size_t nLayers)
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;


        for(int i = (int)nLayers - 1; i >= 0; i--)
        {
            SharedPtr<backward::LayerIface> backwardLayer = _backwardLayers->get(i);
            backward::Input *input = backwardLayer->getLayerInput();
            SharedPtr<backward::Result> result = backwardLayer->getLayerResult();

            NextLayers next = _nextLayers->get(i);

            LayerInputLayout inputLayout = input->getLayout();
            if(inputLayout == tensorInput)
            {
                input->set(backward::inputFromForward, _forwardLayers->get(i)->getLayerResult()->get(forward::resultForBackward));
            }

            setBackwardLayerInput<modelFPType>(backwardLayer, input, result, next);

            backwardLayer->allocateInput();
            backwardLayer->allocateResult();
        }
    }

    template<typename modelFPType>
    void setBackwardLayerInput(services::SharedPtr<layers::backward::LayerIface> backwardLayer, layers::backward::Input *input,
                               services::SharedPtr<layers::backward::Result> result, layers::NextLayers next)
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        for(size_t j = 0; j < next.size(); j++)
        {
            LayerInputLayout inputLayout = input->getLayout();
            LayerResultLayout prevResultLayout = _backwardLayers->get(next[j])->getLayerResult()->getLayout();

            if(prevResultLayout == tensorResult)
            {
                SharedPtr<Tensor> gradientTensor = _backwardLayers->get(next[j])->getLayerResult()->get(backward::gradient);

                if(inputLayout == tensorInput)
                {
                    input->set(backward::inputGradient, gradientTensor);
                }
                else if(inputLayout == collectionInput)
                {
                    // split
                    SharedPtr<LayerData> layerData = input->get(backward::inputFromForward);
                    if( !(layerData) )
                    {
                        layerData = SharedPtr<LayerData>(new LayerData());
                    }

                    size_t n = layerData->size();
                    (*(layerData))[n] = gradientTensor;
                    input->set(backward::inputFromForward, layerData);
                }
            }
            else if(prevResultLayout == collectionResult)
            {
                // merge
                if(inputLayout == tensorInput)
                {
                    SharedPtr<Tensor> gradientTensor = staticPointerCast<Tensor, SerializationIface>(
                                                           (*(_backwardLayers->get(next[j])->getLayerResult()->get(
                                                                  backward::resultLayerData)))[0]);
                    input->set(backward::inputGradient, gradientTensor);
                }
                else if(inputLayout == collectionInput)
                {
                    // prev = merge, current = split
                    SharedPtr<LayerData> layerData = input->get(backward::inputFromForward);

                    if( !(layerData) )
                    {
                        layerData = SharedPtr<LayerData>(new LayerData());
                    }
                    size_t n = layerData->size();

                    SharedPtr<Tensor> gradientTensor = staticPointerCast<Tensor, SerializationIface>(
                                                           (*(_backwardLayers->get(next[j])->getLayerResult()->get(
                                                                  backward::resultLayerData)))[j] );

                    (*(layerData))[n] = gradientTensor;
                    input->set(backward::inputFromForward, layerData);
                }
            }
        }
    }

    template<typename modelFPType>
    data_management::HomogenNumericTable<modelFPType> *allocateParameterTable(const size_t nColumns, const size_t nRows)
    {
        using namespace data_management;

        NumericTableFeature feature;
        services::SharedPtr<Dictionary<NumericTableFeature> > dictionary(new Dictionary<NumericTableFeature>(nColumns, true));
        dictionary->setAllFeatures(feature);
        HomogenNumericTable<modelFPType> *table = new HomogenNumericTable<modelFPType>(dictionary);
        table->setNumberOfRows(nRows);
        table->allocateDataMemory();

        return table;
    }

    template<typename modelFPType>
    services::SharedPtr<data_management::HomogenNumericTable<modelFPType> > tensorToTable(
        services::SharedPtr<data_management::Tensor> tensor) const
    {
        using namespace data_management;

        SubtensorDescriptor<modelFPType> subtensor;
        const services::Collection<size_t> dims = tensor->getDimensions();
        size_t firstDimension = dims[0];
        tensor->getSubtensor(0, 0, 0, firstDimension, readOnly, subtensor);
        services::SharedPtr<HomogenNumericTable<modelFPType> > table(
            new HomogenNumericTable<modelFPType>(subtensor.getPtr(), subtensor.getSize(), 1));
        return table;
    }

    template<typename modelFPType>
    services::SharedPtr<data_management::HomogenTensor<modelFPType> > tableToTensor(
        const services::SharedPtr<data_management::HomogenNumericTable<modelFPType> > &table, const services::Collection<size_t> &dimensions)
    {
        using namespace data_management;
        services::SharedPtr<HomogenTensor<modelFPType> > tensor(new HomogenTensor<modelFPType>(dimensions, Tensor::doAllocate));
        modelFPType *tensorData = tensor->getArray();
        modelFPType *tableData = table->getArray();

        size_t size = 1;
        for(size_t i = 0; i < dimensions.size(); i++)
        {
            size *= dimensions[i];
        }

        for(size_t i = 0; i < size; i++)
        {
            tensorData[i] = tableData[i];
        }

        return tensor;
    }

private:
    services::SharedPtr<ForwardLayers> _forwardLayers; /*!< List of forward layers of the network */
    services::SharedPtr<BackwardLayers> _backwardLayers; /*!< List of backward layers of the network */
    services::SharedPtr<services::Collection<layers::NextLayers> > _nextLayers; /*!< List of edges connecting the layers in the network */
    services::ErrorCollection _errors; /*!< Collection of the errors */

    services::SharedPtr<services::Collection<services::SharedPtr<layers::Parameter> > > _parameters; /*!< List of parameters of the layers */

    services::SharedPtr<data_management::NumericTable> _weightsAndBiasesTable; /*!< Weights and biases of all the layers in the network */
    services::SharedPtr<data_management::NumericTable> _weightsAndBiasesDerivativesTable; /*!< Weight and biases derivatives of the layers */
};
} // namespace interface1
using interface1::Parameter;
using interface1::Model;

} // namespace training
} // namespace neural_networks
} // namespace algorithms
} //namespace daal
#endif
