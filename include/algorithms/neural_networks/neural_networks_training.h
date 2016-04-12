/* file: neural_networks_training.h */
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
//  Implementation of neural network training.
//--
*/

#ifndef __NEURAL_NETWORK_TRAINING_H__
#define __NEURAL_NETWORK_TRAINING_H__

#include "algorithms/algorithm.h"

#include "services/daal_defines.h"
#include "algorithms/neural_networks/neural_networks_types.h"
#include "algorithms/neural_networks/neural_networks_training_types.h"
#include "algorithms/neural_networks/neural_networks_training_model.h"
#include "algorithms/neural_networks/layers/layer.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for training and prediction using neural networks
 */
namespace neural_networks
{
namespace training
{
namespace interface1
{
/**
 * \brief Class containing methods to train neural network model using algorithmFPType precision arithmetic
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public TrainingContainerIface<batch>
{
public:
    /** \brief Constructor */
    BatchContainer(daal::services::Environment::env *daalEnv);
    ~BatchContainer();

    void compute();
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__TRAINING__BATCH"></a>
* \brief Provides methods for neural network model-based training in the batch processing mode
*
* \tparam algorithmFPType  Data type to use in intermediate computations for neural network model-based training, double or float
* \tparam method           Neural network training method, training::Method
*
* \par Enumerations
*      - training::Method  Computation methods
*
* \par References
*      - <a href="DAAL-REF-NEURALNETWORK-ALGORITHM">Neural network algorithm description and usage models</a>
*      - \ref interface1::Parameter "Parameter" class
*      - \ref neural_networks::training::interface1::Model "neural_networks::training::Model" class
*      - \ref prediction::interface1::Batch "prediction::Batch" class
*/
template<typename algorithmFPType = float, Method method = defaultDense>
class Batch : public daal::algorithms::Training<batch>
{
public:
    /** Default constructor */
    Batch()
    {
        initialize();
    };

    /**
     * Constructs neural network by copying input objects and parameters of another neural network
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other)
    {
        initialize();
        input.set(data,   other.input.get(data));
        input.set(groundTruth, other.input.get(groundTruth));
        parameter = other.parameter;
    }

    virtual ~Batch() {}

    /**
     * Initializes the neural network configuration of the layers
     * \param[in] dataSize            Collection of sizes of each dimension of input data tensor
     * \param[in] layerDescriptors    Configuration of the layers
     */
    void initialize(const services::Collection<size_t> &dataSize, const services::Collection<layers::LayerDescriptor> &layerDescriptors)
    {
        _result->get(neural_networks::training::model)->initialize<algorithmFPType>(dataSize, layerDescriptors, &parameter);
    }

    /**
    * Returns the structure that contains the results of the neural network algorithm
    * \return Structure that contains the results of the neural network algorithm
    */
    services::SharedPtr<Result> getResult()
    {
        return _result;
    }

    /**
     * Register user-allocated memory to store the results of the neural network algorithm
     * \return Structure to store the results of the neural network algorithm
     */
    void setResult(services::SharedPtr<Result> res)
    {
        _result = res;
        _res = _result.get();
    }

    /**
     * Returns a pointer to the newly allocated neural network
     * with a copy of input objects and parameters of this neural network
     * \return Pointer to the newly allocated layer
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    Input input; /*!< %Input data structure */
    Parameter<algorithmFPType> parameter; /*!< Training parameters */

protected:
    void initialize()
    {
        Training<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in = &input;
        _par = &parameter;
        _result = services::SharedPtr<Result>(new Result());
    }

    virtual Batch<algorithmFPType, method> *cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    virtual void allocateResult()
    {
        _result->allocate<algorithmFPType>(&input, &parameter, (int) method);
        _res = _result.get();
    }

private:
    services::SharedPtr<Result> _result;
};

} // namespace interface1
using interface1::Batch;
using interface1::BatchContainer;

} // namespace training
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
