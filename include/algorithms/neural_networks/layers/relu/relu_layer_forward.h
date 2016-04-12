/* file: relu_layer_forward.h */
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
//  Implementation of neural network layer.
//--
*/

#ifndef __NEURAL_NETWORK_RELU_LAYER_FORWARD_H__
#define __NEURAL_NETWORK_RELU_LAYER_FORWARD_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/relu/relu_layer_types.h"
#include "algorithms/neural_networks/layers/relu/relu_layer_forward_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * \brief Contains classes for the relu layer
 */
namespace relu
{
/**
 * \brief Contains classes for the forward relu layer
 */
namespace forward
{
namespace interface1
{
/**
 * \brief Class containing methods for the forward relu layer using algorithmFPType precision arithmetic
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public AnalysisContainerIface<batch>
{
public:
    /**
    * Constructs the container for the forward relu layer with the specified environment
    * \param[in] daalEnv   Environment object
    */
    BatchContainer(daal::services::Environment::env *daalEnv);
    ~BatchContainer();

    void compute();
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__RELU__FORWARD__BATCH"></a>
 * \brief Computes the results of the forward relu layer in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the forward relu layer, double or float
 * \tparam method           The forward relu layer computation method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method                     Computation methods for the forward relu layer
 *      - \ref forward::InputId           Identifiers of input objects for the forward relu layer
 *      - \ref forward::ResultId          Identifiers of result objects for the forward relu layer
 *      - \ref forward::ResultLayerDataId Identifiers of extra results computed by the forward relu layer
 *      - \ref LayerDataId                Identifiers of collection in result objects for the forward relu layer
 *
 * \par References
 *      - <a href="DAAL-REF-RELUFORWARD-ALGORITHM">Forward relu layer description and usage models</a>
 *      - \ref backward::interface1::Batch "backward::Batch" class
 */
template<typename algorithmFPType = float, Method method = defaultDense>
class Batch : public layers::forward::LayerIface
{
public:

    Input input;         /*!< %Input objects of the layer */

    /** \brief Default constructor */
    Batch()
    {
        initialize();
    };

    /**
     * Constructs the forward relu layer by copying input objects of
     * another forward relu layer
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other)
    {
        initialize();
        input.set(layers::forward::data, other.input.get(layers::forward::data));
        input.set(layers::forward::weights, other.input.get(layers::forward::weights));
        input.set(layers::forward::biases, other.input.get(layers::forward::biases));
    }

    /**
    * Returns method of the forward relu layer
    * \return Method of the forward relu layer
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains input objects of the forward relu layer
     * \return Structure that contains input objects of the forward relu layer
     */
    virtual Input *getLayerInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the structure that contains parameters of the forward relu layer
     * \return Structure that contains parameters of the forward relu layer
     */
    virtual Parameter *getLayerParameter() { return NULL; };

    /**
     * Returns the structure that contains results of the forward relu layer
     * \return Structure that contains results of the forward relu layer
     */
    services::SharedPtr<layers::forward::Result> getLayerResult()
    {
        return _result;
    }

    /**
     * Returns the structure that contains the result of the forward relu layer
     * \return Structure that contains the result of forward relu layer
     */
    services::SharedPtr<Result> getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store results of the forward relu layer
     * \param[in] result  Structure to store  results of the forward relu layer
     */
    void setResult(services::SharedPtr<Result> result)
    {
        _result = result;
        _res = _result.get();
    }

    /**
     * Returns a pointer to the newly allocated forward relu layer
     * with a copy of input objects of this forward relu layer
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
    * Allocates memory to store the result of the forward relu layer
    */
    virtual void allocateResult()
    {
        this->_result->template allocate<algorithmFPType>(&(this->input), NULL, (int) method);
        this->_res = this->_result.get();
    }

    /**
     * Allocates memory buffers needed for the computations
     */
    virtual void allocateLayerData() DAAL_C11_OVERRIDE
    {
        this->_result->template allocateLayerData<algorithmFPType>(&(this->input), NULL, (int) method);
        this->_res = this->_result.get();
    }

protected:
    virtual Batch<algorithmFPType, method> *cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in = &input;
        _result = services::SharedPtr<Result>(new Result());
    }

private:
    services::SharedPtr<Result> _result;
};
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;
} // namespace forward
} // namespace relu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
