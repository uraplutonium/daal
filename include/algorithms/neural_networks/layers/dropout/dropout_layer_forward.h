/* file: dropout_layer_forward.h */
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
//  Implementation of the forward dropout layer.
//--
*/

#ifndef __NEURAL_NETWORK_DROPOUT_LAYER_FORWARD_H__
#define __NEURAL_NETWORK_DROPOUT_LAYER_FORWARD_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/dropout/dropout_layer_types.h"
#include "algorithms/neural_networks/layers/dropout/dropout_layer_forward_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace dropout
{
namespace forward
{
namespace interface1
{
/**
* \brief Provides methods to run implementations of the of the forward dropout layer
*        This class is associated with the daal::algorithms::neural_networks::layers::dropout::forward::Batch class
*        and supports the method of forward dropout layer computation in the batch processing mode
*
* \tparam algorithmFPType  Data type to use in intermediate computations of forward dropout layer, double or float
* \tparam method           Computation method of the layer, \ref daal::algorithms::neural_networks::layers::dropout::Method
* \tparam cpu              Version of the cpu-specific implementation of the layer, \ref daal::CpuType
*/
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs the container for the forward dropout with the specified environment
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    ~BatchContainer();

    void compute();
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__DROPOUT__FORWARD__BATCH"></a>
 * \brief Provides methods for the forward dropout layer in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the forward dropout layer, double or float
 * \tparam method           Forward dropout layer method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method                     Computation methods for the forward dropout layer
 *      - \ref forward::InputId           Identifiers of input objects for the forward dropout layer
 *      - \ref forward::ResultId          Identifiers of result objects for the forward dropout layer
 *      - \ref forward::ResultLayerDataId Identifiers of extra results computed by the forward dropout layer
 *      - \ref LayerDataId                Identifiers of collection in result objects for the forward dropout layer
 *
 * \par References
 *      - <a href="DAAL-REF-DROPOUTFORWARD-ALGORITHM">Forward dropout layer description and usage models</a>
 *      - \ref interface1::Parameter "Parameter" class
 *      - \ref backward::interface1::Batch "backward::Batch" class
 */
template<typename algorithmFPType = float, Method method = defaultDense>
class Batch : public layers::forward::LayerIface
{
public:
    Parameter *parameter; /*!< Forward dropout layer parameters */
    Input input;          /*!< Forward dropout layer input */

    /** Default constructor */
    Batch() : parameter(new Parameter())
    {
        initialize();
    };

    /**
     * Constructs a forward dropout layer by copying input objects
     * and parameters of another forward dropout layer in the batch processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the layer
     */
    Batch(const Batch<algorithmFPType, method> &other)
    {
        initialize();
        input.set(layers::forward::data, other.input.get(layers::forward::data));
        input.set(layers::forward::weights, other.input.get(layers::forward::weights));
        input.set(layers::forward::biases, other.input.get(layers::forward::biases));
        parameter = other.parameter;
    }

    /**
    * Returns the method of the layer
    * \return Method of the layer
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains the input objects of the forward dropout layer
     * \return Structure that contains the input objects of the forward dropout layer
     */
    virtual Input *getLayerInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the structure that contains the parameters of the forward dropout layer
     * \return Structure that contains the parameters of the forward dropout layer
     */
    virtual Parameter *getLayerParameter() { return parameter; };

    /**
     * Returns the structure that contains result of the forward dropout layer
     * \return Structure that contains result of the forward dropout layer
     */
    services::SharedPtr<layers::forward::Result> getLayerResult()
    {
        return getResult();
    }

    /**
     * Returns the structure that contains the result of the forward dropout layer
     * \return Structure that contains the result of the forward dropout layer
     */
    services::SharedPtr<Result> getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store the result of the forward dropout layer
     * \param[in] result Structure to store the result of the forward dropout layer
     */
    void setResult(services::SharedPtr<Result> result)
    {
        _result = result;
        _res = _result.get();
    }

    /**
     * Returns a pointer to a newly allocated forward dropout layer
     * with a copy of the input objects and parameters for this forward dropout layer
     * in the batch processing mode
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
    * Allocates memory to store the result of the forward dropout layer
    */
    virtual void allocateResult()
    {
        this->_result->template allocate<algorithmFPType>(&(this->input), parameter, (int) method);
        this->_res = this->_result.get();
    }

    /**
     * Allocates memory buffers needed for the computations
     */
    virtual void allocateLayerData() DAAL_C11_OVERRIDE
    {
        this->_result->template allocateLayerData<algorithmFPType>(&(this->input), parameter, (int) method);
        this->_res = this->_result.get();
    }

protected:
    virtual Batch<algorithmFPType, method> *cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    void setParameter() DAAL_C11_OVERRIDE { _par = parameter; };

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in = &input;
        _par = parameter;
        _result = services::SharedPtr<Result>(new Result());
    }

private:
    services::SharedPtr<Result> _result;
};
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;
} // namespace forward
} // namespace dropout
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
