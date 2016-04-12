/* file: concat_layer_forward.h */
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

#ifndef __NEURAL_NETWORK_CONCAT_LAYER_FORWARD_H__
#define __NEURAL_NETWORK_CONCAT_LAYER_FORWARD_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/concat/concat_layer_types.h"
#include "algorithms/neural_networks/layers/concat/concat_layer_forward_types.h"

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
namespace interface1
{
/**
* \brief Provides methods to run implementations of the of the forward concat layer
*        This class is associated with the daal::algorithms::neural_networks::layers::concat::forward::Batch class
*        and supports the method of forward concat layer computation in the batch processing mode
*
* \tparam algorithmFPType  Data type to use in intermediate computations of forward concat layer, double or float
* \tparam method           Computation method of the layer, \ref daal::algorithms::neural_networks::layers::concat::Method
* \tparam cpu              Version of the cpu-specific implementation of the layer, \ref daal::CpuType
*/
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public AnalysisContainerIface<batch>
{
public:
    /**
    * Constructs the container for the forward concat layer with the specified environment
    * \param[in] daalEnv   Environment object
    */
    BatchContainer(daal::services::Environment::env *daalEnv);
    ~BatchContainer();

    void compute();
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONCAT__FORWARD__BATCH"></a>
 * \brief Computes the results of the forward concat layer in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the forward concat layer, double or float
 * \tparam method           The forward concat layer computation method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method                     Computation methods for the forward concat layer
 *      - \ref forward::InputId           Identifiers of input objects for the forward concat layer
 *      - \ref forward::ResultId          Identifiers of result objects for the forward concat layer
 *      - \ref forward::ResultLayerDataId Identifiers of auxilary results computed by the forward concat layer
 *      - \ref LayerDataId                Identifiers of collection in result objects for the forward concat layer
 *
 * \par References
 *      - <a href="DAAL-REF-CONCATFORWARD-ALGORITHM">Forward concat layer description and usage models</a>
 *      - \ref backward::interface1::Batch "backward::Batch" class
 */
template<typename algorithmFPType = float, Method method = defaultDense>
class Batch : public layers::forward::LayerIface
{
public:
    Parameter *parameter; /*!< %Parameters of the algorithm */
    Input input;         /*!< %Input objects of the layer */

    /**
    * Constructs forward concat layer
    * \param[in] concatDimension Index of dimension along which concatenation is implemented
    */
    Batch(size_t concatDimension = 0) : parameter(new Parameter(concatDimension))
    {
        initialize();
    };

    /**
     * Constructs the forward concat layer by copying input objects of
     * another forward concat layer
     * \param[in] other An algorithm to be used as the source to initialize the input objects of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) : parameter(other.parameter)
    {
        initialize();
        input.set(layers::forward::inputLayerData, other.input.get(layers::forward::inputLayerData));
        parameter = other.parameter;
    }

    /**
    * Returns method of the forward concat layer
    * \return Method of the forward concat layer
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains input objects of the forward concat layer
     * \return Structure that contains input objects of the forward concat layer
     */
    virtual Input *getLayerInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the structure that contains parameters of the forward concat layer
     * \return Structure that contains parameters of the forward concat layer
     */
    virtual Parameter *getLayerParameter() { return parameter; };

    /**
     * Returns the structure that contains the result of the forward concat layer
     * \return Structure that contains the result of the forward concat layer
     */
    services::SharedPtr<layers::forward::Result> getLayerResult()
    {
        return getResult();
    }

    /**
     * Returns the structure that contains the result of the forward concat layer
     * \return Structure that contains the result of forward concat layer
     */
    services::SharedPtr<Result> getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store results of the forward concat layer
     * \param[in] result  Structure to store  results of the forward concat layer
     */
    void setResult(services::SharedPtr<Result> result)
    {
        _result = result;
        _res = _result.get();
    }

    /**
     * Returns a pointer to the newly allocated forward concat layer
     * with a copy of input objects of this forward concat layer
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
    * Allocates memory to store the result of the forward concat layer
    */
    virtual void allocateResult() DAAL_C11_OVERRIDE
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
} // namespace concat
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
