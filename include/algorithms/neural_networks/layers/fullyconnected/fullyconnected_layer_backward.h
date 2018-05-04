/* file: fullyconnected_layer_backward.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation
* All Rights Reserved.
*
* If this  software was obtained  under the  Intel Simplified  Software License,
* the following terms apply:
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*
*
* If this  software  was obtained  under the  Apache License,  Version  2.0 (the
* "License"), the following terms apply:
*
* You may  not use this  file except  in compliance  with  the License.  You may
* obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
*
*
* Unless  required  by   applicable  law  or  agreed  to  in  writing,  software
* distributed under the License  is distributed  on an  "AS IS"  BASIS,  WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*
* See the   License  for the   specific  language   governing   permissions  and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of the interface for the backward fully-connected layer
//  in the batch processing mode
//--
*/

#ifndef __FULLYCONNECTED_LAYER_BACKWARD_H__
#define __FULLYCONNECTED_LAYER_BACKWARD_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/fullyconnected/fullyconnected_layer_types.h"
#include "algorithms/neural_networks/layers/fullyconnected/fullyconnected_layer_backward_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace fullyconnected
{
namespace backward
{
namespace interface1
{
/**
 * @defgroup fullyconnected_backward_batch Batch
 * @ingroup fullyconnected_backward
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FULLYCONNECTED__BACKWARD__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the of the backward fully-connected layer
 *        This class is associated with the daal::algorithms::neural_networks::layers::fullyconnected::backward::Batch class
 *        and supports the method of backward fully-connected layer computation in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of backward fully-connected layer, double or float
 * \tparam method           Computation method of the layer, \ref daal::algorithms::neural_networks::layers::fullyconnected::Method
 * \tparam cpu              Version of the cpu-specific implementation of the layer, \ref daal::CpuType
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the backward fully-connected layer with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the backward fully-connected layer in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FULLYCONNECTED__BACKWARD__BATCH"></a>
 * \brief Provides methods for backward fully-connected layer computations in the batch processing mode
 * <!-- \n<a href="DAAL-REF-FULLYCONNECTEDBACKWARD-ALGORITHM">Backward fully-connected layer description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of backward fully-connected layer, double or float
 * \tparam method           Computation method of the layer, \ref Method
 *
 * \par Enumerations
 *      - \ref Method          Computation methods for the backward fully-connected layer
 *      - \ref LayerDataId  Identifiers of input objects for the backward fully-connected layer
 *
 * \par References
 *      - forward::interface1::Batch class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public layers::backward::LayerIfaceImpl
{
public:
    typedef layers::backward::LayerIfaceImpl super;

    typedef algorithms::neural_networks::layers::fullyconnected::backward::Input     InputType;
    typedef algorithms::neural_networks::layers::fullyconnected::Parameter           ParameterType;
    typedef algorithms::neural_networks::layers::fullyconnected::backward::Result    ResultType;

    ParameterType &parameter; /*!< \ref interface1::Parameter "Parameters" of the layer */
    InputType input;          /*!< %Input objects of the layer */

    /**
     * Constructs backward fully-connected layer
     *  \param[in] nOutputs A number of layer outputs
     */
    Batch(const size_t nOutputs) : _defaultParameter(nOutputs), parameter(_defaultParameter)
    {
        initialize();
    };

    /**
     * Constructs a backward fully-connected layer in the batch processing mode
     * and initializes its parameter with the provided parameter
     * \param[in] parameter Parameter to initialize the parameter of the layer
     */
    Batch(ParameterType& parameter) : parameter(parameter), _defaultParameter(parameter)
    {
        initialize();
    }

    /**
     * Constructs backward fully-connected layer by copying input objects and parameters of another fully-connected layer
     * \param[in] other A layer to be used as the source to initialize the input objects
     *                  and parameters of this layer
     */
    Batch(const Batch<algorithmFPType, method> &other) : super(other),
        _defaultParameter(other.parameter), parameter(_defaultParameter), input(other.input)
    {
        initialize();
    }

    /**
     * Returns computation method of the layer
     * \return Computation method of the layer
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains input objects of fully-connected layer
     * \return Structure that contains input objects of fully-connected layer
     */
    virtual InputType *getLayerInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the structure that contains parameters of the backward fully-connected layer
     * \return Structure that contains parameters of the backward fully-connected layer
     */
    virtual ParameterType *getLayerParameter() DAAL_C11_OVERRIDE { return &parameter; };

    /**
     * Returns the structure that contains results of fully-connected layer
     * \return Structure that contains results of fully-connected layer
     */
    layers::backward::ResultPtr getLayerResult() DAAL_C11_OVERRIDE
    {
        return getResult();
    }

    /**
     * Returns the structure that contains results of fully-connected layer
     * \return Structure that contains results of fully-connected layer
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store results of fully-connected layer
     * \param[in] result  Structure to store  results of fully-connected layer
     *
     * \return Status of computations
     */
    services::Status setResult(const ResultPtr& result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res = _result.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated fully-connected layer
     * with a copy of input objects and parameters of this fully-connected layer
     * \return Pointer to the newly allocated layer
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
     * Allocates memory to store the result of the backward fully-connected layer
     *
     * \return Status of computations
     */
    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = this->_result->template allocate<algorithmFPType>(&(this->input), &parameter, (int) method);
        this->_res = this->_result.get();
        return s;
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
        _par = &parameter;
        _result.reset(new ResultType());
    }

private:
    ResultPtr _result;
    ParameterType _defaultParameter;
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;
} // namespace backward
} // namespace fullyconnected
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
