/* file: layer_types.h */
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
//  Implementation of neural_networks Network layer.
//--
*/

#ifndef __NEURAL_NETWORKS__LAYERS__TYPES_H__
#define __NEURAL_NETWORKS__LAYERS__TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "services/collection.h"
#include "data_management/data/data_collection.h"
#include "algorithms/neural_networks/initializers/initializer.h"
#include "algorithms/neural_networks/initializers/uniform/uniform_initializer.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for training and prediction using neural network
 */
namespace neural_networks
{
/**
 * \brief Contains classes for neural network layers
 */
namespace layers
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LAYERINPUTLAYOUT"></a>
 * Available identifiers of layouts of the input of the layer
 */
enum LayerInputLayout
{
    tensorInput = 0,
    collectionInput = 1,
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LAYERRESULTLAYOUT"></a>
 * Available identifiers of layouts of the result of the layer
 */
enum LayerResultLayout
{
    tensorResult = 0,
    collectionResult = 1,
};
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__PARAMETER"></a>
 * Parameters of the neural network layer
 */
class Parameter: public daal::algorithms::Parameter
{
public:
    /** Default constructor */
    Parameter(): weightsInitializer(new initializers::uniform::Batch<>()), biasesInitializer(new initializers::uniform::Batch<>()) {};

    /** Layer weights initializer */
    services::SharedPtr<initializers::InitializerIface> weightsInitializer;
    /** Layer biases initializer */
    services::SharedPtr<initializers::InitializerIface> biasesInitializer;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LAYERDATA"></a>
 * \brief Contains extra input and output object of neural network layer
 */
typedef data_management::KeyValueDataCollection LayerData;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__NEXTLAYERS"></a>
 * \brief Contains list of layer indexes of layers following the current layer
 */
class NextLayers : public services::Collection<size_t>
{
public:
    /** \brief Constructor
    */
    NextLayers() : services::Collection<size_t>(0)
    {};

    /** \brief Constructor
    * \param[in] index1    First index of the next layer
    */
    NextLayers(const size_t index1) : services::Collection<size_t>()
    {
        push_back(index1);
    };

    /** \brief Constructor
    * \param[in] index1    First index of the next layer
    * \param[in] index2    Second index of the next layer
    */
    NextLayers(const size_t index1, const size_t index2) : services::Collection<size_t>()
    {
        push_back(index1);
        push_back(index2);
    };

    /** \brief Constructor
    * \param[in] index1    First index of the next layer
    * \param[in] index2    Second index of the next layer
    * \param[in] index3    Third index of the next layer
    */
    NextLayers(const size_t index1, const size_t index2, const size_t index3) : services::Collection<size_t>()
    {
        push_back(index1);
        push_back(index2);
        push_back(index3);
    };

    /** \brief Constructor
    * \param[in] index1    First index of the next layer
    * \param[in] index2    Second index of the next layer
    * \param[in] index3    Third index of the next layer
    * \param[in] index4    Fourth index of the next layer
    */
    NextLayers(const size_t index1, const size_t index2, const size_t index3, const size_t index4) : services::Collection<size_t>()
    {
        push_back(index1);
        push_back(index2);
        push_back(index3);
        push_back(index4);
    };

    /** \brief Constructor
    * \param[in] index1    First index of the next layer
    * \param[in] index2    Second index of the next layer
    * \param[in] index3    Third index of the next layer
    * \param[in] index4    Fourth index of the next layer
    * \param[in] index5    Fifth index of the next layer
    */
    NextLayers(const size_t index1, const size_t index2, const size_t index3, const size_t index4,
               const size_t index5) : services::Collection<size_t>()
    {
        push_back(index1);
        push_back(index2);
        push_back(index3);
        push_back(index4);
        push_back(index5);
    };

    /** \brief Constructor
    * \param[in] index1    First index of the next layer
    * \param[in] index2    Second index of the next layer
    * \param[in] index3    Third index of the next layer
    * \param[in] index4    Fourth index of the next layer
    * \param[in] index5    Fifth index of the next layer
    * \param[in] index6    Sixth index of the next layer
    */
    NextLayers(const size_t index1, const size_t index2, const size_t index3, const size_t index4, const size_t index5,
               const size_t index6) : services::Collection<size_t>()
    {
        push_back(index1);
        push_back(index2);
        push_back(index3);
        push_back(index4);
        push_back(index5);
        push_back(index6);
    };
};
}
using interface1::LayerData;
using interface1::NextLayers;
using interface1::Parameter;

} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
