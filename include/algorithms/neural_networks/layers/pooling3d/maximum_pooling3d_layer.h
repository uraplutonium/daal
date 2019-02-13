/* file: maximum_pooling3d_layer.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of maximum 3D pooling layer.
//--
*/

#ifndef __MAXIMUM_POOLING3D_LAYER_H__
#define __MAXIMUM_POOLING3D_LAYER_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/pooling3d/maximum_pooling3d_layer_types.h"
#include "algorithms/neural_networks/layers/pooling3d/maximum_pooling3d_layer_forward.h"
#include "algorithms/neural_networks/layers/pooling3d/maximum_pooling3d_layer_backward.h"


namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * \brief Contains classes for maximum three-dimensional (3D) pooling layer
 */
namespace maximum_pooling3d
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @ingroup maximum_pooling3d
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__MAXIMUM_POOLING3D__BATCH"></a>
 * \brief Provides methods for the maximum 3D pooling layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-MAXIMUMPOOLING3DFORWARD-ALGORITHM">Forward maximum 3D pooling layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-MAXIMUMPOOLING3DBACKWARD-ALGORITHM">Backward maximum 3D pooling layer description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the maximum 3D pooling layer, double or float
 * \tparam method           Maximum 3D pooling layer method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method      Computation methods
 *      - \ref LayerDataId Identifiers of collection in results of forward and in input objects for the backward average 3D pooling layer
 *
 * \par References
 *      - \ref forward::interface1::Batch "forward::Batch" class
 *      - \ref backward::interface1::Batch "backward::Batch" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public LayerIface
{
public:
    /**
     * Constructs the maximum 3D pooling layer
     * \param[in] nDimensions Number of dimensions in input gradient tensor
     */
    Batch(size_t nDimensions) : parameter(nDimensions - 3, nDimensions - 2, nDimensions - 1)
    {
        forward::Batch<algorithmFPType, method> *forwardLayerObject = new forward::Batch<algorithmFPType, method>(parameter);
        backward::Batch<algorithmFPType, method> *backwardLayerObject = new backward::Batch<algorithmFPType, method>(parameter);

        LayerIface::forwardLayer = services::SharedPtr<forward::Batch<algorithmFPType, method> >(forwardLayerObject);
        LayerIface::backwardLayer = services::SharedPtr<backward::Batch<algorithmFPType, method> >(backwardLayerObject);
    }

    Parameter parameter; /*!< Maximum 3D pooling layer parameters */
};
/** @} */
} // namespace interface1
using interface1::Batch;

} // namespace maximum_pooling3d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
