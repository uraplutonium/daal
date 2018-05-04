/* file: softmax_cross_layer.h */
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
//  Implementation of the softmax cross layer.
//--
*/

#ifndef __NEURAL_NENTWORK_LOSS_SOFTMAX_CROSS_LAYER_H__
#define __NEURAL_NENTWORK_LOSS_SOFTMAX_CROSS_LAYER_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/loss/loss_layer.h"
#include "algorithms/neural_networks/layers/loss/softmax_cross_layer_types.h"
#include "algorithms/neural_networks/layers/loss/softmax_cross_layer_forward.h"
#include "algorithms/neural_networks/layers/loss/softmax_cross_layer_backward.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace loss
{
/**
 * \brief Contains classes for softmax cross-entropy layer
 */
namespace softmax_cross
{
namespace interface1
{
/**
 * @ingroup softmax_cross
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__SOFTMAX_CROSS__BATCH"></a>
 * \brief Provides methods for the softmax cross-entropy layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-SOFTMAX_CROSSFORWARD-ALGORITHM">Forward softmax cross-entropy layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-SOFTMAX_CROSSBACKWARD-ALGORITHM">Backward softmax cross-entropy layer description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the softmax cross-entropy layer, double or float
 * \tparam method           Dropout layer method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref forward::interface1::Batch "forward::Batch" class
 *      - \ref backward::interface1::Batch "backward::Batch" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public loss::Batch
{
public:
    /** Default constructor */
    Batch()
    {
        forward::Batch<algorithmFPType, method> *forwardLayerObject = new forward::Batch<algorithmFPType, method>(parameter);
        backward::Batch<algorithmFPType, method> *backwardLayerObject = new backward::Batch<algorithmFPType, method>(parameter);

        LayerIface::forwardLayer = services::SharedPtr<forward::Batch<algorithmFPType, method> >(forwardLayerObject);
        LayerIface::backwardLayer = services::SharedPtr<backward::Batch<algorithmFPType, method> >(backwardLayerObject);
    };

    Parameter parameter; /*!< Loss layer parameters */
};
/** @} */
} // namespace interface1
using interface1::Batch;

} // namespace softmax_cross
} // namespace loss
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
