/* file: lrn_layer_forward_fpt.cpp */
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
//  Implementation of lrn calculation algorithm and types methods.
//--
*/

#include "lrn_layer_forward_types.h"
#include "lrn_layer_types.h"

#include "tensor.h"
#include "service_mkl_tensor.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace lrn
{
namespace forward
{
namespace interface1
{

using namespace daal::data_management;

/**
 * Allocates memory to store the result of the forward local response normalization layer
 * \param[in] input Pointer to an object containing the input data
 * \param[in] method Computation method for the layer
 * \param[in] parameter %Parameter of the forward local response normalization layer
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    using daal::data_management::Tensor;
    using daal::data_management::TensorPtr;
    using daal::internal::MklTensor;

    const layers::forward::Input *in = static_cast<const layers::forward::Input * >(input);

    if (!get(layers::forward::value))
    {
        set(layers::forward::value, TensorPtr(new MklTensor<algorithmFPType>(in->get(layers::forward::data)->getDimensions())));
    }
    if (!get(layers::forward::resultForBackward))
    {
        set(layers::forward::resultForBackward, LayerDataPtr(new LayerData()));
    }
    if (!get(lrn::auxSmBeta))
    {
        set(lrn::auxSmBeta, TensorPtr(new MklTensor<algorithmFPType>(in->get(layers::forward::data)->getDimensions())));
    }

    services::Status s;
    const layers::Parameter *par = static_cast<const layers::Parameter * >(parameter);
    if(!par->predictionStage)
    {
        s |= setResultForBackward(input);
    }
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace forward
}// namespace lrn
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
