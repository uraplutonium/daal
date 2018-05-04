/* file: lrn_layer_types.h */
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
//  Implementation of the local response normalization layer types.
//--
*/

#ifndef __LRN_LAYER_TYPES_H__
#define __LRN_LAYER_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * @defgroup lrn Local Response Normalization Layer
 * \copydoc daal::algorithms::neural_networks::layers::lrn
 * @ingroup layers
 * @{
 */
namespace lrn
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LRN__METHOD"></a>
 * \brief Computation methods for the local response normalization layer
 */
enum Method
{
    defaultDense = 0, /*!<  Default: performance-oriented method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LRN__LAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward stage and results for the forward stage of the local response normalization layer
 */
enum LayerDataId
{
    auxData = layers::lastLayerInputLayout + 1, /*!< Data processed at the forward stage of the layer */
    auxSmBeta, /*!< Pointer to the tensor of size n1 x n2 x ... x np, that stores value of (kappa + alpha * sum((x_i)^2))^(-beta) */
    lastLayerDataId = auxSmBeta
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LRN__PARAMETER"></a>
 * \brief Parameters for the local response normalization layer
 *
 * \snippet neural_networks/layers/lrn/lrn_layer_types.h Parameter source code
 */
/* [Parameter source code] */
class DAAL_EXPORT Parameter: public layers::Parameter
{
public:
    /**
    *  Constructs parameters of the local response normalization layer
    *  \param[in] dimension_ Numeric table of size 1 x 1 with index of type size_t to calculate local response normalization.
    *  \param[in] kappa_     Value of hyper-parameter kappa
    *  \param[in] alpha_     Value of hyper-parameter alpha
    *  \param[in] beta_      Value of hyper-parameter beta
    *  \param[in] nAdjust_   Value of hyper-parameter n
    */
    Parameter(
        data_management::NumericTablePtr dimension_ = data_management::HomogenNumericTable<size_t>::create(1, 1, data_management::NumericTableIface::doAllocate, 1),
        const double kappa_ = 2,
        const double alpha_ = 1.0e-04,
        const double beta_ = 0.75,
        const size_t nAdjust_ = 5 );

    data_management::NumericTablePtr dimension; /*!< Numeric table of size 1 x 1 with index of type size_t
                                                                       to calculate local response normalization. */
    double kappa;     /*!< Value of hyper-parameter kappa */
    double alpha;     /*!< Value of hyper-parameter alpha */
    double beta;      /*!< Value of hyper-parameter beta */
    size_t nAdjust;   /*!< Value of hyper-parameter n */

    /**
     * Checks the correctness of the parameter
     *
     * \return Status of computations
     */
    virtual services::Status check() const;
};
/* [Parameter source code] */

} // namespace interface1
using interface1::Parameter;

} // namespace lrn
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
