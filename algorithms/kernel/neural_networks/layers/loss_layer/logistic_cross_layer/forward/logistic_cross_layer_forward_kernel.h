/* file: logistic_cross_layer_forward_kernel.h */
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

//++
//  Implementation of the forward logistic cross layer
//--


#ifndef __LOGISTIC_CROSS_LAYER_FORWARD_KERNEL_H__
#define __LOGISTIC_CROSS_LAYER_FORWARD_KERNEL_H__

// #include "neural_networks/layers/logistic_layer/forward/logistic_layer_forward_kernel.h"
#include "neural_networks/layers/loss/logistic_cross_layer.h"
#include "neural_networks/layers/loss/logistic_cross_layer_types.h"
#include "neural_networks/layers/loss/logistic_cross_layer_forward_types.h"
#include "kernel.h"
#include "service_math.h"
#include "service_tensor.h"
#include "service_numeric_table.h"

using namespace daal::data_management;
using namespace daal::services;

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
namespace logistic_cross
{
namespace forward
{
namespace internal
{
/**
 *  \brief Kernel for logistic_cross calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class LogisticCrossKernel : public Kernel
{
public:
    services::Status compute(const Tensor &inputTensor, const Tensor &groundTruthTensor, Tensor &resultTensor);
};

} // internal
} // forward
} // logistic_cross
} // loss
} // layers
} // neural_networks
} // algorithms
} // daal

#endif
