/* file: neural_networks_training_feedforward_kernel.h */
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

//++
//  Declaration of template function that calculate neural networks.
//--


#ifndef __NEURAL_NETWORKS_TRAINING_FEEDFORWARD_KERNEL_H__
#define __NEURAL_NETWORKS_TRAINING_FEEDFORWARD_KERNEL_H__

#include "neural_networks/neural_networks_training.h"
#include "neural_networks/neural_networks_training_types.h"
#include "neural_networks_training_service.h"
#include "kernel.h"
#include "numeric_table.h"
#include "service_numeric_table.h"
#include "../objective_function/cross_entropy/cross_entropy_batch.h"
#include "../objective_function/precomputed/precomputed_batch.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace training
{
namespace internal
{
/**
 *  \brief Kernel for neural network calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class NeuralNetworksFeedforwardTrainingKernel : public Kernel
{
public:
    void compute(const Input *input, const neural_networks::training::Parameter<algorithmFPType> *parameter, Result *result);
};

} // namespace daal::internal
} // namespace training
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
