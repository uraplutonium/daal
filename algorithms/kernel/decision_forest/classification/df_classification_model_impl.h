/* file: df_classification_model_impl.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of the class defining the decision forest model
//--
*/

#ifndef __DF_CLASSIFICATION_MODEL_IMPL__
#define __DF_CLASSIFICATION_MODEL_IMPL__

#include "df_model_impl.h"
#include "algorithms/decision_forest/decision_forest_classification_model.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace internal
{

class ModelImpl : public daal::algorithms::decision_forest::classification::Model,
    public daal::algorithms::decision_forest::internal::ModelImpl
{
public:
    ModelImpl(){}
    ~ModelImpl(){}
};

} // namespace internal
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
