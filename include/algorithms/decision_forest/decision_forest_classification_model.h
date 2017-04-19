/* file: decision_forest_classification_model.h */
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
//  Implementation of class defining decision_forest classification model.
//--
*/

#ifndef __DECISION_FOREST_CLASSIFICATION_MODEL_H__
#define __DECISION_FOREST_CLASSIFICATION_MODEL_H__

#include "algorithms/algorithm.h"
#include "algorithms/classifier/classifier_model.h"

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
/**
 * \brief Contains classes for the decision_forest classification algorithm
 */
namespace classification
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @ingroup decision_forest_classification
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__CLASSIFICATION__MODEL"></a>
 * \brief %Model of the classifier trained by the decision_forest::training::Batch algorithm.
 *
 * \par References
 *      - \ref classification::training::interface1::Batch "training::Batch" class
 *      - \ref classification::prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public classifier::Model
{
public:
    DECLARE_SERIALIZABLE();
    DAAL_DOWN_CAST_OPERATOR(Model, classifier::Model)

    /**
     * Empty constructor for deserialization
     */
    Model() : classifier::Model()
    {}

protected:
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        classifier::Model::serialImpl<Archive, onDeserialize>(arch);
    }

    void serializeImpl(data_management::InputDataArchive *archive) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(archive);}

    void deserializeImpl(data_management::OutputDataArchive *archive) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(archive);}
};
/** @} */
typedef services::SharedPtr<Model> ModelPtr;
} // namespace interface1
using interface1::Model;
using interface1::ModelPtr;

} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
#endif
