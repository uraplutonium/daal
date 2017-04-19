/* file: decision_forest_regression_model.h */
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
//  Implementation of the class defining the decision forest regression model
//--
*/

#ifndef __DECISION_FOREST_REGRESSION_MODEL_H__
#define __DECISION_FOREST_REGRESSION_MODEL_H__

#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/model.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * @ingroup decision_forest_regression
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST_REGRESSION_MODEL"></a>
 * \brief %Base class for models trained with the decision forest regression algorithm
 *
 * \tparam modelFPType  Data type to store decision forest model data, double or float
 *
 * \par References
 *      - \ref regression::training::interface1::Batch "regression::training::Batch" class
 *      - \ref regression::prediction::interface1::Batch "regression::prediction::Batch" class
 */
class DAAL_EXPORT Model : public daal::algorithms::Model
{
public:
    DECLARE_SERIALIZABLE();

    virtual ~Model();

protected:
    Model();
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Model::serialImpl<Archive, onDeserialize>(arch);
    }

    void serializeImpl(data_management::InputDataArchive *archive) DAAL_C11_OVERRIDE {}

    void deserializeImpl(data_management::OutputDataArchive *archive) DAAL_C11_OVERRIDE {}
};
typedef services::SharedPtr<Model> ModelPtr;
typedef services::SharedPtr<const Model> ModelConstPtr;

/** @} */
} // namespace interface1
using interface1::Model;
using interface1::ModelPtr;
using interface1::ModelConstPtr;

}
}
}
}
#endif
