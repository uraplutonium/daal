/* file: decision_tree_regression_predict_batch.cpp */
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
//  Implementation of the interface for Decision tree model-based prediction
//--
*/

#include "algorithm.h"
#include "serialization_utils.h"
#include "algorithms/decision_tree/decision_tree_regression_predict_types.h"
#include "decision_tree_regression_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace regression
{
namespace prediction
{
namespace interface1
{

using namespace daal::data_management;
using namespace daal::services;

__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_DECISION_TREE_REGRESSION_PREDICTION_RESULT_ID);

Input::Input() : daal::algorithms::Input(2) {}

NumericTablePtr Input::get(NumericTableInputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

ModelPtr Input::get(ModelInputId id) const
{
    return services::staticPointerCast<decision_tree::regression::interface1::Model, data_management::SerializationIface>(Argument::get(id));
}

void Input::set(NumericTableInputId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

void Input::set(ModelInputId id, const ModelPtr & value)
{
    Argument::set(id, value);
}

services::Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, daal::algorithms::Input::check(parameter, method));

    DAAL_CHECK(Argument::size() == 2, ErrorIncorrectNumberOfInputNumericTables);
    NumericTablePtr dataTable = get(data);

    DAAL_CHECK_STATUS(s, checkNumericTable(dataTable.get(), dataStr()));

    const decision_tree::regression::ModelPtr model = get(prediction::model);
    DAAL_CHECK(model, ErrorNullModel);
    return s;
}

Result::Result() : daal::algorithms::Result(1) {}

NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void Result::set(ResultId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfOutputNumericTables);
    NumericTableConstPtr predictionTable = get(prediction);

    const Input * const in = static_cast<const Input *>(input);
    size_t nVectors = in->get(data)->getNumberOfRows();

    return checkNumericTable(predictionTable.get(), predictionStr(), 0, 0, 1, nVectors);
}

} // namespace interface1
} // namespace prediction
} // namespace regression
} // namespace decision_tree
} // namespace algorithms
} // namespace daal
