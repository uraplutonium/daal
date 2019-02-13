/* file: decision_tree_regression_predict_batch.cpp */
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
//  Implementation of the interface for Decision tree model-based prediction
//--
*/

#include "algorithm.h"
#include "serialization_utils.h"
#include "algorithms/decision_tree/decision_tree_regression_predict_types.h"
#include "decision_tree_regression_model_impl.h"
#include "daal_strings.h"

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

Input::Input() : algorithms::regression::prediction::Input(lastModelInputId + 1) {}
Input::Input(const Input &other) : algorithms::regression::prediction::Input(other) {}

NumericTablePtr Input::get(NumericTableInputId id) const
{
    return algorithms::regression::prediction::Input::get(algorithms::regression::prediction::NumericTableInputId(id));
}

ModelPtr Input::get(ModelInputId id) const
{
    return staticPointerCast<decision_tree::regression::interface1::Model, data_management::SerializationIface>(Argument::get(id));
}

void Input::set(NumericTableInputId id, const data_management::NumericTablePtr & ptr)
{
    algorithms::regression::prediction::Input::set(algorithms::regression::prediction::NumericTableInputId(id), ptr);
}

void Input::set(ModelInputId id, const ModelPtr & value)
{
    algorithms::regression::prediction::Input::set(algorithms::regression::prediction::ModelInputId(id), value);
}

Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    return algorithms::regression::prediction::Input::check(parameter, method);
}

Result::Result() : algorithms::regression::prediction::Result(lastResultId + 1) {}

NumericTablePtr Result::get(ResultId id) const
{
    return algorithms::regression::prediction::Result::get(algorithms::regression::prediction::ResultId(id));
}

void Result::set(ResultId id, const NumericTablePtr &value)
{
    algorithms::regression::prediction::Result::set(algorithms::regression::prediction::ResultId(id), value);
}

Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, algorithms::regression::prediction::Result::check(input, par, method));
    DAAL_CHECK_EX(get(prediction)->getNumberOfColumns() == 1, ErrorIncorrectNumberOfColumns, ArgumentName, predictionStr());
    return s;
}

} // namespace interface1
} // namespace prediction
} // namespace regression
} // namespace decision_tree
} // namespace algorithms
} // namespace daal
