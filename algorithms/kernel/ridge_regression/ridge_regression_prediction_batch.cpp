/* file: ridge_regression_prediction_batch.cpp */
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
//  Implementation of ridge regression algorithm classes.
//--
*/

#include "algorithms/ridge_regression/ridge_regression_predict_types.h"
#include "serialization_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace prediction
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_RIDGE_REGRESSION_PREDICTION_RESULT_ID);

/** Default constructor */
Input::Input() : linear_model::prediction::Input(lastModelInputId + 1) {}
Input::Input(const Input& other) : linear_model::prediction::Input(other){}

/**
 * Returns an input object for making ridge regression model-based prediction
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(NumericTableInputId id) const
{
    return linear_model::prediction::Input::get(linear_model::prediction::NumericTableInputId(id));
}

/**
 * Returns an input object for making ridge regression model-based prediction
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
ridge_regression::ModelPtr Input::get(ModelInputId id) const
{
    return ridge_regression::Model::cast(linear_model::prediction::Input::get(linear_model::prediction::ModelInputId(id)));
}

/**
 * Sets an input object for making ridge regression model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Input::set(NumericTableInputId id, const NumericTablePtr &value)
{
    linear_model::prediction::Input::set(linear_model::prediction::NumericTableInputId(id), value);
}

/**
 * Sets an input object for making ridge regression model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Input::set(ModelInputId id, const ridge_regression::ModelPtr & value)
{
    linear_model::prediction::Input::set(linear_model::prediction::ModelInputId(id), value);
}


Result::Result() : linear_model::prediction::Result(lastResultId + 1) {}

/**
 * Returns the result of ridge regression model-based prediction
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return linear_model::prediction::Result::get(linear_model::prediction::ResultId(id));
}

/**
 * Sets the result of ridge regression model-based prediction
 * \param[in] id      Identifier of the input object
 * \param[in] value   %Input object
 */
void Result::set(ResultId id, const NumericTablePtr &value)
{
    linear_model::prediction::Result::set(linear_model::prediction::ResultId(id), value);
}

} // namespace interface1
} // namespace prediction
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal
