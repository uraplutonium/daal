/* file: ridge_regression_training_result.cpp */
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

#include "algorithms/ridge_regression/ridge_regression_training_types.h"
#include "serialization_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace training
{
namespace interface1
{

__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_RIDGE_REGRESSION_TRAINING_RESULT_ID);
Result::Result() : linear_model::training::Result(lastResultId + 1) {}

/**
 * Returns the result of ridge regression model-based training
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
daal::algorithms::ridge_regression::ModelPtr Result::get(ResultId id) const
{
    return ridge_regression::Model::cast(linear_model::training::Result::get(linear_model::training::ResultId(id)));
}

/**
 * Sets the result of ridge regression model-based training
 * \param[in] id      Identifier of the result
 * \param[in] value   Result
 */
void Result::set(ResultId id, const daal::algorithms::ridge_regression::ModelPtr & value)
{
    linear_model::training::Result::set(linear_model::training::ResultId(id), value);
}

/**
 * Checks the result of ridge regression model-based training
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 *
 * \return Status of computations
 */
services::Status Result::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    Status s;
    DAAL_CHECK_STATUS(s, linear_model::training::Result::check(input, par, method));

    /* input object can be an instance of both Input and DistributedInput<step2Master> classes.
       Both classes have multiple inheritance with InputIface as a second base class.
       That's why we use dynamic_cast here. */
    const InputIface *in = dynamic_cast<const InputIface *>(input);

    size_t nBeta = in->getNumberOfFeatures() + 1;
    size_t nResponses = in->getNumberOfDependentVariables();

    const ridge_regression::ModelPtr model = get(training::model);

    return ridge_regression::checkModel(model.get(), *par, nBeta, nResponses, method);
}

/**
 * Checks the result of the ridge regression model-based training
 * \param[in] pr      %PartialResult of the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 *
 * \return Status of computations
 */
services::Status Result::check(const daal::algorithms::PartialResult * pr, const daal::algorithms::Parameter * par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfOutputNumericTables);
    const PartialResult *partRes = static_cast<const PartialResult *>(pr);

    ridge_regression::ModelPtr model = get(training::model);

    size_t nBeta = partRes->getNumberOfFeatures() + 1;
    size_t nResponses = partRes->getNumberOfDependentVariables();

    return ridge_regression::checkModel(model.get(), *par, nBeta, nResponses, method);
}

} // namespace interface1
} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal
