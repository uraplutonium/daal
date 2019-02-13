/* file: ridge_regression_training_partialresult.cpp */
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
__DAAL_REGISTER_SERIALIZATION_CLASS(PartialResult, SERIALIZATION_RIDGE_REGRESSION_PARTIAL_RESULT_ID);

PartialResult::PartialResult() : linear_model::training::PartialResult(lastPartialResultID + 1) {};

/**
* Returns a partial result of ridge regression model-based training
* \param[in] id    Identifier of the partial result
* \return          Partial result that corresponds to the given identifier
*/
daal::algorithms::ridge_regression::ModelPtr PartialResult::get(PartialResultID id) const
{
    return staticPointerCast<daal::algorithms::ridge_regression::Model, SerializationIface>(Argument::get(id));
}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
size_t PartialResult::getNumberOfFeatures() const { return get(partialModel)->getNumberOfFeatures(); }

/**
* Returns the number of dependent variables
* \return Number of dependent variables
*/
size_t PartialResult::getNumberOfDependentVariables() const { return get(partialModel)->getNumberOfResponses(); }

/**
 * Sets an argument of the partial result
 * \param[in] id      Identifier of the argument
 * \param[in] value   Pointer to the argument
 */
void PartialResult::set(PartialResultID id, const daal::algorithms::ridge_regression::ModelPtr &value)
{
    Argument::set(id, value);
}

/**
 * Checks a partial result of the ridge regression algorithm
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 *
 * \return Status of computations
 */
services::Status PartialResult::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfOutputNumericTables);
    /* input object can be an instance of both Input and DistributedInput<step2Master> classes.
       Both classes have multiple inheritance with InputIface as a second base class.
       That's why we use dynamic_cast here. */
    const InputIface *in = dynamic_cast<const InputIface *>(input);

    ridge_regression::ModelPtr partialModel = get(training::partialModel);

    size_t nBeta = in->getNumberOfFeatures() + 1;
    size_t nResponses = in->getNumberOfDependentVariables();

    return ridge_regression::checkModel(partialModel.get(), *par, nBeta, nResponses, method);
}

/**
 * Checks a partial result of the ridge regression algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 *
 * \return Status of computations
 */
services::Status PartialResult::check(const daal::algorithms::Parameter * par, int method) const
{
    DAAL_CHECK(Argument::size() == 1, ErrorIncorrectNumberOfOutputNumericTables);

    ridge_regression::ModelPtr partialModel = get(training::partialModel);
    DAAL_CHECK(partialModel, ErrorNullPartialModel);

    size_t nBeta = partialModel->getNumberOfBetas();
    size_t nResponses = partialModel->getNumberOfResponses();

    return ridge_regression::checkModel(partialModel.get(), *par, nBeta, nResponses, method);
}

} // namespace interface1
} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal
