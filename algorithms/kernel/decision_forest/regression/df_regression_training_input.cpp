/* file: df_regression_training_input.cpp */
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
//  Implementation of decision forest algorithm classes.
//--
*/

#include "algorithms/decision_forest/decision_forest_regression_training_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace training { services::Status checkImpl(const decision_forest::training::Parameter& prm); }

namespace regression
{
namespace training
{
namespace interface1
{

Parameter::Parameter(){}
services::Status Parameter::check() const
{
    return decision_forest::training::checkImpl(*this);
}

/** Default constructor */
Input::Input() : daal::algorithms::Input(2) {}

/**
 * Returns an input object for decision forest model-based training
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for decision forest model-based training
 * \param[in] id      Identifier of the input object
 * \param[in] value   Pointer to the object
 */
void Input::set(InputId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
size_t Input::getNFeatures() const { return get(data)->getNumberOfColumns(); }

/**
* Checks an input object for the decision forest algorithm
* \param[in] par     Algorithm parameter
* \param[in] method  Computation method
*/

services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    DAAL_CHECK(Argument::size() == 2, ErrorIncorrectNumberOfInputNumericTables);

    NumericTablePtr dataTable = get(data);
    NumericTablePtr dependentVariableTable = get(dependentVariable);

    services::Status s;
    DAAL_CHECK_STATUS(s, checkNumericTable(dataTable.get(), dataStr()));

    size_t nRowsInData = dataTable->getNumberOfRows();
    const Parameter* parameter = static_cast<const Parameter*>(par);
    DAAL_CHECK_STATUS(s, checkNumericTable(dependentVariableTable.get(), dependentVariableStr(), 0, 0, 1, nRowsInData));
    const size_t nSamplesPerTree(parameter->observationsPerTreeFraction*dataTable->getNumberOfRows());
    DAAL_CHECK_EX(nSamplesPerTree > 0,
        services::ErrorIncorrectParameter, services::ParameterName, observationsPerTreeFractionStr());
    const auto nFeatures = dataTable->getNumberOfColumns();
    DAAL_CHECK_EX(parameter->featuresPerNode <= nFeatures,
        services::ErrorIncorrectParameter, services::ParameterName, featuresPerNodeStr());
    return s;
}

} // namespace interface1
} // namespace training
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
