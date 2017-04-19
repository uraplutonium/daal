/* file: decision_tree_regression_training_input.cpp */
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
//  Implementation of Decision tree algorithm classes.
//--
*/

#include "algorithms/decision_tree/decision_tree_regression_training_types.h"
#include "serialization_utils.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace regression
{
namespace training
{
namespace interface1
{

using namespace daal::data_management;
using namespace daal::services;

Input::Input() : daal::algorithms::Input(4) {}

NumericTablePtr Input::get(decision_tree::regression::training::InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void Input::set(decision_tree::regression::training::InputId id, const data_management::NumericTablePtr & value)
{
    Argument::set(id, value);
}

size_t Input::getNumberOfFeatures() const
{
    return get(data)->getNumberOfColumns();
}

size_t Input::getNumberOfDependentVariables() const
{
    return get(dependentVariables)->getNumberOfColumns();
}

services::Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, daal::algorithms::Input::check(parameter, method));
    return checkImpl(parameter);
}

services::Status Input::checkImpl(const daal::algorithms::Parameter * parameter) const
{
    services::Status s;
    const decision_tree::regression::Parameter * const par = static_cast<const decision_tree::regression::Parameter *>(parameter);

    if (par->pruning == decision_tree::reducedErrorPruning)
    {
        const NumericTablePtr dataForPruningTable = get(dataForPruning);
        DAAL_CHECK_STATUS(s, checkNumericTable(dataForPruningTable.get(), dataForPruningStr(), 0, 0, this->getNumberOfFeatures()));
        const int unexpectedLabelsLayouts = (int)NumericTableIface::upperPackedSymmetricMatrix
                                            | (int)NumericTableIface::lowerPackedSymmetricMatrix
                                            | (int)NumericTableIface::upperPackedTriangularMatrix
                                            | (int)NumericTableIface::lowerPackedTriangularMatrix;
        DAAL_CHECK_STATUS(s, checkNumericTable(get(dependentVariablesForPruning).get(), dependentVariablesForPruningStr(), unexpectedLabelsLayouts,
                          0, 1, dataForPruningTable->getNumberOfRows()));
    }
    else
    {
        DAAL_CHECK_EX(get(dataForPruning).get() == nullptr, ErrorIncorrectOptionalInput, ArgumentName, dataForPruningStr());
        DAAL_CHECK_EX(get(dependentVariablesForPruning).get() == nullptr, ErrorIncorrectOptionalInput, ArgumentName,
                      dependentVariablesForPruningStr());
    }

    const NumericTablePtr dataTable = get(data);
    const NumericTablePtr dependentVariablesTable = get(dependentVariables);

    DAAL_CHECK_STATUS(s, checkNumericTable(dataTable.get(), dataStr()));
    DAAL_CHECK_STATUS(s, checkNumericTable(dependentVariablesTable.get(), dependentVariableStr(), 0, 0, 1, dataTable->getNumberOfRows()));

    return s;
}

} // namespace interface1

using interface1::Input;

} // namespace training
} // namespace regression
} // namespace decision_tree
} // namespace algorithms
} // namespace daal
