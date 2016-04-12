/* file: mse_types.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of Mean squared error objective function interface.
//--
*/

#ifndef __MSE_TYPES_H__
#define __MSE_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "sum_of_functions_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for computing the Mean squared error objective function
 */
namespace optimization_solver
{
/**
* \brief Contains classes for computing the Mean squared error objective function
*/
namespace mse
{

/**
  * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__MSE__INPUTID"></a>
  * Available identifiers of input objects of the Mean squared error objective function
  */
enum InputId
{
    argument = (int)sum_of_functions::argument, /*!< Numeric table of size 1 x p with input argument of the objective function */
    data = 1,                                   /*!< Numeric table of size n x p with data */
    dependentVariables = 2                      /*!< Numeric table of size n x 1 with dependent variables */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__MSE__METHOD"></a>
 * Available methods for computing results of Mean squared error objective function
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method. */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__MSE__PARAMETER"></a>
 * \brief %Parameter for Mean squared error objective function
 *
 * \snippet optimization_solver/objective_function/mse_types.h Parameter source code
 */
/* [Parameter source code] */
struct Parameter : public sum_of_functions::Parameter
{
    /**
     * Constructs the parameter of Mean squared error objective function
     * \param[in] numberOfTerms    The number of terms in the function
     * \param[in] batchIndices     Numeric table of size 1 x m where m is batch size that represent
                                   a batch of indices used to compute the function results, e.g.,
                                   value of the sum of the functions. If no indices are provided,
                                   all terms will be used in the computations.
     * \param[in] resultsToCompute 64 bit integer flag that indicates the results to compute
     */
    Parameter(size_t numberOfTerms,
              services::SharedPtr<data_management::NumericTable> batchIndices = services::SharedPtr<data_management::NumericTable>(),
              const DAAL_UINT64 resultsToCompute = objective_function::gradient) :
        sum_of_functions::Parameter(numberOfTerms, batchIndices, resultsToCompute)
    {}

    /**
     * Constructs an Parameter by copying input objects and parameters of another Parameter
     * \param[in] other An object to be used as the source to initialize object
     */
    Parameter(const Parameter &other) :
        sum_of_functions::Parameter(other)
    {}

    /**
     * Checks the correctness of the parameter
     */
    virtual void check() const
    {
        sum_of_functions::Parameter::check();
    }

    virtual ~Parameter() {}
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__MSE__INPUT"></a>
 * \brief %Input objects for the Mean squared error objective function
 */
class Input : public sum_of_functions::Input
{
public:
    /** Default constructor */
    Input() : sum_of_functions::Input(3)
    {}

    /** Destructor */
    virtual ~Input() {}

    /**
     * Sets one input object for Mean squared error objective function
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Returns the input numeric table for Mean squared error objective function
     * \param[in] id    Identifier of the input numeric table
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Checks the correctness of the input
     * \param[in] par       Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        sum_of_functions::Input::check(par, method);
        if(Argument::size() != 3)
        { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        services::SharedPtr<services::Error> error(new services::Error());

        error = checkTable(get(data), "data");
        if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }

        size_t nRowsInData = get(data)->getNumberOfRows();

        error = checkTable(get(dependentVariables), "dependentVariables", nRowsInData, 1);
        if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }

        error = checkTable(get(argument), "argument", 0, get(data)->getNumberOfColumns() + 1);
        if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }
    }

protected:
    /**
     * Checks the correctness of the numeric table
     * \param[in] nt              Pointer to the numeric table
     * \param[in] argumentName    Name of checked argument
     * \param[in] requiredRows    Number of required rows. If it equal 0 or not mentioned, the numeric table can't have 0 rows
     * \param[in] requiredColumns Number of required columns. If it equal 0 or not mentioned, the numeric table can't have 0 columns
     */
    services::SharedPtr<services::Error> checkTable(services::SharedPtr<data_management::NumericTable> nt, const char *argumentName,
            size_t requiredRows = 0, size_t requiredColumns = 0) const
    {
        services::SharedPtr<services::Error> error(new services::Error());
        if(!nt) { error->setId(services::ErrorNullInputNumericTable); }
        else if(nt->getNumberOfRows()    == 0) { error->setId(services::ErrorEmptyInputNumericTable); }
        else if(nt->getNumberOfColumns() == 0) { error->setId(services::ErrorEmptyInputNumericTable); }
        else if(requiredRows    != 0 && nt->getNumberOfRows()    != requiredRows)    { error->setId(services::ErrorIncorrectNumberOfObservations);  }
        else if(requiredColumns != 0 && nt->getNumberOfColumns() != requiredColumns) { error->setId(services::ErrorInconsistentNumberOfColumns);    }
        if(error->id() != services::NoErrorMessageFound) { error->addStringDetail(services::ArgumentName, argumentName);}
        return error;
    }
};

} // namespace interface1
using interface1::Parameter;
using interface1::Input;

} // namespace mse
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif
