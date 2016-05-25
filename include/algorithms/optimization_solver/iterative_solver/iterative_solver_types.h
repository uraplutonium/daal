/* file: iterative_solver_types.h */
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
//  Implementation of the iterative solver algorithm types.
//--
*/

#ifndef __ITERATIVE_SOLVER_TYPES_H__
#define __ITERATIVE_SOLVER_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/optimization_solver/objective_function/sum_of_functions_batch.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
/**
 * \brief Contains classes for computing the iterative solver
 */
namespace iterative_solver
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__INPUTID"></a>
 * Available identifiers of input for the iterative solver
 */
enum InputId
{
    inputArgument = 0 /*!< Initial value to start optimization */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__RESULTID"></a>
 * Available identifiers of results for the iterative solver algorithm
 */
enum ResultId
{
    minimum = 0,    /*!< Numeric table of size 1 x p with the argument */
    nIterations = 1 /*!< Table containing the number of executed iterations */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__PARAMETER"></a>
 * \brief %Parameter base class for the iterative solver algorithm
 *
 * \snippet optimization_solver/iterative_solver/iterative_solver_types.h Parameter source code
 */
/* [Parameter source code] */
struct Parameter : public daal::algorithms::Parameter
{
    /**
     * Constructs the parameter base class of the iterative solver algorithm
     * \param[in] function_             Objective function represented as sum of functions
     * \param[in] nIterations_          Maximal number of iterations of the algorithm
     * \param[in] accuracyThreshold_    Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
     */
    Parameter(services::SharedPtr<sum_of_functions::Batch> function_, const size_t nIterations_ = 100, const double accuracyThreshold_ = 1.0e-05) :
        function(function_), nIterations(nIterations_), accuracyThreshold(accuracyThreshold_)
    {}

    /**
    * Constructs an Parameter by copying input objects and parameters of another Parameter
    * \param[in] other An object to be used as the source to initialize object
    */
    Parameter(const Parameter &other) : function(other.function), nIterations(other.nIterations), accuracyThreshold(other.accuracyThreshold)
    {}

    virtual ~Parameter() {}

    /**
     * Checks the correctness of the parameter
     */
    virtual void check() const
    {
        if(function.get() == NULL)
        {
            services::SharedPtr<services::Error> error(new services::Error());
            error->setId(services::ErrorIncorrectParameter);
            error->addStringDetail(services::ArgumentName, "function");
            this->_errors->add(error);
            return;
        }

        if(function->sumOfFunctionsParameter == NULL)
        {
            services::SharedPtr<services::Error> error(new services::Error());
            error->setId(services::ErrorNullParameterNotSupported);
            error->addStringDetail(services::ArgumentName, "sumOfFunctionsParameter");
            this->_errors->add(error);
            return;
        }

        if(function->sumOfFunctionsInput == NULL)
        {
            services::SharedPtr<services::Error> error(new services::Error());
            error->setId(services::ErrorNullInput);
            error->addStringDetail(services::ArgumentName, "sumOfFunctionsInput");
            this->_errors->add(error);
            return;
        }

        if(accuracyThreshold < 0)
        {
            services::SharedPtr<services::Error> error(new services::Error());
            error->setId(services::ErrorIncorrectParameter);
            error->addStringDetail(services::ArgumentName, "accuracyThreshold");
            this->_errors->add(error);
            return;
        }
    }

    services::SharedPtr<sum_of_functions::Batch>       function;    /*!< Objective function represented as sum of functions */
    size_t                                             nIterations; /*!< Maximal number of iterations of the algorithm */
    double                                             accuracyThreshold;    /*!< Accuracy of the algorithm. The algorithm terminates when
                                                                                  this accuracy is achieved */

};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__INPUT"></a>
 * \brief %Input parameters for the iterative solver algorithm
 */
class Input :  public daal::algorithms::Input
{
public:
    Input() : daal::algorithms::Input(1) {}

    virtual ~Input() {}

    /**
     * Returns input NumericTable of the iterative solver algorithm
     * \param[in] id    Identifier of the input numeric table
     * \return          %Input numeric table that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets input for the iterative solver algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks the correctness of the input
     * \param[in] par       Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        services::SharedPtr<services::Error> error(new services::Error());

        if(this->size() != 1) {this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        error = checkTable(get(inputArgument), "inputArgument", 1);
        if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }
    }

protected:
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

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__RESULT"></a>
 * \brief Results obtained with the compute() method of the iterative solver algorithm in the batch processing mode
 */
class Result : public daal::algorithms::Result
{
public:
    Result() : daal::algorithms::Result(2)
    {}

    virtual ~Result() {};

    /**
     * Allocates memory to store the results of the iterative solver algorithm
     * \param[in] input  Pointer to the input structure
     * \param[in] par    Pointer to the parameter structure
     * \param[in] method Computation method of the algorithm
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
    {
        const Input *algInput = static_cast<const Input *>(input);
        size_t nFeatures = algInput->get(inputArgument)->getNumberOfColumns();
        Argument::set(minimum, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(nFeatures, 1, data_management::NumericTable::doAllocate, 0.0)));
        Argument::set(nIterations, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<size_t>(1, 1, data_management::NumericTable::doAllocate, (size_t)0)));
    }

    /**
     * Returns result of the iterative solver algorithm
     * \param[in] id   Identifier of the result
     * \return         Final result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the result of the iterative solver algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks the result of the iterative solver algorithm
     * \param[in] input   %Input of algorithm
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method of the algorithm
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
               int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        const Input *algInput = static_cast<const Input *>(input);
        size_t nFeatures = algInput->get(inputArgument)->getNumberOfColumns();

        services::SharedPtr<services::Error> error(new services::Error());
        error = checkTable(get(minimum), "minimum", 1, nFeatures);
        if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }

        error = checkTable(get(nIterations), "nIterations", 1, 1);
        if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }
    }

    /**
     * Returns a serialization tag, a unique identifier of this class used in serialization
     * \return Serialization tag
     */
    int getSerializationTag() DAAL_C11_OVERRIDE { return SERIALIZATION_ITERATIVE_SOLVER_RESULT_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for the serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes the object
    *  \param[in]  arch  Storage for the deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
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
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};

} // namespace interface1
using interface1::Parameter;
using interface1::Input;
using interface1::Result;

} // namespace iterative_solver
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif
