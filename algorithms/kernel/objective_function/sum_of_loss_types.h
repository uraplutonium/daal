/* file: sum_of_loss_types.h */
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
//  Implementation of sum of loss objective function interface.
//--
*/

#ifndef __SUM_OF_LOSS_TYPES_H__
#define __SUM_OF_LOSS_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "sum_of_functions_types.h"


namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for computing the sum of loss objective function
 */
namespace optimization_solver
{
namespace internal
{
namespace sum_of_loss
{

/**
  * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__sum_of_loss__INPUTID"></a>
  * Available identifiers of input objects of the sum of loss objective function
  */
enum InputId
{
    probabilities = 0, // [n x p]
    groundTruth = 1 // [n x 1]
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__sum_of_loss__PARAMETER"></a>
 * \brief %Parameter for sum of loss objective function
 *
 * \snippet optimization_solver/objective_function/sum_of_loss_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public sum_of_functions::Parameter
{
    /**
     * Constructs the parameter of sum of loss objective function
     * \param[in] numberOfTerms    The number of terms in the function
     * \param[in] batchIndices     Numeric table of size 1 x m where m is batch size that represent
                                   a batch of indices used to compute the function results, e.g.,
                                   value of the sum of the functions. If no indices are provided,
                                   all terms will be used in the computations.
     * \param[in] resultsToCompute 64 bit integer flag that indicates the results to compute
     */
    Parameter(size_t numberOfTerms,
              services::SharedPtr<data_management::NumericTable> batchIndices = services::SharedPtr<data_management::NumericTable>(),
              const DAAL_UINT64 resultsToCompute = objective_function::gradient);

    /**
     * Constructs an Parameter by copying input objects and parameters of another Parameter
     * \param[in] other An object to be used as the source to initialize object
     */
    Parameter(const Parameter &other);

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
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__sum_of_loss__INPUT"></a>
 * \brief %Input objects for the sum of loss objective function
 */
class Input : public sum_of_functions::Input
{
public:
    /** Default constructor */
    Input(size_t n) : sum_of_functions::Input(n)
    {}

    /** Destructor */
    virtual ~Input() {}

    /**
     * Sets one input object for sum of loss objective function
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Returns the input numeric table for sum of loss objective function
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
        if(Argument::size() != 2)
        { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        services::SharedPtr<services::Error> error(new services::Error());

        error = checkTable(get(probabilities), "probabilities");
        if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }

        error = checkTable(get(groundTruth), "groundTruth", get(probabilities)->getNumberOfRows(), 1);
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

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the Objective function in the batch processing mode
 */
class Result: public objective_function::Result
{
public:
    /** Default constructor */
    Result() : objective_function::Result()
    {}

    /** Destructor */
    virtual ~Result() {};

    /**
     * Allocates memory for storing results of the Objective function
     * \param[in] input     Pointer to the input structure
     * \param[in] parameter Pointer to the parameter structure
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        using namespace services;
        using namespace data_management;

        const Input *algInput = static_cast<const Input *>(input);
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);
        if(algParameter == 0)
        {
            this->_errors->add(ErrorNullParameterNotSupported); return;
        }

        size_t nFeatures = algInput->get(probabilities)->getNumberOfColumns();
        size_t nRows = algInput->get(probabilities)->getNumberOfRows();

        SharedPtr<DataCollection> collection = SharedPtr<DataCollection>(new DataCollection(3));

        if(algParameter->resultsToCompute & objective_function::gradient)
        {
            (*collection)[(int)objective_function::gradientIdx] =
                SharedPtr<SerializationIface>(new HomogenNumericTable<algorithmFPType>(nFeatures, nRows, NumericTable::doAllocate, 0));
        }
        if(algParameter->resultsToCompute & objective_function::value)
        {
            (*collection)[(int)objective_function::valueIdx] =
                SharedPtr<SerializationIface>(new HomogenNumericTable<algorithmFPType>(1, 1, NumericTable::doAllocate, 0));
        }

        Argument::set(objective_function::resultCollection, staticPointerCast<DataCollection, SerializationIface>(collection));
    }

    /**
    * Checks the result of the Objective function
    * \param[in] input   %Input of the algorithm
    * \param[in] par     %Parameter of algorithm
    * \param[in] method  Computation method
    */
    virtual void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        using namespace services;

        SharedPtr<Error> error(new Error());

        if(Argument::size() != 1) { this->_errors->add(ErrorIncorrectNumberOfArguments); return; }

        const Input *algInput = static_cast<const Input *>(input);
        const Parameter *algParameter = static_cast<const Parameter *>(par);
        if(algParameter == 0)
        { this->_errors->add(ErrorNullParameterNotSupported); return; }

        size_t nFeatures = algInput->get(probabilities)->getNumberOfColumns();
        size_t nRows = algInput->get(probabilities)->getNumberOfRows();

        if(algParameter->resultsToCompute & objective_function::value)
        {
            error = checkTable(get(objective_function::resultCollection, objective_function::valueIdx), "value", 1, 1);
            if(error->id() != NoErrorMessageFound) { this->_errors->add(error); return; }
        }
        if(algParameter->resultsToCompute & objective_function::gradient)
        {
            error = checkTable(get(objective_function::resultCollection, objective_function::gradientIdx), "gradient", nRows, nFeatures);
            if(error->id() != NoErrorMessageFound) { this->_errors->add(error); return; }
        }
        if(algParameter->resultsToCompute & objective_function::hessian)
        {
            error->setId(services::ErrorNullOutputNumericTable);
            error->addStringDetail(services::ArgumentName, "hessian for sum of loss is not supported");
            this->_errors->add(error); return;
        }
    }

    /**
     * Returns a serialization tag, a unique identifier of this class used in serialization
     * \return Serialization tag
     */
    int getSerializationTag() { return SERIALIZATION_OBJECTIVE_FUNCTION_RESULT_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for a serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes the object
    *  \param[in]  arch  Storage for a deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }

    /**
     * Checks the correctness of the numeric table
     * \param[in] nt              Pointer to the numeric table
     * \param[in] argumentName    Name of checked argument
     * \param[in] requiredRows    Number of required rows
     * \param[in] requiredColumns Number of required columns
     */
    services::SharedPtr<services::Error> checkTable(services::SharedPtr<data_management::NumericTable> nt, const char *argumentName,
                                                    size_t requiredRows, size_t requiredColumns) const
    {
        services::SharedPtr<services::Error> error(new services::Error());
        if(!nt)                                              { error->setId(services::ErrorNullOutputNumericTable); }
        else if(nt->getNumberOfRows()    != requiredRows)    { error->setId(services::ErrorIncorrectNumberOfObservations); }
        else if(nt->getNumberOfColumns() != requiredColumns) { error->setId(services::ErrorInconsistentNumberOfColumns); }
        if(error->id() != services::NoErrorMessageFound)     { error->addStringDetail(services::ArgumentName, argumentName); }
        return error;
    }
};

} // namespace interface1
using interface1::Parameter;
using interface1::Input;
using interface1::Result;

} // namespace sum_of_loss
} // namespace internal
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif
