/* file: objective_function_types.h */
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
//  Implementation of the Objective function interface.
//--
*/

#ifndef __OBJECTIVE_FUNCTION_TYPES_H__
#define __OBJECTIVE_FUNCTION_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for the Objective function
 */
namespace optimization_solver
{
/**
* \brief Contains classes for computing the Objective function
*/
namespace objective_function
{

/**
  * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__INPUTID"></a>
  * Available identifiers of input objects of the Objective function
  */
enum InputId
{
    argument = 0 /*!< Numeric table of size 1 x p with input argument of the objective function */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__RESULTTOCOMPUTEID"></a>
 * Available identifiers to specify the Objective function result
 */
enum ResultToComputeId
{
    gradient = 0x00000001ULL, /*!< Numeric table of size 1 x p with the gradient of the objective function in the given argument */
    value    = 0x00000002ULL, /*!< Numeric table of size 1 x 1 with the value    of the objective function in the given argument */
    hessian  = 0x00000004ULL  /*!< Numeric table of size p x p with the hessian  of the objective function in the given argument */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__RESULTID"></a>
 * Available identifiers of results of the Objective function
 */
enum ResultId
{
    resultCollection = 0 /*!< Collection of the pointers to the objective function results. */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__RESULTCOLLECTIONID"></a>
 * Available identifiers of results of the Objective function
 */
enum ResultCollectionId
{
    gradientIdx = 0, /*!< Index of the gradient numeric table in the result collection */
    valueIdx = 1,    /*!< Index of the value numeric table in the result collection */
    hessianIdx = 2   /*!< Index of the hessian numeric table in the result collection */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__PARAMETER"></a>
 * \brief %Parameter for the Objective function
 *
 * \snippet optimization_solver/objective_function/objective_function_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /**
     * Constructs the parameter of the Objective function
     * \param[in] resultsToCompute 64 bit integer flag that indicates the results to compute
     */
    Parameter(const DAAL_UINT64 resultsToCompute = gradient);

    /**
     * Constructs an Parameter by copying input objects and parameters of another Parameter
     * \param[in] other An object to be used as the source to initialize object
     */
    Parameter(const Parameter &other);

    virtual ~Parameter() {}

    DAAL_UINT64 resultsToCompute;  /*!< 64 bit integer flag that indicates the results to compute */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__INPUT"></a>
 * \brief %Input objects for the Objective function
 */
class Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    Input(size_t n = 1) : daal::algorithms::Input(n)
    {}

    /** Destructor */
    virtual ~Input() {}

    /**
     * Sets one input object for Objective function
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Returns the input numeric table for Objective function
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
        services::SharedPtr<services::Error> error(new services::Error());
        const Parameter *algParameter = static_cast<const Parameter *>(par);

        if(algParameter == 0)
        { this->_errors->add(services::ErrorNullParameterNotSupported); return; }

        error = checkTable(get(argument), "argument", 1);
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
class Result: public daal::algorithms::Result
{
public:
    /** Default constructor */
    Result() : daal::algorithms::Result(1)
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

        size_t nCols = algInput->get(argument)->getNumberOfColumns();

        SharedPtr<DataCollection> collection = SharedPtr<DataCollection>(new DataCollection(3));

        if(algParameter->resultsToCompute & gradient)
        {
            (*collection)[(int)gradientIdx] =
                SharedPtr<SerializationIface>(new HomogenNumericTable<algorithmFPType>(nCols, 1, NumericTable::doAllocate, 0));
        }
        if(algParameter->resultsToCompute & value)
        {
            (*collection)[(int)valueIdx] =
                SharedPtr<SerializationIface>(new HomogenNumericTable<algorithmFPType>(1, 1, NumericTable::doAllocate, 0));
        }
        if(algParameter->resultsToCompute & hessian)
        {
            (*collection)[(int)hessianIdx] =
                SharedPtr<SerializationIface>(new HomogenNumericTable<algorithmFPType>(nCols, nCols, NumericTable::doAllocate, 0));
        }

        Argument::set(resultCollection, staticPointerCast<DataCollection, SerializationIface>(collection));
    }


    /**
     * Sets the result of the Objective function
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the numeric table with the result
     */
    void set(ResultId id, const services::SharedPtr<data_management::DataCollection> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Returns the collection of the results of the Objective function
     * \param[in] id   Identifier of the result
     * \return         %Result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::DataCollection> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Returns the result numeric table from the Objective function recult collection by index
     * \param[in] id   Identifier of the result
     * \param[in] idx  Indentifier of index in result collection
     * \return         %Result that corresponds to the given identifiers
     */
    services::SharedPtr<data_management::NumericTable> get(ResultId id, ResultCollectionId idx) const
    {
        services::SharedPtr<data_management::DataCollection> collection = this->get(id);
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>((*collection)[(int)idx]);
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

        size_t nFeatures = algInput->get(argument)->getNumberOfColumns();

        if(algParameter->resultsToCompute & value)
        {
            error = checkTable(get(resultCollection, valueIdx), "value", 1, 1);
            if(error->id() != NoErrorMessageFound) { this->_errors->add(error); return; }
        }
        if(algParameter->resultsToCompute & gradient)
        {
            error = checkTable(get(resultCollection, gradientIdx), "gradient", 1, nFeatures);
            if(error->id() != NoErrorMessageFound) { this->_errors->add(error); return; }
        }
        if(algParameter->resultsToCompute & hessian)
        {
            error = checkTable(get(resultCollection, hessianIdx), "hessian", nFeatures, nFeatures);
            if(error->id() != NoErrorMessageFound) { this->_errors->add(error); return; }
        }
    }

    /**
     * Returns a serialization tag, a unique identifier of this class used in serialization
     * \return Serialization tag
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_OBJECTIVE_FUNCTION_RESULT_ID; }

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

} // namespace objective_function
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif
