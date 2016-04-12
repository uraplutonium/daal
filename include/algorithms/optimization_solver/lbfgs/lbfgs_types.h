/* file: lbfgs_types.h */
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
//  Implementation of limited memory Broyden-Fletcher-Goldfarb-Shanno
//  algorithm types.
//--
*/
#ifndef __LBFGS_TYPES_H__
#define __LBFGS_TYPES_H__

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
 * \brief Contains classes for computing the limited memory Broyden-Fletcher-Goldfarb-Shanno algorithm
 */
namespace lbfgs
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__LBFGS__METHOD"></a>
 * Available methods for computing LBFGS
 */
enum Method
{
    defaultDense = 0       /*!< Default: performance-oriented method. */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__LBFGS__INPUTID"></a>
 * Available identifiers of input for the LBFGS algorithm
 */
enum InputId
{
    inputArgument = 0 /*!< Initial value to start optimization */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__LBFGS__RESULTID"></a>
 * Available identifiers of results for the LBFGS algorithm
 */
enum ResultId
{
    minimum = 0,    /*!< Numeric table of size 1 x p with the argument */
    nIterations = 1 /*!< Table containing the number of executed iterations */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__LBFGS__PARAMETER"></a>
 * \brief %Parameter class for LBFGS algorithm
 *
 * \snippet optimization_solver/lbfgs/lbfgs_types.h Parameter source code
 */
/* [Parameter source code] */
struct Parameter : public daal::algorithms::Parameter
{
    /**
     * Constructs the parameters of LBFGS algorithm
     * \param[in] function                  Objective function that can be represented as sum
     * \param[in] nIterations               Maximal number of iterations of the algorithm
     * \param[in] accuracyThreshold         Accuracy of the LBFGS algorithm
     * \param[in] batchSize                 Number of observations to compute the stochastic gradient
     * \param[in] correctionPairBatchSize   The number of observations to compute the sub-sampled Hessian for correction pairs computation
     * \param[in] m                         Memory parameter of LBFGS
     * \param[in] L                         The number of iterations between the curvature estimates calculations
     * \param[in] seed                      Seed for random choosing terms from objective function
     */
    Parameter(services::SharedPtr<sum_of_functions::Batch> function = services::SharedPtr<sum_of_functions::Batch>(),
              size_t nIterations = 100, double accuracyThreshold = 1.0e-5,
              size_t batchSize = 10, size_t correctionPairBatchSize = 100,
              size_t m = 10, size_t L = 10, size_t seed = 777) :
        nIterations(nIterations), accuracyThreshold(accuracyThreshold), batchSize(batchSize), correctionPairBatchSize(correctionPairBatchSize),
        m(m), L(L), seed(seed), function(function),
        stepLengthSequence(new data_management::HomogenNumericTable<>(1, 1, data_management::NumericTableIface::doAllocate, 1.0))
    {}

    virtual ~Parameter() {}

    /** Objective function that can be represented as sum */
    services::SharedPtr<sum_of_functions::Batch> function;

    size_t nIterations;             /*!< Maximal number of iterations of the algorithm. */
    double accuracyThreshold;       /*!< Accuracy of the LBFGS algorithm.
                                         The algorithm finishes when this accuracy is achieved. */
    size_t m;                       /*!< Memory parameter of LBFGS.
                                         The maximum number of correction pairs that define the approximation
                                         of inverse Hessian matrix. */
    size_t L;                       /*!< The number of iterations between the curvature estimates calculations */
    size_t seed;                    /*!< Seed for random choosing terms from objective function. */

    size_t batchSize;               /*!< Number of observations to compute the stochastic gradient. */
    /** Numeric table of size nIterations x batchSize that represent indices that will be used instead of random values
        for the stochastic gradient computations. If not set then random indices will be chosen. */
    services::SharedPtr<data_management::NumericTable> batchIndices;

    size_t correctionPairBatchSize; /*!< Number of observations to compute the sub-sampled Hessian for correction pairs computation */
    /** Numeric table of size (nIterations / L) x correctionPairBatchSize that represent indices that will be used
        instead of random values for the sub-sampled Hessian matrix computations. If not set then random indices will be chosen. */
    services::SharedPtr<data_management::NumericTable> correctionPairBatchIndices;
    /** Numeric table of size:
            - 1 x nIterations that contains values of the step-length sequence a(k), for k = 1, ..., nIterations, or
            - 1 x 1           that contains value of step length at each iteration a(1) = ... = a(nIterations) */
    services::SharedPtr<data_management::NumericTable> stepLengthSequence;
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__LBFGS__INPUT"></a>
 * \brief %Input objects for the LBFGS algorithm
 */
class Input :  public daal::algorithms::Input
{
public:
    /**
     * Constructs the structure for storing input objects for the LBFGS algorithm
     */
    Input() : daal::algorithms::Input(1) {}

    virtual ~Input() {}

    /**
     * Returns the number of columns in the numeric table that stores the initaial value
     * to start the optimization
     * \return The number of columns
     */
    virtual size_t getNumberOfColumns() const
    {
        if(get(inputArgument))
        {
            return get(inputArgument)->getNumberOfColumns();
        }
        return 0;
    }

    /**
     * Returns input NumericTable of the LBFGS algorithm
     * \param[in] id    Identifier of the input numeric table
     * \return          %Input numeric table that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets input for the LBFGS algorithm
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
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__LBFGS__RESULT"></a>
 * \brief Results obtained with the compute() method of the LBFGS algorithm in the batch processing mode
 */
class Result : public daal::algorithms::Result
{
public:
    Result() : daal::algorithms::Result(2)
    {}

    virtual ~Result() {};

    /**
     * Allocates memory to store the results of the LBFGS algorithm
     * \param[in] input  Pointer to the input structure
     * \param[in] par    Pointer to the parameter structure
     * \param[in] method Computation method of the algorithm
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
    {
        const Input *algInput = static_cast<const Input *>(input);
        size_t nFeatures = algInput->getNumberOfColumns();
        Argument::set(minimum, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(nFeatures, 1,
                                                                                    data_management::NumericTable::doAllocate)));
        Argument::set(nIterations, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<size_t>(1, 1,
                                                                           data_management::NumericTable::doAllocate)));
    }

    /**
     * Returns result of the LBFGS algorithm
     * \param[in] id   Identifier of the result
     * \return         Final result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the result of the LBFGS algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks the result of the LBFGS algorithm
     * \param[in] input   %Input of algorithm
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method of the algorithm
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
               int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        const Input *algInput = static_cast<const Input *>(input);
        size_t nFeatures = algInput->getNumberOfColumns();

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
    int getSerializationTag() { return SERIALIZATION_LBFGS_RESULT_ID; }

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

} // namespace lbfgs
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif
