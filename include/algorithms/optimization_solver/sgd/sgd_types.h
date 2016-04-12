/* file: sgd_types.h */
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
//  Implementation of the Stochastic gradient descent algorithm types.
//--
*/

#ifndef __SGD_TYPES_H__
#define __SGD_TYPES_H__

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
 * \brief Contains classes for computing the Stochastic gradient descent
 */
namespace sgd
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__METHOD"></a>
 * Available methods for computing the Stochastic gradient descent
 */
enum Method
{
    defaultDense = 0, /*!< Default: Required gradient is computed using only one term of objective function */
    miniBatch = 1     /*!< Required gradient is computed using batchSize terms of objective function  */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__INPUTID"></a>
 * Available identifiers of input for the Stochastic gradient descent
 */
enum InputId
{
    inputArgument = 0 /*!< Initial value to start optimization */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__RESULTID"></a>
 * Available identifiers of results for the Stochastic gradient descent algorithm
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
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__BASEPARAMETER"></a>
 * \brief %BaseParameter base class for the Stochastic gradient descent algorithm
 *
 * \snippet optimization_solver/sgd/sgd_types.h BaseParameter source code
 */
/* [BaseParameter source code] */
struct BaseParameter : public daal::algorithms::Parameter
{
    /**
     * Constructs the parameter base class of the Stochastic gradient descent algorithm
     * \param[in] function             Objective function represented as sum of functions
     * \param[in] nIterations          Maximal number of iterations of the algorithm
     * \param[in] accuracyThreshold    Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
     * \param[in] batchIndices         Numeric table that represents 32 bit integer indices of terms in the objective function.
     *                                 If no indices are provided, the implementation will generate random indices.
     * \param[in] learningRateSequence Numeric table that contains values of the learning rate sequence
     * \param[in] seed                 Seed for random generation of 32 bit integer indices of terms in the objective function.
     */
    BaseParameter(
        services::SharedPtr<sum_of_functions::Batch>       function,
        const size_t                                       nIterations = 100,
        const double                                       accuracyThreshold = 1.0e-05,
        services::SharedPtr<data_management::NumericTable> batchIndices = services::SharedPtr<data_management::NumericTable>(),
        services::SharedPtr<data_management::NumericTable> learningRateSequence = services::SharedPtr<data_management::NumericTable>(
                                                                                  new data_management::HomogenNumericTable<double>(
                                                                                  1, 1, data_management::NumericTableIface::doAllocate, 1.0)),
        const size_t                                       seed = 777
    ) :
        function(function),
        nIterations(nIterations),
        accuracyThreshold(accuracyThreshold),
        batchIndices(batchIndices),
        learningRateSequence(learningRateSequence),
        seed(seed)
    {}

    /**
    * Constructs an BaseParameter by copying input objects and parameters of another BaseParameter
    * \param[in] other An object to be used as the source to initialize object
    */
    BaseParameter(const BaseParameter &other) :
        function(other.function),
        nIterations(other.nIterations),
        accuracyThreshold(other.accuracyThreshold),
        batchIndices(other.batchIndices),
        learningRateSequence(other.learningRateSequence),
        seed(other.seed)
    {}

    virtual ~BaseParameter() {}

    /**
     * Checks the correctness of the parameter
     */
    virtual void check() const
    {
        if(learningRateSequence.get() != NULL)
        {
            services::SharedPtr<services::Error> error(new services::Error());
            if(learningRateSequence->getNumberOfRows() != nIterations && learningRateSequence->getNumberOfRows() != 1)
            {
                error->setId(services::ErrorIncorrectNumberOfObservations);
            }
            if(learningRateSequence->getNumberOfColumns() != 1)
            {
                error->setId(services::ErrorIncorrectNumberOfFeatures);
            }
            if(error->id() != services::NoErrorMessageFound)
            {
                error->addStringDetail(services::ArgumentName, "learningRateSequence");
                this->_errors->add(error);
            }
        }

        if(function.get() == NULL)
        {
            services::SharedPtr<services::Error> error(new services::Error());
            error->setId(services::ErrorIncorrectParameter);
            error->addStringDetail(services::ArgumentName, "function");
            this->_errors->add(error);
        }

        if(accuracyThreshold < 0)
        {
            services::SharedPtr<services::Error> error(new services::Error());
            error->setId(services::ErrorIncorrectParameter);
            error->addStringDetail(services::ArgumentName, "accuracyThreshold");
            this->_errors->add(error);
        }

        if(function->sumOfFunctionsParameter == NULL)
        {
            services::SharedPtr<services::Error> error(new services::Error());
            error->setId(services::ErrorNullParameterNotSupported);
            error->addStringDetail(services::ArgumentName, "sumOfFunctionsParameter");
            this->_errors->add(error);
        }
        if(function->sumOfFunctionsInput == NULL)
        {
            services::SharedPtr<services::Error> error(new services::Error());
            error->setId(services::ErrorNullInput);
            error->addStringDetail(services::ArgumentName, "sumOfFunctionsInput");
            this->_errors->add(error);
        }
    }

    services::SharedPtr<sum_of_functions::Batch>       function;             /*!< Objective function represented as sum of functions */
    size_t                                             nIterations;          /*!< Maximal number of iterations of the algorithm */
    double                                             accuracyThreshold;    /*!< Accuracy of the algorithm. The algorithm terminates when
                                                                                  this accuracy is achieved */
    services::SharedPtr<data_management::NumericTable> batchIndices;         /*!< Numeric table that represents 32 bit integer indices of terms
                                                                                  in the objective function. If no indices are provided,
                                                                                  the implementation will generate random indices. */
    services::SharedPtr<data_management::NumericTable> learningRateSequence; /*!< Numeric table that contains values of the learning rate sequence */
    size_t                                             seed;                 /*!< Seed for random generation of 32 bit integer indices of terms
                                                                                  in the objective function. */

};
/* [BaseParameter source code] */

template<Method method>
struct Parameter : public BaseParameter {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__PARAMETER"></a>
 * \brief %Parameter for the Stochastic gradient descent algorithm
 *
 * \snippet optimization_solver/sgd/sgd_types.h ParameterDefaultDense source code
 */
/* [ParameterDefaultDense source code] */
template<>
struct Parameter<defaultDense> : public BaseParameter
{
    /**
     * \param[in] function             Objective function represented as sum of functions
     * \param[in] nIterations          Maximal number of iterations of the algorithm
     * \param[in] accuracyThreshold    Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
     * \param[in] batchIndices         Numeric table that represents 32 bit integer indices of terms in the objective function. If no indices are
                                       provided, the implementation will generate random indices.
     * \param[in] learningRateSequence Numeric table that contains values of the learning rate sequence
     * \param[in] seed                 Seed for random generation of 32 bit integer indices of terms in the objective function.
     */
    Parameter(
        services::SharedPtr<sum_of_functions::Batch>       function,
        const size_t                                       nIterations = 100,
        const double                                       accuracyThreshold = 1.0e-05,
        services::SharedPtr<data_management::NumericTable> batchIndices = services::SharedPtr<data_management::NumericTable>(),
        services::SharedPtr<data_management::NumericTable> learningRateSequence = services::SharedPtr<data_management::NumericTable>(
                                                                                  new data_management::HomogenNumericTable<double>(
                                                                                  1, 1, data_management::NumericTableIface::doAllocate, 1.0)),
        const size_t                                       seed = 777
    ) :
        BaseParameter(
            function,
            nIterations,
            accuracyThreshold,
            batchIndices,
            learningRateSequence,
            seed
        )
    {}

    /**
    * Constructs an Parameter by copying input objects and parameters of another Parameter
    * \param[in] other An object to be used as the source to initialize object
    */
    Parameter(const Parameter &other) :
        BaseParameter(
            other.function,
            other.nIterations,
            other.accuracyThreshold,
            other.batchIndices,
            other.learningRateSequence,
            other.seed
        )
    {}

    /**
     * Checks the correctness of the parameter
     */
    virtual void check() const
    {
        BaseParameter::check();
        if(batchIndices.get() != NULL)
        {
            services::SharedPtr<services::Error> error(new services::Error());
            if(batchIndices->getNumberOfRows() != nIterations)    { error->setId(services::ErrorIncorrectNumberOfObservations); }
            if(batchIndices->getNumberOfColumns() != 1)           { error->setId(services::ErrorIncorrectNumberOfFeatures);     }
            if(error->id() != services::NoErrorMessageFound)
            {
                error->addStringDetail(services::ArgumentName, "batchIndices");
                this->_errors->add(error);
            }
            return;
        }
    }

    virtual ~Parameter() {}
};
/* [ParameterDefaultDense source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__PARAMETER"></a>
 * \brief %Parameter for the Stochastic gradient descent algorithm
 *
 * \snippet optimization_solver/sgd/sgd_types.h ParameteMiniBatch source code
 */
/* [ParameteMiniBatch source code] */
template<>
struct Parameter<miniBatch> : public BaseParameter
{
    /**
     * Constructs the parameter class of the Stochastic gradient descent algorithm
     * \param[in] function             Objective function represented as sum of functions
     * \param[in] nIterations          Maximal number of iterations of the algorithm
     * \param[in] accuracyThreshold    Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
     * \param[in] batchIndices         Numeric table that represents 32 bit integer indices of terms in the objective function. If no indices
                                       are provided, the implementation will generate random indices.
     * \param[in] batchSize            Number of batch indices to compute the stochastic gradient. If batchSize is equal to the number of terms
                                       in objective function then no random sampling is performed, and all terms are used to calculate the gradient.
                                       This parameter is ignored if batchIndices is provided.
     * \param[in] conservativeSequence Numeric table of values of the conservative coefficient sequence
     * \param[in] innerNIterations     Number of inner iterations
     * \param[in] learningRateSequence Numeric table that contains values of the learning rate sequence
     * \param[in] seed                 Seed for random generation of 32 bit integer indices of terms in the objective function.
     */
    Parameter(
        services::SharedPtr<sum_of_functions::Batch>       function,
        const size_t                                       nIterations = 100,
        const double                                       accuracyThreshold = 1.0e-05,
        services::SharedPtr<data_management::NumericTable> batchIndices = services::SharedPtr<data_management::NumericTable>(),
        const size_t                                       batchSize = 128,
        services::SharedPtr<data_management::NumericTable> conservativeSequence = services::SharedPtr<data_management::NumericTable>(
                                                                                  new data_management::HomogenNumericTable<double>(
                                                                                  1, 1, data_management::NumericTableIface::doAllocate, 1.0)),
        const size_t                                       innerNIterations = 5,
        services::SharedPtr<data_management::NumericTable> learningRateSequence = services::SharedPtr<data_management::NumericTable>(
                                                                                  new data_management::HomogenNumericTable<double>(
                                                                                  1, 1, data_management::NumericTableIface::doAllocate, 1.0)),
        const size_t                                       seed = 777
    ) :
        BaseParameter(
            function,
            nIterations,
            accuracyThreshold,
            batchIndices,
            learningRateSequence,
            seed
        ),
        batchSize(batchSize),
        conservativeSequence(conservativeSequence),
        innerNIterations(innerNIterations)
    {}

    /**
    * Constructs an Parameter by copying input objects and parameters of another Parameter
    * \param[in] other An object to be used as the source to initialize object
    */
    Parameter(const Parameter &other) :
        BaseParameter(
            other.function,
            other.nIterations,
            other.accuracyThreshold,
            other.batchIndices,
            other.learningRateSequence,
            other.seed
        ),
        batchSize(other.batchSize),
        conservativeSequence(other.conservativeSequence),
        innerNIterations(other.innerNIterations)
    {}

    /**
     * Checks the correctness of the parameter
     */
    virtual void check() const
    {
        BaseParameter::check();
        if(batchIndices.get() != NULL)
        {
            services::SharedPtr<services::Error> error(new services::Error());
            if(batchIndices->getNumberOfRows() != nIterations) {error->setId(services::ErrorIncorrectNumberOfObservations);}
            if(batchIndices->getNumberOfColumns() != batchSize) {error->setId(services::ErrorIncorrectNumberOfFeatures);}
            if(error->id() != services::NoErrorMessageFound)
            {
                error->addStringDetail(services::ArgumentName, "batchIndices");
                this->_errors->add(error);
            }
            return;
        }

        if(conservativeSequence.get() != NULL)
        {
            services::SharedPtr<services::Error> error(new services::Error());
            if(conservativeSequence->getNumberOfRows() != nIterations && conservativeSequence->getNumberOfRows() != 1)
            {
                error->setId(services::ErrorIncorrectNumberOfObservations);
            }
            if(conservativeSequence->getNumberOfColumns() != 1)
            {
                error->setId(services::ErrorIncorrectNumberOfFeatures);
            }
            if(error->id() != services::NoErrorMessageFound)
            {
                error->addStringDetail(services::ArgumentName, "conservativeSequence");
                this->_errors->add(error);
            }
        }

        if(batchSize > function->sumOfFunctionsParameter->numberOfTerms)
        {
            services::SharedPtr<services::Error> error(new services::Error());
            error->setId(services::ErrorIncorrectParameter);
            error->addStringDetail(services::ArgumentName, "batchSize");
            this->_errors->add(error);
        }
    }

    virtual ~Parameter() {}

    size_t                                             batchSize;            /*!< Number of batch indices to compute the stochastic gradient.
                                                                                  If batchSize is equal to the number of terms in objective
                                                                                  function then no random sampling is performed, and all terms are
                                                                                  used to calculate the gradient. This parameter is ignored
                                                                                  if batchIndices is provided. */
    services::SharedPtr<data_management::NumericTable> conservativeSequence; /*!< Numeric table of values of the conservative coefficient sequence */
    size_t                                             innerNIterations;
};
/* [ParameteMiniBatch source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__INPUT"></a>
 * \brief %Input parameters for the Stochastic gradient descent algorithm
 */
class Input :  public daal::algorithms::Input
{
public:
    Input() : daal::algorithms::Input(1) {}

    virtual ~Input() {}

    virtual size_t getNumberOfColumns() const
    {
        if(get(inputArgument))
        {
            return get(inputArgument)->getNumberOfColumns();
        }
        return 0;
    }

    /**
     * Returns input NumericTable of the Stochastic gradient descent algorithm
     * \param[in] id    Identifier of the input numeric table
     * \return          %Input numeric table that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets input for the Stochastic gradient descent algorithm
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
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__RESULT"></a>
 * \brief Results obtained with the compute() method of the Stochastic gradient descent algorithm in the batch processing mode
 */
class Result : public daal::algorithms::Result
{
public:
    Result() : daal::algorithms::Result(2)
    {}

    virtual ~Result() {};

    /**
     * Allocates memory to store the results of the Stochastic gradient descent algorithm
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
                                  data_management::NumericTable::doAllocate, 0.0)));
        Argument::set(nIterations, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<size_t>(1, 1,
                                  data_management::NumericTable::doAllocate, (size_t)0)));
    }

    /**
     * Returns result of the Stochastic gradient descent algorithm
     * \param[in] id   Identifier of the result
     * \return         Final result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the result of the Stochastic gradient descent algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks the result of the Stochastic gradient descent algorithm
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
    int getSerializationTag() { return SERIALIZATION_SGD_RESULT_ID; }

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
using interface1::BaseParameter;
using interface1::Parameter;
using interface1::Input;
using interface1::Result;

} // namespace sgd
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif
