/* file: em_gmm_types.h */
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
//  Implementation of the EM for GMM algorithm interface.
//--
*/

#ifndef __EM_GMM_TYPES_H__
#define __EM_GMM_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/covariance/covariance_batch.h"
#include "em_gmm_init_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for the EM for GMM algorithm
 */
namespace em_gmm
{
/**
 * <a name="DAAL-ENUM-EM_GMM__METHOD"></a>
 * Available methods for computing results of the EM for GMM algorithm
 */
enum Method
{
    defaultDense = 0       /*!< Default: performance-oriented method. */
};

/**
 * <a name="DAAL-ENUM-EM_GMM__INPUTID"></a>
 * Available identifiers of input objects of the EM for GMM algorithm
 */
enum InputId
{
    data         = 0,           /*!< %Input data table */
    inputWeights = 1,           /*!< Input weights */
    inputMeans   = 2            /*!< Input means */
};

/**
 * <a name="DAAL-ENUM-EM_GMM__INPUTCOVARIANCESID"></a>
 * Available identifiers of input covariances for the EM for GMM algorithm
 */
enum InputCovariancesId
{
    inputCovariances = 3       /*!< %Collection of input covariances */
};

/**
 * <a name="DAAL-ENUM-EM_GMM__INPUTVALUESID"></a>
 * Available identifiers of input values for the EM for GMM algorithm
 */
enum InputValuesId
{
    inputValues = 4             /*!< Input objects of the EM for GMM algorithm */
};

/**
 * <a name="DAAL-ENUM-EM_GMM__RESULTID"></a>
 * Available identifiers of results (means or weights) of the EM for GMM algorithm
 */
enum ResultId
{
    weights = 0,                /*!< Weights */
    means   = 1,                /*!< Means */
    goalFunction = 3,           /*!< Table containing log-likelyhood value */
    nIterations  = 4            /*!< Table containing the number of executed iterations */
};

/**
 * <a name="DAAL-ENUM-EM_GMM__RESULTCOVARIANCESID"></a>
 * Available identifiers of computed covariances for the EM for GMM algorithm
 */
enum ResultCovariancesId
{
    covariances = 2             /*!< %Collection of covariances */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-EM_GMM__PARAMETER"></a>
 * \brief %Parameter for the EM for GMM algorithm
 *
 * \snippet em/em_gmm_types.h Parameter source code
 */
/* [Parameter source code] */
struct Parameter : public daal::algorithms::Parameter
{
    /**
     * Constructs the parameter of EMM for GMM algorithm
     * \param[in] nComponents              Number of components in the Gaussian mixture model
     * \param[in] maxIterations            Maximal number of iterations of the algorithm
     * \param[in] accuracyThreshold        Threshold for the termination of the algorithm
     * \param[in] covariance               Pointer to the algorithm that computes the covariance
     */
    Parameter(const size_t nComponents,
              const services::SharedPtr<covariance::BatchIface> &covariance,
              const size_t maxIterations = 10,
              const double accuracyThreshold = 1.0e-04) :
        nComponents(nComponents),
        maxIterations(maxIterations),
        accuracyThreshold(accuracyThreshold),
        covariance(covariance)
    {}

    Parameter(const Parameter &other) :
        nComponents(other.nComponents),
        maxIterations(other.maxIterations),
        accuracyThreshold(other.accuracyThreshold),
        covariance(other.covariance)
    {}

    virtual ~Parameter() {}

    size_t nComponents;                           /*!< Number of components in the Gaussian mixture model */
    size_t maxIterations;                         /*!< Maximal number of iterations of the algorithm. */
    double accuracyThreshold;                     /*!< Threshold for the termination of the algorithm.    */
    services::SharedPtr<covariance::BatchIface> covariance; /*!< Pointer to the algorithm that computes the covariance */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-EM_GMM__INPUT"></a>
 * \brief %Input objects for the EM for GMM algorithm
 */
class Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    Input() : daal::algorithms::Input(4)
    {}

    virtual ~Input() {}

    /**
     * Sets one input object for the EM for GMM algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Sets the input covariance object for the EM for GMM algorithm
     * \param[in] id    Identifier of the input covariance collection object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputCovariancesId id, const services::SharedPtr<data_management::DataCollection> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Sets input objects for the EM for GMM algorithm[IE1]
     * \param[in] id    Identifier of the input values object. Result of the EM for GMM initialization algorithm can be used.
     * \param[in] ptr   Pointer to the object
     */
    void set(InputValuesId id, const services::SharedPtr<init::Result> &ptr)
    {
        set(inputWeights,     ptr->get(init::weights));
        set(inputMeans,       ptr->get(init::means));
        set(inputCovariances, ptr->get(init::covariances));
    }


    /**
     * Returns the input numeric table for the EM for GMM algorithm
     * \param[in] id    Identifier of the input numeric table
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Returns the collection of input covariances for the EM for GMM algorithm
     * \param[in] id    Identifier of the  collection of input covariances
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::DataCollection> get(InputCovariancesId id) const
    {
        return services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Returns a covariance with a given index from the collection of input covariances
     * \param[in] id    Identifier of the collection of input covariances
     * \param[in] index Index of the covariance to be returned
     * \return          Pointer to the table with the input covariance
     */
    services::SharedPtr<data_management::NumericTable> get(InputCovariancesId id, size_t index) const
    {
        services::SharedPtr<data_management::DataCollection> covCollection = this->get(id);
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>((*covCollection)[index]);
    }


    /**
     * Checks the correctness of the input result
     * \param[in] par       Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        services::SharedPtr<services::Error> error(new services::Error());
        Parameter *algParameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));

        if(algParameter == 0)                   { this->_errors->add(services::ErrorNullParameterNotSupported); return; }
        if(algParameter->accuracyThreshold < 0) { this->_errors->add(services::ErrorEMIncorrectToleranceToConverge); return; }
        if(algParameter->maxIterations <= 0)    { this->_errors->add(services::ErrorEMIncorrectMaxNumberOfIterations); return; }
        if(algParameter->nComponents <= 0)      { this->_errors->add(services::ErrorEMIncorrectNumberOfComponents); return; }

        size_t nComponents = algParameter->nComponents;
        if(this->size() != 4) {this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        error = checkTable(get(data), "data");
        if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }

        size_t nFeatures = get(data)->getNumberOfColumns();

        error = checkTable(get(inputWeights), "inputWeights", 1, nComponents);
        if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }

        error = checkTable(get(inputMeans), "inputMeans", nComponents, nFeatures);
        if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }

        services::SharedPtr<data_management::DataCollection> initCovCollection = get(inputCovariances);
        if(!initCovCollection)
        { this->_errors->add(services::ErrorNullInputDataCollection); return; }

        if(initCovCollection->size() != nComponents)
        { this->_errors->add(services::ErrorIncorrectNumberOfElementsInInputCollection); return; }

        for(size_t i = 0; i < nComponents; i++)
        {
            services::SharedPtr<data_management::NumericTable> nt =
                services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>((
                            *initCovCollection)[i]);
            error = checkTable(nt, "inputCovariances", nFeatures, nFeatures);
            if(error->id() != services::NoErrorMessageFound)
            {
                error->addIntDetail(services::ElementInCollection, (int)i);
                this->_errors->add(error);
                return;
            }
        }
    }

protected:
    services::SharedPtr<services::Error> checkTable(services::SharedPtr<data_management::NumericTable> nt, const char* argumentName,
            size_t requiredRows = 0, size_t requiredColumns = 0) const
    {
        services::SharedPtr<services::Error> error(new services::Error());
        if(nt.get() == 0)  { error->setId(services::ErrorNullInputNumericTable); }
        else if(nt->getNumberOfRows() == 0)     { error->setId(services::ErrorEmptyInputNumericTable); }
        else if(nt->getNumberOfColumns() == 0)  { error->setId(services::ErrorEmptyInputNumericTable); }
        else if(requiredRows != 0    && nt->getNumberOfRows()    != requiredRows)    { error->setId(services::ErrorIncorrectNumberOfObservations);  }
        else if(requiredColumns != 0 && nt->getNumberOfColumns() != requiredColumns) { error->setId(services::ErrorInconsistentNumberOfColumns);    }
        if(error->id() != services::NoErrorMessageFound)                             { error->addStringDetail(services::ArgumentName, argumentName);}
        return error;
    }
};

/**
 * <a name="DAAL-CLASS-EM_GMM__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the EM for GMM algorithm in the batch processing mode
 */
class Result: public daal::algorithms::Result
{
public:
    /** Default constructor */
    Result() : daal::algorithms::Result(5)
    {
        Argument::set(covariances, services::SharedPtr<data_management::DataCollection>(new data_management::DataCollection()));
    }

    virtual ~Result() {};

    /**
     * Allocates memory for storing results of the EM for GMM algorithm
     * \param[in] input     Pointer to the input structure
     * \param[in] parameter Pointer to the parameter structure
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);

        size_t nFeatures   = algInput->get(data)->getNumberOfColumns();
        size_t nComponents = algParameter->nComponents;

        Argument::set(weights, services::SharedPtr<data_management::SerializationIface>(new data_management::HomogenNumericTable<algorithmFPType>
                      (nComponents, 1,
                       data_management::NumericTable::doAllocate, 0)));
        Argument::set(means, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(
                              nFeatures, nComponents, data_management::NumericTable::doAllocate, 0)));

        services::SharedPtr<data_management::DataCollection> covarianceCollection =
            services::SharedPtr<data_management::DataCollection>(new data_management::DataCollection());
        for(size_t i = 0; i < nComponents; i++)
        {
            covarianceCollection->push_back(services::SharedPtr<data_management::NumericTable>(
                                                new data_management::HomogenNumericTable<algorithmFPType>(
                                                    nFeatures, nFeatures, data_management::NumericTable::doAllocate, 0)));
        }
        Argument::set(covariances, services::staticPointerCast<data_management::SerializationIface, data_management::DataCollection>
                      (covarianceCollection));

        Argument::set(goalFunction, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(
                              1, 1, data_management::NumericTable::doAllocate, 0)));

        Argument::set(nIterations, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<int>(
                              1, 1, data_management::NumericTable::doAllocate, 0)));
    }

    /**
     * Sets the result of the EM for GMM algorithm
     * \param[in] id    %Result identifier
     * \param[in] ptr   Pointer to the numeric table with the result
     */
    void set(ResultId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Sets the collection of covariances for the EM for GMM algorithm
     * \param[in] id    Identifier of the collection of covariances
     * \param[in] ptr   Pointer to the collection of covariances
     */
    void set(ResultCovariancesId id, const services::SharedPtr<data_management::DataCollection> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Returns the result of the EM for GMM algorithm
     * \param[in] id   %Result identifier
     * \return         %Result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Returns the collection of computed covariances of the EM for GMM algorithm
     * \param[in] id   Identifier of the collection of computed covariances
     * \return         Collection of computed covariances that corresponds to the given identifier
     */
    services::SharedPtr<data_management::DataCollection> get(ResultCovariancesId id) const
    {
        return services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Returns the covariance with a given index from the collection of computed covariances
     * \param[in] id    Identifier of the collection of covariances
     * \param[in] index Index of the covariance to be returned
     * \return          Pointer to the table with the computed covariance
     */
    services::SharedPtr<data_management::NumericTable> get(ResultCovariancesId id, size_t index) const
    {
        services::SharedPtr<data_management::DataCollection> covCollection = this->get(id);
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>((*covCollection)[index]);
    }

    /**
    * Checks the result parameter of the EM for GMM algorithm
    * \param[in] input   %Input of the algorithm
    * \param[in] par     %Parameter of algorithm
    * \param[in] method  Computation method
    */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
               int method) const DAAL_C11_OVERRIDE
    {
        services::SharedPtr<services::Error> error(new services::Error());
        Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));

        Parameter *algParameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));

        size_t nComponents = algParameter->nComponents;
        size_t nFeatures = get(means)->getNumberOfColumns();
        size_t nInFeatures = algInput->get(data)->getNumberOfColumns();
        size_t nr = Argument::size();

        if(nFeatures != nInFeatures) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }

        if(nr != 5) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        error = checkTable(get(weights), 1, nComponents, "weights");
        if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }

        error = checkTable(get(means), nComponents, nFeatures, "means");
        if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }

        if(get(covariances).get() == 0) { this->_errors->add(services::ErrorNullOutputDataCollection); return; }
        services::SharedPtr<data_management::DataCollection> resultCovCollection = get(covariances);
        if(resultCovCollection->size() != nComponents) { this->_errors->add(services::ErrorIncorrectNumberOfElementsInInputCollection); return; }
        for(size_t i = 0; i < nComponents; i++)
        {
            services::SharedPtr<data_management::NumericTable> nt =
                services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>((*resultCovCollection)[i]);
            error = checkTable(nt, nFeatures, nFeatures, "covariances");
            if(error->id() != services::NoErrorMessageFound)
            {
                error->addIntDetail(services::ElementInCollection, (int)i);
                this->_errors->add(error);
                return;
            }
        }

        error = checkTable(get(goalFunction), 1, 1, "goalFunction");
        if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }

        error = checkTable(get(nIterations), 1, 1, "nIterations");
        if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }
    }

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_EM_GMM_RESULT_ID; }

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

    services::SharedPtr<services::Error> checkTable(services::SharedPtr<data_management::NumericTable> nt,
            size_t requiredRows, size_t requiredColumns, const char* argumentName) const
    {
        services::SharedPtr<services::Error> error(new services::Error());
        if(!nt)                                              { error->setId(services::ErrorNullOutputNumericTable);         }
        else if(nt->getNumberOfRows()    != requiredRows)    { error->setId(services::ErrorIncorrectNumberOfObservations);  }
        else if(nt->getNumberOfColumns() != requiredColumns) { error->setId(services::ErrorInconsistentNumberOfColumns);    }
        if(error->id() != services::NoErrorMessageFound)     { error->addStringDetail(services::ArgumentName, argumentName);}
        return error;
    }
};
} // namespace interface1
using interface1::Parameter;
using interface1::Input;
using interface1::Result;

} // namespace em_gmm
} // namespace algorithm
} // namespace daal
#endif
