/* file: em_gmm_init_types.h */
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
//  Implementation of the EM for GMM initialization interface.
//--
*/

#ifndef __EM_GMM_INIT_TYPES_H__
#define __EM_GMM_INIT_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
namespace em_gmm
{
/**
 * \brief Contains classes for the EM for GMM initialization algorithm
 */
namespace init
{
/**
 * <a name="DAAL-ENUM-EM_GMM__INIT__INPUTID"></a>
 * Available identifiers of input objects for the computation of initial values for the EM for GMM algorithm
 */
enum InputId
{
    data = 0               /*!< %Input data table */
};

/**
 * <a name="DAAL-ENUM-EM_GMM__INIT__RESULTID"></a>
 * Available identifiers of results for the computation of initial values for the EM for GMM algorithm
 */
enum ResultId
{
    weights = 0,          /*!< Weights */
    means   = 1           /*!< Means */
};

/**
 * <a name="DAAL-ENUM-EM_GMM__INIT__RESULTCOVARIANCESID"></a>
 * Available identifiers of initialized covariances for the EM for GMM algorithm
 */
enum ResultCovariancesId
{
    covariances = 2       /*!< %Collection of initialized covariances */
};

/**
 * <a name="DAAL-ENUM-EM_GMM__INIT__METHOD"></a>
 * Available methods for the computation of initial values for the EM for GMM algorithm
 */
enum Method
{
    defaultDense = 0       /*!< Default: performance-oriented method. */
};


/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-EM_GMM__INIT__PARAMETER"></a>
 * \brief %Parameter for the computation of initial values for the EM for GMM algorithm
 *
 * \snippet em/em_gmm_init_types.h Parameter source code
 */
/* [Parameter source code] */

struct Parameter : public daal::algorithms::Parameter
{
    Parameter(size_t nComponents, size_t nTrials = 20, size_t nIterations = 10, size_t seed = 777,
              double accuracyThreshold = 1.0e-04) :
        nComponents(nComponents),
        nTrials(nTrials),
        nIterations(nIterations),
        seed(seed),
        accuracyThreshold(accuracyThreshold)
    {}

    virtual ~Parameter() {}

    size_t nComponents;       /*!< Number of components in the Gaussian mixture model */
    size_t nTrials;           /*!< Number of trials of short EM runs */
    size_t nIterations;       /*!< Number of iterations in every short EM run */
    size_t seed;              /*!< Seed for randomly generating data points to start the initialization of short EM */
    double accuracyThreshold; /*!< Threshold for the termination of the algorithm */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-EM_GMM__INIT__DEFAULT_INITILIZATION__INPUT"></a>
 * \brief %Input objects for the computation of initial values for the EM for GMM algorithm
 */
class Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    Input() : daal::algorithms::Input(1)
    {}

    virtual ~Input() {}

    /**
    * Sets the input for the EM for GMM algorithm
    * \param[in] id    Identifier of the input
    * \param[in] ptr   Pointer to the value
    */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
    * Returns the input NumericTable for the computation of initial values for the EM for GMM algorithm
    * \param[in] id    Identifier of the input NumericTable
    * \return          %Input NumericTable that corresponds to the given identifier
    */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, SerializationIface>(Argument::get(id));
    }

    /**
    * Checks input for the computation of initial values for the EM for GMM algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Method of the algorithm
    */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        Parameter *algParameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));
        if(algParameter == 0)                   { this->_errors->add(services::ErrorNullParameterNotSupported); return; }
        if(algParameter->accuracyThreshold < 0) { this->_errors->add(services::ErrorEMInitIncorrectToleranceToConverge); return;  }
        if(algParameter->nIterations <= 0)      { this->_errors->add(services::ErrorEMInitIncorrectDepthNumberIterations); return;  }
        if(algParameter->nTrials <= 0)          { this->_errors->add(services::ErrorEMInitIncorrectNumberOfTrials); return;  }
        if(algParameter->nComponents <= 0)      { this->_errors->add(services::ErrorEMInitIncorrectNumberOfComponents); return;  }

        services::SharedPtr<data_management::NumericTable> inTable = get(data);
        size_t nComponents = algParameter->nComponents;
        size_t nFeatures = inTable->getNumberOfColumns();
        size_t na = Argument::size();

        if(inTable.get() == 0)                 {  this->_errors->add(services::ErrorNullInputNumericTable); return;         }
        if(inTable->getNumberOfRows() == 0)    {  this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(inTable->getNumberOfColumns() == 0) {  this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;     }

        if(inTable->getNumberOfRows() < nComponents) {  this->_errors->add(services::ErrorEMInitInconsistentNumberOfComponents); return; }

        if(na != 1) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return;}

        if(algParameter->accuracyThreshold < 0) {  this->_errors->add(services::ErrorEMIncorrectToleranceToConverge); return; }
    }
};

/**
 * <a name="DAAL-CLASS-EM_GMM__INIT__RESULT"></a>
 * \brief %Results obtained with the compute() method of the initialization of the EM for GMM algorithm in the batch processing mode
 */
class Result: public daal::algorithms::Result
{
public:
    /** Default constructor */
    Result() : daal::algorithms::Result(3)
    {
        Argument::set(covariances, services::SharedPtr<data_management::DataCollection>(new data_management::DataCollection()));
    }

    virtual ~Result() {};

    /**
     * Allocates memory for storing initial values for results of the EM for GMM algorithm
     * \param[in] input        Pointer to the input structure
     * \param[in] parameter    Pointer to the parameter structure
     * \param[in] method       Method of the algorithm
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
        Parameter *algParameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(parameter));

        size_t nFeatures   = algInput->get(data)->getNumberOfColumns();
        size_t nComponents = algParameter->nComponents;

        Argument::set(weights, services::SharedPtr<SerializationIface>(new data_management::HomogenNumericTable<algorithmFPType>(nComponents, 1,
                      data_management::NumericTable::doAllocate, 0)));
        Argument::set(means, services::SharedPtr<SerializationIface>(new data_management::HomogenNumericTable<algorithmFPType>(
                          nFeatures, nComponents, data_management::NumericTable::doAllocate, 0)));

        services::SharedPtr<data_management::DataCollection> covarianceCollection =
            services::SharedPtr<data_management::DataCollection>(new data_management::DataCollection());
        for(size_t i = 0; i < nComponents; i++)
        {
            covarianceCollection->push_back(services::SharedPtr<data_management::NumericTable>(
                                                new data_management::HomogenNumericTable<algorithmFPType>(
                                                    nFeatures, nFeatures, data_management::NumericTable::doAllocate, 0)));
        }
        Argument::set(covariances, services::staticPointerCast<SerializationIface, data_management::DataCollection>(covarianceCollection));
    }

    /**
     * Sets the result for the computation of initial values for the EM for GMM algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the numeric table for the result
     */
    void set(ResultId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Sets the covariance collection for initialization of EM for GMM algorithm
     * \param[in] id    Identifier of the collection of covariances
     * \param[in] ptr   Pointer to the collection of covariances
     */
    void set(ResultCovariancesId id, const services::SharedPtr<data_management::DataCollection> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Returns the result for the computation of initial values for the EM for GMM algorithm
     * \param[in] id   %Result identifier
     * \return         %Result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, SerializationIface>(Argument::get(id));
    }

    /**
     * Returns the collection of initialized covariances for the EM for GMM algorithm
     * \param[in] id   Identifier of the collection of covariances
     * \return         Collection of covariances that corresponds to the given identifier
     */
    services::SharedPtr<data_management::DataCollection> get(ResultCovariancesId id) const
    {
        return services::staticPointerCast<data_management::DataCollection, SerializationIface>(Argument::get(id));
    }

    /**
     * Returns the covariance with a given index from the collection of initialized covariances
     * \param[in] id    Identifier of the collection of covariances
     * \param[in] index Index of the covariance to be returned
     * \return          Pointer to the table with initialized covariances
     */
    services::SharedPtr<data_management::NumericTable> get(ResultCovariancesId id, size_t index) const
    {
        services::SharedPtr<data_management::DataCollection> covCollection = this->get(id);
        return services::staticPointerCast<data_management::NumericTable, SerializationIface>((*covCollection)[index]);
    }

    /**
    * Checks the result of the computation of initial values for the EM for GMM algorithm
    * \param[in] input   %Input of the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Method of the algorithm
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

        if(nFeatures != nInFeatures) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;}

        if(nr != 3) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return;}

        error = checkTable(get(weights), 1, nComponents, "weights");
        if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }

        error = checkTable(get(means), nComponents, nFeatures, "means");
        if(error->id() != services::NoErrorMessageFound) { this->_errors->add(error); return; }

        if(get(covariances).get() == 0) { this->_errors->add(services::ErrorNullOutputDataCollection); return; }
        services::SharedPtr<data_management::DataCollection> resultCovCollection = get(covariances);
        if(resultCovCollection->size() != nComponents) {  this->_errors->add(services::ErrorIncorrectNumberOfElementsInInputCollection); return; }
        for(size_t i = 0; i < nComponents; i++)
        {
            services::SharedPtr<data_management::NumericTable> nt = services::staticPointerCast<data_management::NumericTable, SerializationIface>(
                        (*resultCovCollection)[i]);
            error = checkTable(nt, nFeatures, nFeatures, "covariances");
            if(error->id() != services::NoErrorMessageFound)
            {
                error->addIntDetail(services::ElementInCollection, (int)i);
                this->_errors->add(error);
                return;
            }
        }
    }

    int getSerializationTag() { return SERIALIZATION_EM_GMM_INIT_RESULT_ID; }

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

    /** \private */
    services::SharedPtr<services::Error> checkTable(services::SharedPtr<data_management::NumericTable> nt, size_t requiredRows,
            size_t requiredColumns, const char *argumentName) const
    {
        services::SharedPtr<services::Error> error(new services::Error());
        if(nt.get() == 0)                                    { error->setId(services::ErrorNullOutputNumericTable);         }
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

} // namespace init
} // namespace em_gmm
} // namespace algorithm
} // namespace daal
#endif
