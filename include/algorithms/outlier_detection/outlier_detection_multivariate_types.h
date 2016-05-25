/* file: outlier_detection_multivariate_types.h */
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
//  Outlier Detection algorithm parameter structure
//--
*/

#ifndef __OUTLIERDETECTION_MULTIVARIATE_TYPES_H__
#define __OUTLIERDETECTION_MULTIVARIATE_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
/**
* \brief Contains classes for computing the multivariate outlier detection
*/
namespace multivariate_outlier_detection
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__MULTIVARIATE_OUTLIER_DETECTION__METHOD"></a>
 * Available computation methods for the multivariate outlier detection algorithm
 */
enum Method
{
    defaultDense = 0,       /*!< Default method */
    baconDense = 1          /*!< Blocked Adaptive Computationally-efficient Outlier Nominators(BACON) method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__MULTIVARIATE_OUTLIER_DETECTION__BACONINITIALIZATIONMETHOD"></a>
 * Available initialization method for the BACON multivariate outlier detection algorithm
 */
enum BaconInitializationMethod
{
    baconMedian = 0,            /*!< Median-based method */
    baconMahalanobis = 1        /*!< Mahalanobis distance-based method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__MULTIVARIATE_OUTLIER_DETECTION__INPUTID"></a>
 * Available identifiers of input objects for the multivariate outlier detection algorithm
 */
enum InputId
{
    data = 0                /*!< %Input data table */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__MULTIVARIATE_OUTLIER_DETECTION__RESULTID"></a>
 * Available identifiers of the results of the multivariate outlier detection algorithm
 */
enum ResultId
{
    weights = 0             /*!< Outlier detection results */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTIVARIATE_OUTLIER_DETECTION__INITIFACE"></a>
 * \brief Abstract interface class that provides function for the initialization procedure
 */
struct InitIface
{
    /**
    * Returns initial parameters for the outlier detection algorithm
    * \param[in] data          Pointer to input objects
    * \param[in] scatter       Measure of spread, the variance-covariance matrix of size p x p
    * \param[in] location      Vector of mean estimates
    * \param[in] threshold     Limit that defines the outlier region, the array of size 1 x 1 containing a non-negative number
    */
    virtual void operator()(data_management::NumericTable *data,
                            data_management::NumericTable *location,
                            data_management::NumericTable *scatter,
                            data_management::NumericTable *threshold) = 0;

    virtual ~InitIface() {}
};

template <Method method>
struct Parameter : public daal::algorithms::Parameter {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTIVARIATE_OUTLIER_DETECTION__PARAMETER"></a>
 * \brief Parameters of the outlier detection computation using the defaultDense method
 *
 * \snippet outlier_detection/outlier_detection_multivariate_types.h ParameterDefault source code
 */
/* [ParameterDefault source code] */
template <> struct Parameter<defaultDense> : public daal::algorithms::Parameter
{
    Parameter() : daal::algorithms::Parameter(), initializationProcedure() {}
    services::SharedPtr<InitIface>
    initializationProcedure;     /*!< Initialization procedure for setting initial parameters of the algorithm */
};
/* [ParameterDefault source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTIVARIATE_OUTLIER_DETECTION__PARAMETER"></a>
 * \brief Parameters of the outlier detection computation using the baconDense method
 *
 * \snippet outlier_detection/outlier_detection_multivariate_types.h ParameterBacon source code
 */
/* [ParameterBacon source code] */
template <> struct Parameter<baconDense> : public daal::algorithms::Parameter
{
    Parameter(BaconInitializationMethod initMethod = baconMedian,
              double alpha = 0.05, double toleranceToConverge = 0.005) :
        initMethod(initMethod), alpha(alpha), toleranceToConverge(toleranceToConverge) {}

    BaconInitializationMethod initMethod;   /*!< Initialization method, \ref BaconInitializationMethod */
    double alpha;                           /*!< One-tailed probability that defines the \f$(1 - \alpha)\f$ quantile
                                                 of the \f$\chi^2\f$ distribution with \f$p\f$ degrees of freedom.
                                                 Recommended value: \f$\alpha / n\f$, where n is the number of observations. */
    double toleranceToConverge;             /*!< Stopping criterion: the algorithm is terminated if the size of the basic subset
                                                 is changed by less than the threshold */
};
/* [ParameterBacon source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTIVARIATE_OUTLIER_DETECTION__INPUT"></a>
 * \brief %Input objects for the multivariate outlier detection algorithm
 */
class Input : public daal::algorithms::Input
{
public:
    Input() : daal::algorithms::Input(1) {}

    virtual ~Input() {}

    /**
     * Returns input object for the multivariate outlier detection algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets input object for the multivariate outlier detection algorithm
     * \param[in] id    Identifier of the %input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks input object for the multivariate outlier detection algorithm
     * \param[in] par     Algorithm parameters
     * \param[in] method  Computation method for the algorithm
          */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        services::SharedPtr<data_management::NumericTable> inTable = get(data);

        if(inTable.get() == 0)                 { this->_errors->add(services::ErrorNullInputNumericTable); return;         }
        if(inTable->getNumberOfRows() == 0)    { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(inTable->getNumberOfColumns() == 0) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;     }

        size_t nFeatures = inTable->getNumberOfColumns();
        size_t nVectors  = inTable->getNumberOfRows();
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTIVARIATE_OUTLIER_DETECTION__RESULT"></a>
 * \brief Results obtained with the compute() method of the multivariate outlier detection algorithm in the %batch processing mode
 */
class Result : public daal::algorithms::Result
{
public:
    Result() : daal::algorithms::Result(1) {}

    virtual ~Result() {};

    /**
     * Allocates memory to store the results of the multivariate outlier detection algorithm
     * \tparam algorithmFPType  Data type to use for storing results, double or float
     * \param[in] input   Pointer to %Input objects of the algorithm
     * \param[in] parameter     Pointer to the parameters of the algorithm
     * \param[in] method  Computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
        size_t nVectors = algInput->get(data)->getNumberOfRows();
        Argument::set(weights, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(1, nVectors, data_management::NumericTable::doAllocate)));
    }

    /**
     * Returns result of the multivariate outlier detection algorithm
     * \param[in] id   Identifier of the result
     * \return         Final result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the result of the multivariate outlier detection algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks the result object of the multivariate outlier detection algorithm
     * \param[in] input   Pointer to %Input objects of the algorithm
     * \param[in] par     Pointer to the parameters of the algorithm
     * \param[in] method  Computation method
          */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
               int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        services::SharedPtr<data_management::NumericTable> resTable = get(weights);

        if(resTable.get() == 0)                 { this->_errors->add(services::ErrorNullInputNumericTable); return;         }
        if(resTable->getNumberOfRows() == 0)    { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(resTable->getNumberOfColumns() == 0) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;     }

        Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));

        if(resTable->getNumberOfRows() != algInput->get(data)->getNumberOfRows() ||
           resTable->getNumberOfColumns() != 1)
        { this->_errors->add(services::ErrorIncorrectSizeOfOutputNumericTable); return; }
    }

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_OUTLIER_DETECTION_MULTIVARIATE_RESULT_ID; }

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
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
} // namespace interface1
using interface1::InitIface;
using interface1::Parameter;
using interface1::Input;
using interface1::Result;

} // namespace multivariate_outlier_detection
} // namespace algorithm
} // namespace daal
#endif
