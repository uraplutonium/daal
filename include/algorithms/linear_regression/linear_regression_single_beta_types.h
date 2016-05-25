/* file: linear_regression_single_beta_types.h */
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
//  Interface for the linear regression algorithm quality metrics for a single beta coefficient
//--
*/

#ifndef __LINEAR_REGRESSION_SINGLE_BETA_TYPES_H__
#define __LINEAR_REGRESSION_SINGLE_BETA_TYPES_H__

#include "services/daal_shared_ptr.h"
#include "algorithms/algorithm.h"
#include "algorithms/linear_regression/linear_regression_model.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace quality_metric
{
namespace single_beta
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__LINEAR_REGRESSION__SINGLE_BETA__METHOD"></a>
 * Available methods for computing the quality metrics for a single beta coefficient
 */
enum Method
{
    defaultDense = 0    /*!< Default method */
};

/**
* <a name="ALGORITHMS__LINEAR_REGRESSION__SINGLE_BETA__DATA_INPUT_ID"></a>
* \brief Available identifiers of input objects for a single beta quality metrics
*/
enum DataInputId
{
    expectedResponses = 0,   /*!< NumericTable n x k. Expected responses (Y), dependent variables */
    predictedResponses = 1   /*!< NumericTable n x k. Predicted responses (Z) */
};

/**
* <a name="ALGORITHMS__LINEAR_REGRESSION__SINGLE_BETA__MODEL_ID"></a>
* \brief Available identifiers of input objects for single beta quality metrics
*/
enum ModelInputId
{
    model = 2                /*!< Linear regression model */
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__LINEAR_REGRESSION__SINGLE_BETA__RESULT_ID"></a>
* \brief Available identifiers of the result of single beta quality metrics
*/
enum ResultId
{
    rms = 0,                 /*!< NumericTable 1 x k. Root means square errors computed for each response (dependent variable) */
    variance = 1,            /*!< NumericTable 1 x k. Variance computed for each response (dependent variable) */
    zScore = 2,              /*!< NumericTable k x nBeta. Z-score statistics used in testing of insignificance one beta coefficient. H0: beta[i]=0 */
    confidenceIntervals = 3, /*!< NumericTable k x 2 x nBeta. Limits of the confidence intervals for each beta */
    inverseOfXtX = 4         /*!< NumericTable nBeta x nBeta. Inverse(Xt * X) matrix */
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__LINEAR_REGRESSION__SINGLE_BETA__RESULT_ID"></a>
* \brief Available identifiers of the result of single beta quality metrics
*/
enum ResultDataCollectionId
{
    betaCovariances = 5     /*!< DataColection, contains k numeric tables with nBeta x nBeta variance-covariance matrix for betas of each response (dependent variable) */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="ALGORITHMS__LINEAR_REGRESSION__SINGLE_BETA__PARAMETER"></a>
 * \brief Parameters for the compute() method of single beta quality metrics
 *
 * \snippet linear_regression/linear_regression_single_beta_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter: public daal::algorithms::Parameter
{
    Parameter(double alphaVal = 0.05, double accuracyVal = 0.001): alpha(alphaVal), accuracyThreshold(accuracyVal) {}
    virtual ~Parameter() {}

    double alpha;                /*!< Significance level used in the computation of betas confidence intervals */
    double accuracyThreshold;    /*!< Values below this threshold are considered equal to it*/
    /**
    * Checks the correctness of the parameter
    */
    virtual void check() const;
};
/* [Parameter source code] */

/**
* <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__SINGLE_BETA__INPUT"></a>
* \brief %Input objects for single beta quality metrics
*/
class DAAL_EXPORT Input: public daal::algorithms::Input
{
public:
    DAAL_CAST_OPERATOR(Input);

    /** Default constructor */
    Input() : daal::algorithms::Input(3) {}

    virtual ~Input() {}

    /**
    * Returns an input object for linear regression quality metric
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    services::SharedPtr<data_management::NumericTable> get(DataInputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
    * Sets an input object for linear regression quality metric
    * \param[in] id      Identifier of the input object
    * \param[in] value   Pointer to the object
    */
    void set(DataInputId id, const services::SharedPtr<data_management::NumericTable> &value)
    {
        Argument::set(id, value);
    }

    /**
    * Returns an input object representing linear regression model
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    services::SharedPtr<linear_regression::Model> get(ModelInputId id) const
    {
        return services::staticPointerCast<linear_regression::Model, data_management::SerializationIface>(Argument::get(id));
    }

    /**
    * Sets an input object representing linear regression model
    * \param[in] id      Identifier of the input object
    * \param[in] value   %Input object
    */
    void set(ModelInputId id, const services::SharedPtr<linear_regression::Model> &value)
    {
        Argument::set(id, value);
    }

    /**
    * Checks an input object for the linear regression algorithm
    * \param[in] par     Algorithm parameter
    * \param[in] method  Computation method
    */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};
typedef services::SharedPtr<Input> InputPtr;

/**
* <a name="DAAL-CLASS-LINEAR_REGRESSION__SINGLE_BETA__RESULT"></a>
* \brief Provides interface for the result of linear regression quality metrics
*/
class Result: public daal::algorithms::Result
{
public:
    DAAL_CAST_OPERATOR(Result);
    Result() : daal::algorithms::Result(6) {};

    /**
    * Returns the result of linear regression quality metrics
    * \param[in] id    Identifier of the result
    * \return          Result that corresponds to the given identifier
    */
    services::SharedPtr<data_management::NumericTable> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
    * Sets the result of linear regression quality metrics
    * \param[in] id      Identifier of the input object
    * \param[in] value   %Input object
    */
    void set(ResultId id, const services::SharedPtr<data_management::NumericTable> &value)
    {
        Argument::set(id, value);
    }

    /**
    * Returns the result of linear regression quality metrics
    * \param[in] id    Identifier of the result
    * \return          Result that corresponds to the given identifier
    */
    services::SharedPtr<data_management::DataCollection> get(ResultDataCollectionId id) const
    {
        return services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    }

    /**
    * Sets the result of linear regression quality metrics
    * \param[in] id      Identifier of the input object
    * \param[in] value   %Input object
    */
    void set(ResultDataCollectionId id, const services::SharedPtr<data_management::DataCollection> &value)
    {
        Argument::set(id, value);
    }

    /**
    * Allocates memory to store
    * \param[in] input   %Input object
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Algorithm method
    */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
    {
        const services::SharedPtr<data_management::NumericTable> dependentVariableTable = (static_cast<const Input *>(input))->get(expectedResponses);
        const size_t nDepVariable = dependentVariableTable->getNumberOfColumns();

        Argument::set(rms,
            services::SharedPtr<data_management::SerializationIface>(
            new data_management::HomogenNumericTable<algorithmFPType>
            (nDepVariable, 1, data_management::NumericTableIface::doAllocate, 0)));

        Argument::set(variance,
            services::SharedPtr<data_management::SerializationIface>(
            new data_management::HomogenNumericTable<algorithmFPType>
            (nDepVariable, 1, data_management::NumericTableIface::doAllocate, 0)));

        const size_t nBeta = (static_cast<const Input *>(input))->get(model)->getBeta()->getNumberOfColumns();
        services::SharedPtr<data_management::DataCollection> coll(new data_management::DataCollection());
        for (size_t i = 0; i < nDepVariable; ++i)
        {
            coll->push_back(services::SharedPtr<data_management::SerializationIface>(
                new data_management::HomogenNumericTable<algorithmFPType>
                (nBeta, nBeta, data_management::NumericTableIface::doAllocate, 0)));
        }

        Argument::set(betaCovariances, coll);

        Argument::set(zScore,
            services::SharedPtr<data_management::SerializationIface>(
            new data_management::HomogenNumericTable<algorithmFPType>
            (nBeta, nDepVariable, data_management::NumericTableIface::doAllocate, 0)));

        Argument::set(confidenceIntervals,
            services::SharedPtr<data_management::SerializationIface>(
            new data_management::HomogenNumericTable<algorithmFPType>
            (2*nBeta, nDepVariable, data_management::NumericTableIface::doAllocate, 0)));

        Argument::set(inverseOfXtX,
            services::SharedPtr<data_management::SerializationIface>(
            new data_management::HomogenNumericTable<algorithmFPType>
            (nBeta, nBeta, data_management::NumericTableIface::doAllocate, 0)));
    }

    /**
    * Checks the result of linear regression quality metrics
    * \param[in] input   %Input object
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method
    */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    /**
    * Returns the serialization tag of the linear regression model-based prediction result
    * \return         Serialization tag of the linear regression model-based prediction result
    */

    int getSerializationTag() DAAL_C11_OVERRIDE { return SERIALIZATION_LINEAR_REGRESSION_SINGLE_BETA_RESULT_ID; }

    /**
    *  Serializes an object
    *  \param[in]  arch  Storage for a serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    { serialImpl<data_management::InputDataArchive, false>(arch); }

    /**
    *  Deserializes an object
    *  \param[in]  arch  Storage for a deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    { serialImpl<data_management::OutputDataArchive, true>(arch); }

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;

}
using interface1::Parameter;
using interface1::Result;
using interface1::Input;
using interface1::ResultPtr;
using interface1::InputPtr;

}
}
}
}
}

#endif // __LINEAR_REGRESSION_SINGLE_BETA_TYPES_H__
