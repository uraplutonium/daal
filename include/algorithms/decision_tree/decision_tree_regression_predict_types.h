/* file: decision_tree_regression_predict_types.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of the Decision tree regression algorithm interface
//--
*/

#ifndef __DECISION_TREE_REGRESSION_PREDICT_TYPES_H__
#define __DECISION_TREE_REGRESSION_PREDICT_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/data_serialize.h"
#include "algorithms/decision_tree/decision_tree_regression_model.h"

namespace daal
{
namespace algorithms
{

/**
 * \brief Contains classes of the Decision tree algorithm
 */
namespace decision_tree
{

/**
 * \brief Contains classes of the Decision tree regression algorithm
 */
namespace regression
{

/**
 * @defgroup decision_tree_prediction Prediction
 * \copydoc daal::algorithms::decision_tree::regression::prediction
 * @ingroup decision_tree
 * @{
 */
/**
 * \brief Contains a class for making Decision tree model-based prediction
 */
namespace prediction
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_TREE__REGRESSION__PREDICTION__METHOD"></a>
 * \brief Available methods for making Decision tree model-based prediction
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_TREE__PREDICTION__NUMERICTABLEINPUTID"></a>
 * \brief Available identifiers of input numeric tables for making decision tree model-based prediction
 */
enum NumericTableInputId
{
    data = 0 /*!< Input data table */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_TREE__PREDICTION__MODELINPUTID"></a>
 * \brief Available identifiers of input models for making decision tree model-based prediction
 */
enum ModelInputId
{
    model = 1 /*!< Trained decision tree model */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_TREE__PREDICTION__RESULTID"></a>
 * \brief Available identifiers of the result for making decision tree model-based prediction
 */
enum ResultId
{
    prediction = 0 /*!< Result of decision tree model-based prediction */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__REGRESSION__PREDICTION__INPUT"></a>
 * \brief Provides an interface for input objects for making Decision tree model-based prediction
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    Input();

    using daal::algorithms::Input::get;

    /**
     * Returns an input object for making Decision tree model-based prediction
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(NumericTableInputId id) const;

    /**
     * Returns the input Model object in the prediction stage of the Decision tree algorithm
     * \param[in] id    Identifier of the input Model object
     * \return          %Input object that corresponds to the given identifier
     */
    ModelPtr get(ModelInputId id) const;

    /**
     * Sets the input NumericTable object in the prediction stage of the regression algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(NumericTableInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets the input Model object in the prediction stage of the Decision tree algorithm
     * \param[in] id      Identifier of the input object
     * \param[in] value   Input Model object
     */
    void set(ModelInputId id, const ModelPtr & value);

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

typedef services::SharedPtr<Input> InputPtr;
typedef services::SharedPtr<const Input> InputConstPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_TREE__PREDICTION__RESULT"></a>
 * \brief Provides interface for the result of decision tree model-based prediction
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE();
    Result();

    /**
     * Returns the result of decision tree model-based prediction
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of decision tree model-based prediction
     * \param[in] id      Identifier of the input object
     * \param[in] value   %Input object
     */
    void set(ResultId id, const data_management::NumericTablePtr &value);

    /**
     * Allocates memory to store a partial result of decision tree model-based prediction
     * \param[in] input   %Input object
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Algorithm method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method);

    /**
     * Checks the result of decision tree model-based prediction
     * \param[in] input   %Input object
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch) { daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch); }

    void serializeImpl(data_management::InputDataArchive   *arch) DAAL_C11_OVERRIDE { serialImpl<data_management::InputDataArchive, false>(arch); }

    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE { serialImpl<data_management::OutputDataArchive, true>(arch); }
};

typedef services::SharedPtr<Result> ResultPtr;
typedef services::SharedPtr<const Result> ResultConstPtr;

} // namespace interface1

using interface1::Input;
using interface1::InputPtr;
using interface1::InputConstPtr;
using interface1::Result;
using interface1::ResultPtr;
using interface1::ResultConstPtr;

} // namespace prediction
/** @} */
} // namespace regression
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
