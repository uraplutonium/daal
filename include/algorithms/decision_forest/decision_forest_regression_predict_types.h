/* file: decision_forest_regression_predict_types.h */
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
//  Implementation of the decision forest algorithm interface
//--
*/

#ifndef __DECISION_FOREST_REGRESSSION_PREDICT_TYPES_H__
#define __DECISION_FOREST_REGRESSSION_PREDICT_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/data_serialize.h"
#include "services/daal_defines.h"
#include "algorithms/decision_forest/decision_forest_regression_model.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes of the decision forest algorithm
 */
namespace decision_forest
{
/**
 * @defgroup decision_forest_prediction Prediction
 * \copydoc daal::algorithms::decision_forest::prediction
 * @ingroup decision_forest
 * @{
 */
namespace regression
{
/**
 * \brief Contains a class for making decision forest model-based prediction
 */
namespace prediction
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_FOREST__REGRESSSION__PREDICTION__METHOD"></a>
 * \brief Available methods for making decision forest model-based prediction
 */
enum Method
{
    defaultDense = 0
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_FOREST__REGRESSSION__PREDICTION__NUMERICTABLEINPUTID"></a>
 * \brief Available identifiers of input numeric tables for making decision forest model-based prediction
 */
enum NumericTableInputId
{
    data = 0 /*!< Input data table */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_FOREST__PREDICTION__REGRESSSION__MODELINPUTID"></a>
 * \brief Available identifiers of input models for making decision forest model-based prediction
 */
enum ModelInputId
{
    model = 1 /*!< Trained decision forest model */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_FOREST__PREDICTION__REGRESSSION__RESULTID"></a>
 * \brief Available identifiers of the result for making decision forest model-based prediction
 */
enum ResultId
{
    prediction = 0 /*!< Result of decision forest model-based prediction */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface
 */
namespace interface1
{
/**
* @ingroup decision_forest
* @{
*/

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__REGRESSSION__PREDICTION__INPUT"></a>
 * \brief Provides an interface for input objects for making decision forest model-based prediction
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    Input();

    /** Copy constructor */
    Input(const Input& other) : daal::algorithms::Input(other){}

    /**
     * Returns an input object for making decision forest model-based prediction
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(NumericTableInputId id) const;

    /**
     * Returns an input object for making decision forest model-based prediction
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    decision_forest::regression::ModelPtr get(ModelInputId id) const;

    /**
     * Sets an input object for making decision forest model-based prediction
     * \param[in] id      Identifier of the input object
     * \param[in] value   %Input object
     */
    void set(NumericTableInputId id, const data_management::NumericTablePtr &value);

    /**
     * Sets an input object for making decision forest model-based prediction
     * \param[in] id      Identifier of the input object
     * \param[in] value   %Input object
     */
    void set(ModelInputId id, const decision_forest::regression::ModelPtr &value);

    /**
     * Checks an input object for making decision forest model-based prediction
     * \return Status of checking
     */
    services::Status check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__PREDICTION__RESULT"></a>
 * \brief Provides interface for the result of decision forest model-based prediction
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE();
    Result();

    /**
     * Returns the result of decision forest model-based prediction
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of decision forest model-based prediction
     * \param[in] id      Identifier of the input object
     * \param[in] value   %Input object
     */
    void set(ResultId id, const data_management::NumericTablePtr &value);

    /**
     * Allocates memory to store a partial result of decision forest model-based prediction
     * \param[in] input   %Input object
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Algorithm method
     * \return Status of allocation
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

    /**
     * Checks the result of decision forest model-based prediction
     * \param[in] input   %Input object
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     * \return Status of checking
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }

    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}
};
typedef services::SharedPtr<Result> ResultPtr;

} // namespace interface1
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;

}
/** @} */
}
}
}
} // namespace daal
#endif
