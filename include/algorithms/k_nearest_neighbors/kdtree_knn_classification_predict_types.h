/* file: kdtree_knn_classification_predict_types.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
* All Rights Reserved.
*
* If this  software was obtained  under the  Intel Simplified  Software License,
* the following terms apply:
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*
*
* If this  software  was obtained  under the  Apache License,  Version  2.0 (the
* "License"), the following terms apply:
*
* You may  not use this  file except  in compliance  with  the License.  You may
* obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
*
*
* Unless  required  by   applicable  law  or  agreed  to  in  writing,  software
* distributed under the License  is distributed  on an  "AS IS"  BASIS,  WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*
* See the   License  for the   specific  language   governing   permissions  and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of the K-Nearest Neighbors (kNN) algorithm interface
//--
*/

#ifndef __KDTREE_KNN_CLASSIFICATION_PREDICT_TYPES_H__
#define __KDTREE_KNN_CLASSIFICATION_PREDICT_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/data_serialize.h"
#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_model.h"
#include "algorithms/classifier/classifier_predict_types.h"

namespace daal
{
namespace algorithms
{

/**
 * \brief Contains classes of the KD-tree based kNN algorithm
 */
namespace kdtree_knn_classification
{

/**
 * @defgroup kdtree_knn_classification_prediction Prediction
 * \copydoc daal::algorithms::kdtree_knn_classification::prediction
 * @ingroup kdtree_knn_classification
 * @{
 */
/**
 * \brief Contains a class for making KD-tree based kNN model-based prediction
 */
namespace prediction
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__METHOD"></a>
 * \brief Available methods for making KD-tree based kNN model-based prediction
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__NUMERICTABLEINPUTSTEP1ID"></a>
 * Available identifiers of input numeric table objects of the KD-tree based kNN prediction algorithm
 * of the distributed processing mode
 */
enum NumericTableInputStep1Id
{
    data                    = 0     /*!< Input data set */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__NUMERICTABLEINPUTSTEP1ID"></a>
 * Available identifiers of input partial model objects of the KD-tree based kNN prediction algorithm
 * of the distributed processing mode
 */
enum PartialModelInputId
{
    partialModel            = 1     /*!< Input partial model trained by the classification algorithm */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__NUMERICTABLEINPUTSTEP2ID"></a>
 * Available identifiers of input numeric table objects of the KD-tree based kNN prediction algorithm
 * of the distributed processing mode
 */
enum NumericTableInputStep2Id
{
    arrangedData            = 0,    /*!< Input arranged data set */
    intermediatePrediction  = 2,    /*!< Intermediate predicted labels */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__INTINPUTSTEP2ID"></a>
 * Available identifiers of input integers of the KD-tree based kNN prediction algorithm
 * of the distributed processing mode
 */
enum IntInputStep2Id
{
    key                     = 3,     /*!< Key */
    round                   = 4      /*!< Round */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__NUMERICTABLEINPUTSTEP2ID"></a>
 * Available identifiers of input numeric table objects of the KD-tree based kNN prediction algorithm
 * of the distributed processing mode
 */
enum NumericTableInputStep2PerNodeId
{
    communicationResponses    = 5,      /*!< Communication responses from computation nodes */
    communicationInputQueries = 6       /*!< Communication input queries from computation nodes */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__DISTRIBUTEDPARTIALRESULTSTEP1ID"></a>
 * Partial results obtained in the previous step and required by the second step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep1Id
{
    keys                    = 0     /*!< Keys of computation nodes */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__DISTRIBUTEDPARTIALRESULTSTEP2ID"></a>
 * Partial results obtained in the previous step and required by the third step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep2Id
{
    prediction              = 0     /*!< Predicted labels */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__DISTRIBUTEDPARTIALRESULTSTEP2PERNODEID"></a>
 * Partial results obtained in the previous step and required by the third step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep2PerNodeId
{
    communicationQueries         = 1,   /*!< Communication queries for computation nodes */
    communicationOutputResponses = 2    /*!< Communication output responses for computation nodes */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__INPUT"></a>
 * \brief Provides an interface for input objects for making KD-tree based kNN model-based prediction
 */
class DAAL_EXPORT Input : public classifier::prediction::Input
{
    typedef classifier::prediction::Input super;
public:
    /** Default constructor */
    Input();

    using super::get;
    using super::set;

    /**
     * Returns the input Model object in the prediction stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input Model object
     * \return          %Input object that corresponds to the given identifier
     */
    kdtree_knn_classification::ModelPtr get(classifier::prediction::ModelInputId id) const;

    /**
     * Sets the input NumericTable object in the prediction stage of the classification algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(classifier::prediction::NumericTableInputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Sets the input Model object in the prediction stage of the KD-tree based kNN algorithm
     * \param[in] id      Identifier of the input object
     * \param[in] value   Input Model object
     */
    void set(classifier::prediction::ModelInputId id, const kdtree_knn_classification::interface1::ModelPtr & value);

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__DISTRIBUTEDINPUT"></a>
 * \brief %Input objects for the KD-tree based kNN prediction algorithm in the distributed processing mode
 */
template <ComputeStep step>
class DistributedInput
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__DISTRIBUTEDINPUT_STEP1LOCAL"></a>
 * \brief %Input objects for the KD-tree based kNN prediction algorithm in the first step
 * of the distributed processing mode
 */
template<>
class DAAL_EXPORT DistributedInput<step1Local> : public classifier::prediction::Input
{
public:
    /** Default constructor */
    DistributedInput();

    virtual ~DistributedInput() {}

    /**
     * Returns the input Numeric Table object in the prediction stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(NumericTableInputStep1Id id) const;

    /**
     * Returns an input object for the prediction stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    services::SharedPtr<PartialModel> get(PartialModelInputId id) const;

    /**
     * Sets the input NumericTable object in the prediction stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(NumericTableInputStep1Id id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets an input object for the prediction stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(PartialModelInputId id, const services::SharedPtr<PartialModel> & ptr);

    /**
     * Checks the parameters of the prediction stage of the KD-tree based kNN algorithm
     * \param[in] parameter     Algorithm %parameter
     * \param[in] method        Computation method for the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    services::Status checkImpl(const daal::algorithms::Parameter * parameter) const;
};


/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__DISTRIBUTEDINPUT_STEP2LOCAL"></a>
 * \brief %Input objects for the KD-tree based kNN prediction algorithm in the second step
 * of the distributed processing mode
 */
template<>
class DAAL_EXPORT DistributedInput<step2Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    virtual ~DistributedInput() {}

    /**
     * Returns the input Numeric Table object in the prediction stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(NumericTableInputStep2Id id) const;

    /**
     * Returns the input int in the prediction stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input int
     * \return          Value that corresponds to the given identifier
     */
    int get(IntInputStep2Id id) const;

    /**
     * Returns the input KeyValueDataCollection object in the prediction stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(NumericTableInputStep2PerNodeId id) const;

    /**
     * Returns an input object for the prediction stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    services::SharedPtr<PartialModel> get(PartialModelInputId id) const;

    /**
     * Sets the input NumericTable object in the prediction stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(NumericTableInputStep2Id id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets the input int in the prediction stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value Pointer to the input object
     */
    void set(IntInputStep2Id id, int value);

    /**
     * Sets the input KeyValueDataCollection object in the prediction stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(NumericTableInputStep2PerNodeId id, const data_management::KeyValueDataCollectionPtr & ptr);

    /**
     * Sets an input object for the prediction stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(PartialModelInputId id, const services::SharedPtr<PartialModel> & ptr);

    /**
     * Adds an input object for the prediction stage of the KD-tree based kNN algorithm
     * \param[in] id            Identifier of the input object
     * \param[in] key           Key of the source of the input object
     * \param[in] value         Pointer to the input object
     */
    void add(NumericTableInputStep2PerNodeId id, size_t key, const data_management::NumericTablePtr & value);

    /**
     * Checks the parameters of the prediction stage of the KD-tree based kNN algorithm
     * \param[in] parameter     Algorithm %parameter
     * \param[in] method        Computation method for the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

    /**
     * Returns the number of rows in the input data set
     * \return Number of rows in the input data set
     */
    size_t getNumberOfRows() const;

    /**
     * Returns the number of columns in the input data set
     * \return Number of columns in the input data set
     */
    size_t getNumberOfColumns() const;

protected:
    services::Status checkImpl(const daal::algorithms::Parameter * parameter) const;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__DISTRIBUTEDPARTIALRESULTSTEP1"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the KD-tree based kNN algorithm in the first step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep1 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE();

    /** Default constructor */
    DistributedPartialResultStep1();

    /**
     * Allocates memory to store a partial result of the KD-tree based kNN prediction algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method);

    /**
     * Returns a partial result of the KD-tree based kNN prediction algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep1Id id) const;

    /**
     * Sets a partial result of the KD-tree based kNN prediction algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep1Id id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks a partial result of the KD-tree based kNN algorithm
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }

    services::Status serializeImpl(data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        return serialImpl<data_management::InputDataArchive, false>(arch);
    }

    services::Status deserializeImpl(const data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        return serialImpl<const data_management::OutputDataArchive, true>(arch);
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__PREDICTION__DISTRIBUTEDPARTIALRESULTSTEP2"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the KD-tree based kNN algorithm in the second step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep2 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE();

    /** Default constructor */
    DistributedPartialResultStep2();

    /**
     * Allocates memory to store a partial result of the KD-tree based kNN prediction algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method);

    /**
     * Returns a partial result of the KD-tree based kNN prediction algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep2Id id) const;

    /**
     * Returns a partial result of the KD-tree based kNN prediction algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(DistributedPartialResultStep2PerNodeId id) const;

    /**
     * Returns a partial result of the KD-tree based kNN prediction algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] key   Identifier of the computation node
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep2PerNodeId id, int key) const;

    /**
     * Sets a partial result of the KD-tree based kNN prediction algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep2Id id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets a partial result of the KD-tree based kNN prediction algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep2PerNodeId id, const data_management::KeyValueDataCollectionPtr & ptr);

    /**
     * Checks a partial result of the KD-tree based kNN algorithm
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }

    services::Status serializeImpl(data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        return serialImpl<data_management::InputDataArchive, false>(arch);
    }

    services::Status deserializeImpl(const data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        return serialImpl<const data_management::OutputDataArchive, true>(arch);
    }
};

} // namespace interface1

using interface1::Input;
using interface1::DistributedInput;
using interface1::DistributedPartialResultStep1;
using interface1::DistributedPartialResultStep2;

} // namespace prediction
/** @} */
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
