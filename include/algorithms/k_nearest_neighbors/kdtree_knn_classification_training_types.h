/* file: kdtree_knn_classification_training_types.h */
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
//  Implementation of the k-Nearest Neighbor (kNN) algorithm interface
//--
*/

#ifndef __KNN_CLASSIFICATION_TRAINING_TYPES_H__
#define __KNN_CLASSIFICATION_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/data_serialize.h"
#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_model.h"
#include "algorithms/classifier/classifier_training_types.h"

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
 * @defgroup kdtree_knn_classification_training Training
 * \copydoc daal::algorithms::kdtree_knn_classification::training
 * @ingroup kdtree_knn_classification
 * @{
 */
/**
 * \brief Contains a class for KD-tree based kNN model-based training
 */
namespace training
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__METHOD"></a>
 * \brief Computation methods for KD-tree based kNN model-based training
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUTSTEP2ID"></a>
 * Available identifiers of input numeric table objects of the KD-tree based kNN training algorithm
 * of the distributed processing mode
 */
enum DistributedInputStep2Id
{
    inputOfStep2 = 0     /*!< Input of second step */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUTSTEP3ID"></a>
 * Available identifiers of input numeric table objects of the KD-tree based kNN training algorithm
 * of the distributed processing mode
 */
enum DistributedInputStep3Id
{
    dataForStep3 = 0,           /*!< Training data for third step */
    labelsForStep3 = 1,         /*!< Training labels for third step */
    boundingBoxesForStep3 = 3,  /*!< Bounding boxes for third step */
    numberOfLoopsForStep3 = 4   /*!< Number of loops for third step */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUTSTEP3INTID"></a>
 * Available identifiers of input integers of the KD-tree based kNN training algorithm
 * of the distributed processing mode
 */
enum DistributedInputStep3IntId
{
    loopNumberForStep3 = 5,     /*!< Loop number for third step */
    nodeIndexForStep3 = 6,      /*!< Computation node index for third step */
    nodeCountForStep3 = 7       /*!< Node count for third step */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUTSTEP4ID"></a>
 * Available identifiers of input numeric table objects of the KD-tree based kNN training algorithm
 * of the distributed processing mode
 */
enum DistributedInputStep4Id
{
    dataForStep4 = 0,           /*!< Training data for fourth step */
    labelsForStep4 = 1,         /*!< Training labels for fourth step */
    dimensionForStep4 = 3,      /*!< Dimension (feature) for fourth step */
    boundingBoxesForStep4 = 4   /*!< Bounding boxes for fourth step */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUTSTEP4PERNODEID"></a>
 * Available identifiers of input numeric table objects of the KD-tree based kNN training algorithm
 * of the distributed processing mode
 */
enum DistributedInputStep4PerNodeId
{
    samplesForStep4 = 5         /*!< Samples for fourth step */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUTSTEP5ID"></a>
 * Available identifiers of input numeric table objects of the KD-tree based kNN training algorithm
 * of the distributed processing mode
 */
enum DistributedInputStep5Id
{
    dataForStep5 = 0,           /*!< Training data for fifth step */
    labelsForStep5 = 1,         /*!< Training labels for fifth step */
    dimensionForStep5 = 3       /*!< Dimension (feature) for fifth step */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUTSTEP5BOOLID"></a>
 * Available identifiers of input boolean of the KD-tree based kNN training algorithm
 * of the distributed processing mode
 */
enum DistributedInputStep5BoolId
{
    isPartnerGreaterForStep5 = 4    /*!< Flag, which indicates that partner computation node is logically greater relative to this one*/
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUTSTEP5PERNODEID"></a>
 * Available identifiers of input numeric table objects of the KD-tree based kNN training algorithm
 * of the distributed processing mode
 */
enum DistributedInputStep5PerNodeId
{
    histogramForStep5 = 5           /*!< Histogram for fifth step */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUTSTEP6ID"></a>
 * Available identifiers of input numeric table objects of the KD-tree based kNN training algorithm
 * of the distributed processing mode
 */
enum DistributedInputStep6Id
{
    dataForStep6 = 0,               /*!< Training data for sixth step */
    labelsForStep6 = 1,             /*!< Training labels for sixth step */
    dataFromPartnerForStep6 = 3,    /*!< Training data from partner computation node for sixth step */
    labelsFromPartnerForStep6 = 4,  /*!< Training labels from partner computation node for sixth step */
    markersForStep6 = 5             /*!< Slot markers for sixth step */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUTSTEP7ID"></a>
 * Available identifiers of input numeric table objects of the KD-tree based kNN training algorithm
 * of the distributed processing mode
 */
enum DistributedInputStep7Id
{
    boundingBoxesForStep7 = 0,      /*!< Bounding boxes for seventh step */
    numberOfLoopsForStep7 = 1       /*!< Number of loops for seventh step */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUTSTEP7INTID"></a>
 * Available identifiers of input numeric table objects of the KD-tree based kNN training algorithm
 * of the distributed processing mode
 */
enum DistributedInputStep7IntId
{
    loopNumberForStep7 = 2          /*!< Loop number for seventh step */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUTSTEP7PARTIALMODELID"></a>
 * Available identifiers of input partial model of the KD-tree based kNN training algorithm
 * of the distributed processing mode
 */
enum DistributedInputStep7PartialModelId
{
    partialModelForStep7 = 3        /*!< Partial model for seventh step */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUTSTEP7PERNODEID"></a>
 * Available identifiers of input numeric table objects of the KD-tree based kNN training algorithm
 * of the distributed processing mode
 */
enum DistributedInputStep7PerNodeId
{
    dimensionForStep7 = 4,          /*!< Dimension (feature) for seventh step */
    medianForStep7    = 5           /*!< Median value for seventh step */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUTSTEP8ID"></a>
 * Available identifiers of input numeric table objects of the KD-tree based kNN training algorithm
 * of the distributed processing mode
 */
enum DistributedInputStep8Id
{
    dataForStep8 = 0,               /*!< Training data for eighth step */
    labelsForStep8 = 1,             /*!< Training labels for eighth step */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUTSTEP8PARTIALMODELID"></a>
 * Available identifiers of input numeric table objects of the KD-tree based kNN training algorithm
 * of the distributed processing mode
 */
enum DistributedInputStep8PartialModelId
{
    partialModelForStep8 = 3        /*!< Partial model for eighth step */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP1ID"></a>
 * Partial results obtained in the previous step and required by the first step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep1Id
{
    boundingBoxes = 0       /*!< Bounding boxes */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP2ID"></a>
 * Partial results obtained in the previous step and required by the second step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep2Id
{
    globalBoundingBoxes = 0,    /*!< Global bounding boxes */
    numberOfLoops       = 1     /*!< Number of loops */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP3ID"></a>
 * Partial results obtained in the previous step and required by the third step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep3Id
{
    localSamples = 0,           /*!< Local samples */
    dimension    = 1,           /*!< Dimension (feature) */
    color        = 2,           /*!< Color (communication group identifier) */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP4ID"></a>
 * Partial results obtained in the previous step and required by the fourth step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep4Id
{
    localHistogram = 0          /*!< Local histogram */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP5ID"></a>
 * Partial results obtained in the previous step and required by the fifth step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep5Id
{
    dataForPartner   = 0,       /*!< Data for partner communication node */
    labelsForPartner = 1,       /*!< Labels for partner communication node */
    median           = 2,       /*!< Median value */
    markers          = 3        /*!< Slot markers */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP6ID"></a>
 * Partial results obtained in the previous step and required by the sixth step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep6Id
{
    concatenatedData   = 0,     /*!< Concatenated Data */
    concatenatedLabels = 1      /*!< Concatenated Labels */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP7ID"></a>
 * Partial results obtained in the previous step and required by the seventh step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep7Id
{
    boundingBoxesOfStep7ForStep3 = 0    /*!< Bounding boxes of step 7 for step 3 */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP7ID"></a>
 * Partial results obtained in the previous step and required by the seventh step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep7PartialModelId
{
    partialModelOfStep7 = 1         /*!< Partial model of step 7 */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP8ID"></a>
 * Partial results obtained in the previous step and required by the eighth step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep8Id
{
    partialModel = 0                /*!< Partial model */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method of KD-tree based kNN model-based training
 */
class DAAL_EXPORT Result : public classifier::training::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);
    Result();

    /**
     * Returns the result of KD-tree based kNN model-based training
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    daal::algorithms::kdtree_knn_classification::interface1::ModelPtr get(classifier::training::ResultId id) const;

    /**
     * Allocates memory to store the result of KD-tree based kNN model-based training
     * \param[in] input Pointer to an object containing the input data
     * \param[in] parameter %Parameter of KD-tree based kNN model-based training
     * \param[in] method Computation method for the algorithm
     */
    template<typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const Parameter *parameter, int method);

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        classifier::training::Result::serialImpl<Archive, onDeserialize>(arch);

        return services::Status();
    }

    services::Status serializeImpl(data_management::InputDataArchive   *arch) DAAL_C11_OVERRIDE
    {
        serialImpl<data_management::InputDataArchive, false>(arch);

        return services::Status();
    }

    services::Status deserializeImpl(const data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {
        serialImpl<const data_management::OutputDataArchive, true>(arch);

        return services::Status();
    }
};
typedef services::SharedPtr<Result> ResultPtr;



/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUT"></a>
 * \brief %Input objects for the KD-tree based kNN training algorithm in the distributed processing mode
 */
template <ComputeStep step>
class DistributedInput
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUT_STEP1LOCAL"></a>
 * \brief %Input objects for the KD-tree based kNN training algorithm in the first step
 * of the distributed processing mode
 */
template<>
class DAAL_EXPORT DistributedInput<step1Local> : public classifier::training::Input
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUT_STEP2MASTER"></a>
 * \brief %Input objects for the KD-tree based kNN training algorithm in the second step
 * of the distributed processing mode
 */
template<>
class DAAL_EXPORT DistributedInput<step2Master> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /**
     * Sets the input NumericTable object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(DistributedInputStep2Id id, const data_management::KeyValueDataCollectionPtr & ptr);

    /**
     * Returns the input Numeric Table object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(DistributedInputStep2Id id) const;

    /**
     * Adds an input object for the training stage of the KD-tree based kNN algorithm
     * \param[in] id            Identifier of the input object
     * \param[in] key           Key of the source of the input object
     * \param[in] value         Pointer to the input object
     */
    void add(DistributedInputStep2Id id, size_t key, const data_management::NumericTablePtr & value);

    /**
     * Checks the parameters of the training stage of the KD-tree based kNN algorithm
     * \param[in] parameter     Algorithm %parameter
     * \param[in] method        Computation method for the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

    /**
     * Returns the number of features
     * \return Number of features
     */
    size_t getNumberOfFeatures() const;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUT_STEP3LOCAL"></a>
 * \brief %Input objects for the KD-tree based kNN training algorithm in the third step
 * of the distributed processing mode
 */
template<>
class DAAL_EXPORT DistributedInput<step3Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /**
     * Returns the input Numeric Table object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedInputStep3Id id) const;

    /**
     * Returns the input integer in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input object that corresponds to the given identifier
     */
    int get(DistributedInputStep3IntId id) const;

    /**
     * Sets the input NumericTable object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value Pointer to the input object
     */
    void set(DistributedInputStep3Id id, const data_management::NumericTablePtr & value);

    /**
     * Sets the input integer in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value Value of the input object
     */
    void set(DistributedInputStep3IntId id, int value);

    /**
     * Checks the parameters of the training stage of the KD-tree based kNN algorithm
     * \param[in] parameter     Algorithm %parameter
     * \param[in] method        Computation method for the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

    /**
     * Returns the number of features
     * \return Number of features
     */
    size_t getNumberOfFeatures() const;

protected:
    services::Status checkImpl(const daal::algorithms::Parameter * parameter) const;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUT_STEP4LOCAL"></a>
 * \brief %Input objects for the KD-tree based kNN training algorithm in the fourth step
 * of the distributed processing mode
 */
template<>
class DAAL_EXPORT DistributedInput<step4Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /**
     * Sets the input Key-Value Data Collection object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(DistributedInputStep4PerNodeId id, const data_management::KeyValueDataCollectionPtr & ptr);

    /**
     * Sets the input NumericTable object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value Pointer to the input object
     */
    void set(DistributedInputStep4Id id, const data_management::NumericTablePtr & value);

    /**
     * Returns the input Key-Value Data Collection object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(DistributedInputStep4PerNodeId id) const;

    /**
     * Returns the input Numeric Table object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedInputStep4Id id) const;

    /**
     * Adds an input object for the training stage of the KD-tree based kNN algorithm
     * \param[in] id            Identifier of the input object
     * \param[in] key           Key of the source of the input object
     * \param[in] value         Pointer to the input object
     */
    void add(DistributedInputStep4PerNodeId id, size_t key, const data_management::NumericTablePtr & value);

    /**
     * Checks the parameters of the training stage of the KD-tree based kNN algorithm
     * \param[in] parameter     Algorithm %parameter
     * \param[in] method        Computation method for the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

    /**
     * Returns the number of features
     * \return Number of features
     */
    size_t getNumberOfFeatures() const;

protected:
    services::Status checkImpl(const daal::algorithms::Parameter * parameter) const;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUT_STEP5LOCAL"></a>
 * \brief %Input objects for the KD-tree based kNN training algorithm in the fifth step
 * of the distributed processing mode
 */
template<>
class DAAL_EXPORT DistributedInput<step5Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /**
     * Sets the input Key-Value Data Collection object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(DistributedInputStep5PerNodeId id, const data_management::KeyValueDataCollectionPtr & ptr);

    /**
     * Sets the input NumericTable object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value Pointer to the input object
     */
    void set(DistributedInputStep5Id id, const data_management::NumericTablePtr & value);

    /**
     * Sets the input boolean in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value Value of the input object
     */
    void set(DistributedInputStep5BoolId id, bool value);

    /**
     * Returns the input Key-Value Data Collection object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(DistributedInputStep5PerNodeId id) const;

    /**
     * Returns the input Numeric Table object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedInputStep5Id id) const;

    /**
     * Returns the input boolean in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input object that corresponds to the given identifier
     */
    bool get(DistributedInputStep5BoolId id) const;

    /**
     * Adds an input object for the training stage of the KD-tree based kNN algorithm
     * \param[in] id            Identifier of the input object
     * \param[in] key           Key of the source of the input object
     * \param[in] value         Pointer to the input object
     */
    void add(DistributedInputStep5PerNodeId id, size_t key, const data_management::NumericTablePtr & value);

    /**
     * Checks the parameters of the training stage of the KD-tree based kNN algorithm
     * \param[in] parameter     Algorithm %parameter
     * \param[in] method        Computation method for the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

    /**
     * Returns the number of features
     * \return Number of features
     */
    size_t getNumberOfFeatures() const;

protected:
    services::Status checkImpl(const daal::algorithms::Parameter * parameter) const;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUT_STEP6LOCAL"></a>
 * \brief %Input objects for the KD-tree based kNN training algorithm in the sixth step
 * of the distributed processing mode
 */
template<>
class DAAL_EXPORT DistributedInput<step6Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /**
     * Sets the input NumericTable object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value Pointer to the input object
     */
    void set(DistributedInputStep6Id id, const data_management::NumericTablePtr & value);

    /**
     * Returns the input Numeric Table object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedInputStep6Id id) const;

    /**
     * Checks the parameters of the training stage of the KD-tree based kNN algorithm
     * \param[in] parameter     Algorithm %parameter
     * \param[in] method        Computation method for the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

    /**
     * Returns the number of features
     * \return Number of features
     */
    size_t getNumberOfFeatures() const;

protected:
    services::Status checkImpl(const daal::algorithms::Parameter * parameter) const;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUT_STEP7MASTER"></a>
 * \brief %Input objects for the KD-tree based kNN training algorithm in the seventh step
 * of the distributed processing mode
 */
template<>
class DAAL_EXPORT DistributedInput<step7Master> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /**
     * Sets the input Key-Value Data Collection object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(DistributedInputStep7PerNodeId id, const data_management::KeyValueDataCollectionPtr & ptr);

    /**
     * Sets the input NumericTable object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(DistributedInputStep7Id id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets the input integer in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value   Value of the input object
     */
    void set(DistributedInputStep7IntId id, int value);

    /**
     * Sets the input Partial model object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value   Pointer to the input object
     */
    void set(DistributedInputStep7PartialModelId id, const services::SharedPtr<kdtree_knn_classification::PartialModel> & value);

    /**
     * Returns the input Key-Value Data Collection object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(DistributedInputStep7PerNodeId id) const;

    /**
     * Returns the input Numeric Table object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedInputStep7Id id) const;

    /**
     * Returns the input integer in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input value that corresponds to the given identifier
     */
    int get(DistributedInputStep7IntId id) const;

    /**
     * Returns the Partil model object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input value that corresponds to the given identifier
     */
    services::SharedPtr<kdtree_knn_classification::PartialModel> get(DistributedInputStep7PartialModelId id) const;

    /**
     * Adds an input object for the training stage of the KD-tree based kNN algorithm
     * \param[in] id            Identifier of the input object
     * \param[in] key           Key of the source of the input object
     * \param[in] value         Pointer to the input object
     */
    void add(DistributedInputStep7PerNodeId id, size_t key, const data_management::NumericTablePtr & value);

    /**
     * Checks the parameters of the training stage of the KD-tree based kNN algorithm
     * \param[in] parameter     Algorithm %parameter
     * \param[in] method        Computation method for the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

    /**
     * Returns the number of features
     * \return Number of features
     */
    size_t getNumberOfFeatures() const;

protected:
    services::Status checkImpl(const daal::algorithms::Parameter * parameter) const;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDINPUT_STEP8LOCAL"></a>
 * \brief %Input objects for the KD-tree based kNN training algorithm in the eighth step
 * of the distributed processing mode
 */
template<>
class DAAL_EXPORT DistributedInput<step8Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /**
     * Sets the input NumericTable object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value   Pointer to the input object
     */
    void set(DistributedInputStep8Id id, const data_management::NumericTablePtr & value);

    /**
     * Sets the input Partial model object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value   Pointer to the input object
     */
    void set(DistributedInputStep8PartialModelId id, const services::SharedPtr<kdtree_knn_classification::PartialModel> & value);

    /**
     * Returns the input Numeric Table object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedInputStep8Id id) const;

    /**
     * Returns the input Partial model object in the training stage of the KD-tree based kNN algorithm
     * \param[in] id    Identifier of the input NumericTable object
     * \return          Input object that corresponds to the given identifier
     */
    services::SharedPtr<kdtree_knn_classification::PartialModel> get(DistributedInputStep8PartialModelId id) const;

    /**
     * Checks the parameters of the training stage of the KD-tree based kNN algorithm
     * \param[in] parameter     Algorithm %parameter
     * \param[in] method        Computation method for the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

    /**
     * Returns the number of features
     * \return Number of features
     */
    size_t getNumberOfFeatures() const;

protected:
    services::Status checkImpl(const daal::algorithms::Parameter * parameter) const;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP1"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the KD-tree based kNN algorithm in the first step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep1 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE();
    DAAL_CAST_OPERATOR(DistributedPartialResultStep1);

    /** Default constructor */
    DistributedPartialResultStep1();

    /**
     * Allocates memory to store a partial result of the KD-tree based kNN training algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Returns a partial result of the KD-tree based kNN training algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep1Id id) const;

    /**
     * Sets a partial result of the KD-tree based kNN training algorithm
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
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP2"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the KD-tree based kNN algorithm in the second step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep2 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE();
    DAAL_CAST_OPERATOR(DistributedPartialResultStep2);

    /** Default constructor */
    DistributedPartialResultStep2();

    /**
     * Allocates memory to store a partial result of the KD-tree based kNN training algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Returns a partial result of the KD-tree based kNN training algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep2Id id) const;

    /**
     * Sets a partial result of the KD-tree based kNN training algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep2Id id, const data_management::NumericTablePtr & ptr);

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
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP3"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the KD-tree based kNN algorithm in the third step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep3 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE();
    DAAL_CAST_OPERATOR(DistributedPartialResultStep3);

    /** Default constructor */
    DistributedPartialResultStep3();

    /**
     * Allocates memory to store a partial result of the KD-tree based kNN training algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Returns a partial result of the KD-tree based kNN training algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep3Id id) const;

    /**
     * Sets a partial result of the KD-tree based kNN training algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep3Id id, const data_management::NumericTablePtr & ptr);

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
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP4"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the KD-tree based kNN algorithm in the fourth step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep4 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE();
    DAAL_CAST_OPERATOR(DistributedPartialResultStep4);

    /** Default constructor */
    DistributedPartialResultStep4();

    /**
     * Allocates memory to store a partial result of the KD-tree based kNN training algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Returns a partial result of the KD-tree based kNN training algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep4Id id) const;

    /**
     * Sets a partial result of the KD-tree based kNN training algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep4Id id, const data_management::NumericTablePtr & ptr);

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
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP5"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the KD-tree based kNN algorithm in the fifth step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep5 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE();
    DAAL_CAST_OPERATOR(DistributedPartialResultStep5);

    /** Default constructor */
    DistributedPartialResultStep5();

    /**
     * Allocates memory to store a partial result of the KD-tree based kNN training algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Returns a partial result of the KD-tree based kNN training algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep5Id id) const;

    /**
     * Sets a partial result of the KD-tree based kNN training algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep5Id id, const data_management::NumericTablePtr & ptr);

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
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP6"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the KD-tree based kNN algorithm in the sixth step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep6 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE();
    DAAL_CAST_OPERATOR(DistributedPartialResultStep6);

    /** Default constructor */
    DistributedPartialResultStep6();

    /**
     * Allocates memory to store a partial result of the KD-tree based kNN training algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Returns a partial result of the KD-tree based kNN training algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep6Id id) const;

    /**
     * Sets a partial result of the KD-tree based kNN training algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep6Id id, const data_management::NumericTablePtr & ptr);

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
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP7"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the KD-tree based kNN algorithm in the seventh step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep7 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE();
    DAAL_CAST_OPERATOR(DistributedPartialResultStep7);

    /** Default constructor */
    DistributedPartialResultStep7();

    /**
     * Allocates memory to store a partial result of the KD-tree based kNN training algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Returns a partial result of the KD-tree based kNN training algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep7Id id) const;

    /**
     * Returns a partial result of the KD-tree based kNN training algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    services::SharedPtr<kdtree_knn_classification::PartialModel> get(DistributedPartialResultStep7PartialModelId id) const;

    /**
     * Sets a partial result of the KD-tree based kNN training algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep7Id id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets a partial result of the KD-tree based kNN training algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep7PartialModelId id, const services::SharedPtr<kdtree_knn_classification::PartialModel> & ptr);

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
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP8"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the KD-tree based kNN algorithm in the eighth step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep8 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE();
    DAAL_CAST_OPERATOR(DistributedPartialResultStep8);

    /** Default constructor */
    DistributedPartialResultStep8();

    /**
     * Allocates memory to store a partial result of the KD-tree based kNN training algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Returns a partial result of the KD-tree based kNN training algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    services::SharedPtr<kdtree_knn_classification::PartialModel> get(DistributedPartialResultStep8Id id) const;

    /**
     * Sets a partial result of the KD-tree based kNN training algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep8Id id, const services::SharedPtr<kdtree_knn_classification::PartialModel> & ptr);

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

using interface1::Result;
using interface1::ResultPtr;
using interface1::DistributedInput;
using interface1::DistributedPartialResultStep1;
using interface1::DistributedPartialResultStep2;
using interface1::DistributedPartialResultStep3;
using interface1::DistributedPartialResultStep4;
using interface1::DistributedPartialResultStep5;
using interface1::DistributedPartialResultStep6;
using interface1::DistributedPartialResultStep7;
using interface1::DistributedPartialResultStep8;

} // namespace training
/** @} */
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
