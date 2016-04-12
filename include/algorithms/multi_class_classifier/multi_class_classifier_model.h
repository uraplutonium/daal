/* file: multi_class_classifier_model.h */
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
//  Multiclass tcc parameter structure
//--
*/

#ifndef __MULTI_CLASS_CLASSIFIER_MODEL_H__
#define __MULTI_CLASS_CLASSIFIER_MODEL_H__

#include "services/daal_defines.h"
#include "algorithms/model.h"
#include "algorithms/classifier/classifier_model.h"
#include "algorithms/classifier/classifier_predict.h"
#include "algorithms/classifier/classifier_training_batch.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for computing the results of the multi-class classifier algorithm
 */
namespace multi_class_classifier
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__PARAMETERBASE"></a>
 * \brief Parameters of the multi-class classifier algorithm
 *
 * \snippet multi_class_classifier/multi_class_classifier_model.h ParameterBase source code
 */
/* [ParameterBase source code] */
struct DAAL_EXPORT ParameterBase : public daal::algorithms::classifier::Parameter
{
    ParameterBase(size_t nClasses): daal::algorithms::classifier::Parameter(nClasses), training(), prediction() {}
    services::SharedPtr<classifier::training::Batch> training;          /*!< Two-class classifier training stage */
    services::SharedPtr<classifier::prediction::Batch> prediction;      /*!< Two-class classifier prediction stage */
};
/* [ParameterBase source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__PARAMETER"></a>
 * \brief Optional multi-class classifier algorithm  parameters that are used with the MultiClassClassifierWu prediction method
 *
 * \snippet multi_class_classifier/multi_class_classifier_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public ParameterBase
{
    Parameter(size_t nClasses, size_t maxIterations = 100, double accuracyThreshold = 1.0e-12) :
        ParameterBase(nClasses), maxIterations(maxIterations), accuracyThreshold(accuracyThreshold) {}

    size_t maxIterations;     /*!< Maximum number of iterations */
    double accuracyThreshold; /*!< Convergence threshold */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTICLASSCLASSIFIER__MODEL"></a>
 * \brief Model of the classifier trained by the multi_class_classifier::training::Batch algorithm.
 */
class Model : public daal::algorithms::classifier::Model
{
public:
    /**
     *  Constructs multi-class classifier model
     *  \param[in] par Parameters of the multi-class classifier algorithm
     */
    Model(const ParameterBase *par) : _nBaseClassifiers(0), _models(NULL)
    {
        size_t nClasses = par->nClasses;
        _nBaseClassifiers = nClasses * (nClasses - 1) / 2;
        _models = new services::SharedPtr<classifier::Model>[_nBaseClassifiers];
    }

    /**
     * Empty constructor for deserialization
     */
    Model() : _nBaseClassifiers(0), _models(NULL) {}

    ~Model()
    {
        delete [] _models;
    }

    /**
     *  Returns a pointer to the array of two-class classifier models in a multi-class classifier model
     *  \return Pointer to the array of two-class classifier models
     */
    services::SharedPtr<classifier::Model> *getTwoClassClassifierModels()
    {
        return _models;
    }

    int getSerializationTag() { return SERIALIZATION_MULTI_CLASS_CLASSIFIER_MODEL_ID; }
    /**
     *  Implements serialization of the multi-class classifier model object
     *  \param[in]  archive  Storage for the serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive *archive)
    {serialImpl<data_management::InputDataArchive, false>(archive);}

    /**
     *  Implements deserialization of the multi-class classifier model object
     *  \param[in]  archive  Storage for the deserialized object or data structure
     */
    void deserializeImpl(data_management::OutputDataArchive *archive)
    {serialImpl<data_management::OutputDataArchive, true>(archive);}

protected:
    size_t _nBaseClassifiers;                           /* Number of two-class classifiers associated with the model */
    services::SharedPtr<classifier::Model> *_models;              /* Array of two-class classifiers associated with the model */

    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::classifier::Model::serialImpl<Archive, onDeserialize>(arch);

        size_t nBaseClassifiers = _nBaseClassifiers;
        arch->set(nBaseClassifiers);
        _nBaseClassifiers = nBaseClassifiers;

        if (onDeserialize)
        {
            if (_models) { delete [] _models; }
            _models = new services::SharedPtr<classifier::Model>[_nBaseClassifiers];
        }

        for (size_t i = 0; i < _nBaseClassifiers; i++)
        {
            arch->setSharedPtrObj(_models[i]);
        }
    }
};
} // namespace interface1
using interface1::ParameterBase;
using interface1::Parameter;
using interface1::Model;

} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal
#endif
