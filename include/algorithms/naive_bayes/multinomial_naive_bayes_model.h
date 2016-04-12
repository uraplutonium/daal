/* file: multinomial_naive_bayes_model.h */
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
//  Implementation of class defining Naive Bayes model.
//--
*/

#ifndef __NAIVE_BAYES_MODEL_H__
#define __NAIVE_BAYES_MODEL_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/classifier/classifier_model.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for multinomial Naive Bayes algorithm
 */
namespace multinomial_naive_bayes
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__PARAMETER"></a>
 * \brief Naive Bayes algorithm parameters
 *
 * \snippet naive_bayes/multinomial_naive_bayes_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public classifier::Parameter
{
    /**
     *  Main constructor
     *  \param[in] nClasses            Number of classes
     *  \param[in] priorClassEstimates Prior class estimates, numeric table of size [nClasses x 1]
     *  \param[in] alpha               Imagined occurrences of the each feature, numeric table of size [1 x nFeatures]
     */
    Parameter(size_t nClasses, data_management::NumericTable *priorClassEstimates = NULL,
              data_management::NumericTable *alpha = NULL):
        classifier::Parameter(nClasses), priorClassEstimates(priorClassEstimates), alpha(alpha) {}

    data_management::NumericTable *priorClassEstimates;   /*!< Prior class estimates */
    data_management::NumericTable *alpha;                 /*!< Imagined occurrences of the each word */
};
/* [Parameter source code] */


/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__MODEL"></a>
 * \brief Multinomial naive Bayes model
 */
class Model : public classifier::Model
{
public:
    /**
     * Empty constructor for deserialization
     */
    Model() {}

    /**
     * Constructs multinomial naive Bayes model
     * \param[in] nFeatures  The number of features
     * \param[in] parameter  The multinomial naive Bayes parameter
     * \param[in] dummy      Dummy variable for the templated constructor
     */
    template<typename modelFPType>
    Model(size_t nFeatures, const Parameter &parameter, modelFPType dummy)
    {
        using namespace data_management;

        const Parameter *par = &parameter;
        if(par == 0 || par->nClasses == 0 || nFeatures == 0)
        {
            return;
        }

        _logP     = services::SharedPtr<NumericTable>(new HomogenNumericTable<modelFPType>(1,         par->nClasses, NumericTable::doAllocate));
        _logTheta = services::SharedPtr<NumericTable>(new HomogenNumericTable<modelFPType>(nFeatures, par->nClasses, NumericTable::doAllocate));
        _auxTable = services::SharedPtr<NumericTable>(new HomogenNumericTable<modelFPType>(nFeatures, par->nClasses, NumericTable::doAllocate));

        _nFeatures = nFeatures;
    }

    /** \private */
    virtual ~Model() {}

    /**
     * Returns a pointer to the Numeric Table with logarithms of priors
     *  \return Pointer to the Numeric Table with logarithms of priors
     */
    services::SharedPtr<data_management::NumericTable> getLogP() { return _logP; }

    /**
     * Returns a pointer to the Numeric Table with logarithms of the conditional probabilities
     *  \return Pointer to the Numeric Table with logarithms of the conditional probabilities
     */
    services::SharedPtr<data_management::NumericTable> getLogTheta() { return _logTheta; }

    /**
     * Returns a pointer to the Numeric Table with logarithms of the conditional probabilities
     *  \return Pointer to the Numeric Table with logarithms of the conditional probabilities
     */
    services::SharedPtr<data_management::NumericTable> getAuxTable() { return _auxTable; }

    int getSerializationTag() { return SERIALIZATION_NAIVE_BAYES_MODEL_ID; }
    /**
     *  Implements serialization of the multinomial naive Bayes model object
     *  \param[in]  archive  Storage for the serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive *archive)
    {serialImpl<data_management::InputDataArchive, false>(archive);}

    /**
     *  Deserializes the multinomial Naive Bayes model object
     *  \param[in]  archive  Storage for the deserialized object or data structure
     */
    void deserializeImpl(data_management::OutputDataArchive *archive)
    {serialImpl<data_management::OutputDataArchive, true>(archive);}

    size_t getNFeatures()
    {
        return _nFeatures;
    }

    void setNFeatures( size_t nFeatures )
    {
        _nFeatures = nFeatures;
    }

protected:
    services::SharedPtr<data_management::NumericTable> _logP;
    services::SharedPtr<data_management::NumericTable> _logTheta;
    services::SharedPtr<data_management::NumericTable> _auxTable;
    size_t _nFeatures;

    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        classifier::Model::serialImpl<Archive, onDeserialize>(arch);

        arch->setSharedPtrObj(_logP    );
        arch->setSharedPtrObj(_logTheta);
        arch->setSharedPtrObj(_auxTable);

        arch->set(_nFeatures);
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__MODEL"></a>
 * \brief PartialModel represents partial multinomial naive Bayes model
 */
class PartialModel : public classifier::Model
{
public:
    /**
     * Empty constructor for deserialization
     */
    PartialModel() {}

    /**
     * Constructs multinomial naive Bayes model
     * \param[in] nFeatures  The number of features
     * \param[in] parameter  Multinomial naive Bayes parameter
     * \param[in] dummy      Dummy variable for the templated constructor
     */
    template<typename modelFPType>
    PartialModel(size_t nFeatures, const Parameter &parameter, modelFPType dummy)
    {
        using namespace data_management;
        const Parameter *par = &parameter;
        if(par == 0 || par->nClasses == 0 || nFeatures == 0)
        {
            return;
        }

        _classSize     = services::SharedPtr<NumericTable>(new HomogenNumericTable<int>(1,         par->nClasses, NumericTable::doAllocate));
        _classGroupSum = services::SharedPtr<NumericTable>(new HomogenNumericTable<int>(nFeatures, par->nClasses, NumericTable::doAllocate));

        _nFeatures = nFeatures;
        _nObservations = 0;
    }

    /** \private */
    virtual ~PartialModel() {}

    int getSerializationTag() { return SERIALIZATION_NAIVE_BAYES_PARTIALMODEL_ID; }
    /**
     *  Implements serialization
     *  \param[in]  archive  Storage for the serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive *archive)
    {serialImpl<data_management::InputDataArchive, false>(archive);}

    /**
     *  Implements deserialization
     *  \param[in]  archive  Storage for the deserialized object or data structure
     */
    void deserializeImpl(data_management::OutputDataArchive *archive)
    {serialImpl<data_management::OutputDataArchive, true>(archive);}

    size_t getNFeatures()
    {
        return _nFeatures;
    }

    void setNFeatures( size_t nFeatures )
    {
        _nFeatures = nFeatures;
    }

    size_t getNObservations()
    {
        return _nObservations;
    }

    void setNObservations( size_t nObservations )
    {
        _nObservations = nObservations;
    }

    services::SharedPtr<data_management::NumericTable> getClassSize()     { return _classSize;     }
    services::SharedPtr<data_management::NumericTable> getClassGroupSum() { return _classGroupSum; }

protected:
    services::SharedPtr<data_management::NumericTable> _classSize;
    services::SharedPtr<data_management::NumericTable> _classGroupSum;
    size_t _nFeatures;
    size_t _nObservations;

    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        classifier::Model::serialImpl<Archive, onDeserialize>(arch);

        arch->setSharedPtrObj(_classSize);
        arch->setSharedPtrObj(_classGroupSum);

        arch->set(_nFeatures);
    }
};
} // namespace interface1
using interface1::Parameter;
using interface1::Model;
using interface1::PartialModel;

} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
#endif
