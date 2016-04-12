/* file: implicit_als_model.h */
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
//  Declaration of the implicit ALS model class
//--
*/

#ifndef __IMPLICIT_ALS_MODEL_H__
#define __IMPLICIT_ALS_MODEL_H__

#include "algorithms/model.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PARAMETER"></a>
 * \brief Parameters for the compute() method of the implicit ALS algorithm
 *
 * \snippet implicit_als/implicit_als_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    Parameter(size_t nFactors = 10, size_t maxIterations = 5, double alpha = 40.0, double lambda = 0.01,
              double preferenceThreshold = 0.0, size_t seed = 777777) :
        nFactors(nFactors), maxIterations(maxIterations), alpha(alpha), lambda(lambda),
        preferenceThreshold(preferenceThreshold)
    {}

    size_t nFactors;            /*!< Number of factors */
    size_t maxIterations;       /*!< Maximum number of iterations of the implicit ALS training algorithm */
    double alpha;               /*!< Confidence parameter of the implicit ALS training algorithm */
    double lambda;              /*!< Regularization parameter */
    double preferenceThreshold; /*!< Threshold used to define preference values */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__MODEL"></a>
 * \brief Model trained by the implicit ALS algorithm in the batch processing mode
 *
 * \par References
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - Parameter class
 */
class Model : public daal::algorithms::Model
{
public:
    /**
     * Constructs the implicit ALS model
     * \param[in] nUsers    Number of users in the input data set
     * \param[in] nItems    Number of items in the input data set
     * \param[in] parameter Implicit ALS parameters
     * \param[in] dummy     Dummy variable for the templated constructor
     */
    template<typename modelFPType>
    Model(size_t nUsers, size_t nItems, const Parameter &parameter, modelFPType dummy)
    {
        size_t nFactors = parameter.nFactors;

        _usersFactors = services::SharedPtr<data_management::NumericTable>(
                    new data_management::HomogenNumericTable<modelFPType>(
                            nFactors, nUsers, data_management::NumericTableIface::doAllocate, 0));
        _itemsFactors = services::SharedPtr<data_management::NumericTable>(
                    new data_management::HomogenNumericTable<modelFPType>(
                            nFactors, nItems, data_management::NumericTableIface::doAllocate, 0));
    }

    /**
     * Empty constructor for deserialization
     */
    Model()
    {}

    virtual ~Model()
    {}

    /**
     * Returns a pointer to the numeric table of users factors constructed during the training
     * of the implicit ALS model
     * \return Numeric table of users factors
     */
    services::SharedPtr<data_management::NumericTable> getUsersFactors() const { return _usersFactors; }

    /**
     * Returns a pointer to the numeric table of items factors constructed during the training
     * of the implicit ALS model
     * \return Numeric table of items factors
     */
    services::SharedPtr<data_management::NumericTable> getItemsFactors() const { return _itemsFactors; }

    /**
     * Returns a serialization tag, a unique identifier of this class used in serialization
     * \return Serialization tag
     */
    int getSerializationTag() { return SERIALIZATION_IMPLICIT_ALS_MODEL_ID; }

    /**
     *  Serializes a model object
     *  \param[in]  archive  Storage for a serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive  *archive)
    {serialImpl<data_management::InputDataArchive, false>(archive);}

    /**
     *  Deserializes a model object
     *  \param[in]  archive  Storage for a deserialized object or data structure
     */
    void deserializeImpl(data_management::OutputDataArchive *archive)
    {serialImpl<data_management::OutputDataArchive, true>(archive);}

protected:
    services::SharedPtr<data_management::NumericTable> _usersFactors;    /* Table of resulting users factors */
    services::SharedPtr<data_management::NumericTable> _itemsFactors;    /* Table of resulting items factors */

    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Model::serialImpl<Archive, onDeserialize>(arch);

        arch->setSharedPtrObj(_usersFactors);
        arch->setSharedPtrObj(_itemsFactors);
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PARTIALMODEL"></a>
 * \brief Partial model trained by the implicit ALS training algorithm in the distributed processing mode
 *
 * \par References
 *      - \ref training::interface1::Distributed "implicit_als::training::Distributed"
 *      - Parameter class
 */
class PartialModel : public daal::algorithms::Model
{
public:
    /**
     * Constructs a partial implicit ALS model of a specified size
     * \param[in] parameter Implicit ALS parameters
     * \param[in] size      Model size
     * \param[in] dummy     Dummy variable for the templated constructor
     */
    template<typename modelFPType>
    PartialModel(const Parameter &parameter, size_t size, modelFPType dummy)
    {
        size_t nFactors = parameter.nFactors;

        _factors = services::SharedPtr<data_management::NumericTable>(
                                  new data_management::HomogenNumericTable<modelFPType>(
                                      nFactors, size, data_management::NumericTableIface::doAllocate));
        data_management::HomogenNumericTable<int> *_indicesTable = new data_management::HomogenNumericTable<int>(
                                      1, size, data_management::NumericTableIface::doAllocate);
        _indices = services::SharedPtr<data_management::NumericTable>(_indicesTable);
        int *indicesData = _indicesTable->getArray();
        int iSize = (int)size;
        for (int i = 0; i < iSize; i++)
        {
            indicesData[i] = i;
        }
    }

    /**
     * Constructs a partial implicit ALS model from the indices of factors
     * \param[in] parameter Implicit ALS parameters
     * \param[in] offset    Index of the first factor in the partial model
     * \param[in] indices   Pointer to the numeric table with the indices of factors
     * \param[in] dummy     Dummy variable for the templated constructor
     */
    template<typename modelFPType>
    PartialModel(const Parameter &parameter, size_t offset,
                 services::SharedPtr<data_management::NumericTable> indices, modelFPType dummy)
    {
        size_t nFactors = parameter.nFactors;

        data_management::BlockDescriptor<int> block;
        size_t size = indices->getNumberOfRows();
        indices->getBlockOfRows(0, size, data_management::readOnly, block);
        int *srcIndicesData = block.getBlockPtr();

        _factors = services::SharedPtr<data_management::NumericTable>(
                                  new data_management::HomogenNumericTable<modelFPType>(
                                      nFactors, size, data_management::NumericTableIface::doAllocate));
        data_management::HomogenNumericTable<int> *_indicesTable = new data_management::HomogenNumericTable<int>(
                                      1, size, data_management::NumericTableIface::doAllocate);
        _indices = services::SharedPtr<data_management::NumericTable>(_indicesTable);
        int *dstIndicesData = _indicesTable->getArray();
        int iOffset = (int)offset;
        for (size_t i = 0; i < size; i++)
        {
            dstIndicesData[i] = srcIndicesData[i] + iOffset;
        }
        indices->releaseBlockOfRows(block);
    }

    /**
     * Constructs a partial implicit ALS model from the indices and factors stored in the numeric tables

     * \param[in] factors   Pointer to the numeric table with factors stored in row-major order
     * \param[in] indices   Pointer to the numeric table with the indices of factors
     */
    PartialModel(services::SharedPtr<data_management::NumericTable> factors,
                 services::SharedPtr<data_management::NumericTable> indices) :
        _factors(factors), _indices(indices)
    {}

    /**
     * Empty constructor for deserialization
     */
    PartialModel()
    {}

    virtual ~PartialModel()
    {}

    /**
     * Returns pointer to the numeric table with factors stored in row-major order
     * \return Pointer to the numeric table with factors stored in row-major order
     */
    services::SharedPtr<data_management::NumericTable> getFactors() const { return _factors; }

    /**
     * Returns the pointer to the numeric table with the indices of factors
     * \return Pointer to the numeric table with the indices of factors
     */
    services::SharedPtr<data_management::NumericTable> getIndices() const { return _indices; }

    /**
     * Returns a serialization tag, a unique identifier of this class used in serialization
     * \return Serialization tag
     */
    int getSerializationTag() { return SERIALIZATION_IMPLICIT_ALS_PARTIALMODEL_ID; }

    /**
     *  Serializes a model object
     *  \param[in]  archive  Storage for a serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive  *archive)
    {serialImpl<data_management::InputDataArchive, false>(archive);}

    /**
     *  Deserializes a model object
     *  \param[in]  archive  Storage for a deserialized object or data structure
     */
    void deserializeImpl(data_management::OutputDataArchive *archive)
    {serialImpl<data_management::OutputDataArchive, true>(archive);}

protected:
    services::SharedPtr<data_management::NumericTable> _factors;      /* Factors in row-major format */
    services::SharedPtr<data_management::NumericTable> _indices;      /* Indices of the factors */

    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Model::serialImpl<Archive, onDeserialize>(arch);

        arch->setSharedPtrObj(_factors);
        arch->setSharedPtrObj(_indices);
    }
};
} // namespace interface1
using interface1::Parameter;
using interface1::Model;
using interface1::PartialModel;

}
}
}

#endif
