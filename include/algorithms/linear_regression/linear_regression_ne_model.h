/* file: linear_regression_ne_model.h */
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
//  Declaration of the linear regression model class for the normal equations method
//--
*/

#ifndef __LINREG_NE_MODEL_H__
#define __LINREG_NE_MODEL_H__

#include "algorithms/linear_regression/linear_regression_model.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{

namespace interface1
{
/**
 * <a name="DAAL-CLASS-LINEARREGRESSIONMODELNORMEQ"></a>
 * \brief %Model trained with the linear regression algorithm using the normal equations method
 *
 * \tparam modelFPType  Data type to store linear regression model data, double or float
 *
 * \par References
 *      - Parameter class
 *      - Model class
 *      - Prediction class
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref training::interface1::Online "training::Online" class
 *      - \ref training::interface1::Distributed "training::Distributed" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class ModelNormEq : public Model
{
public:
    /**
     * Constructs the linear regression model for the normal equations method
     * \param[in] featnum Number of features in the training data set
     * \param[in] nrhs    Number of responses in the training data
     * \param[in] par     Parameters of linear regression model-based training
     * \param[in] dummy   Dummy variable for the templated constructor
     */
    template <typename modelFPType>
    ModelNormEq(size_t featnum, size_t nrhs, const linear_regression::Parameter &par, modelFPType dummy):
        Model(featnum, nrhs, par, dummy)
    {
        size_t dimWithoutBeta = _coefdim;
        if(!_interceptFlag)
        {
            dimWithoutBeta--;
        };

        _xtxTable = services::SharedPtr<data_management::NumericTable>(
            new data_management::HomogenNumericTable<modelFPType>(dimWithoutBeta, dimWithoutBeta,
                                                                  data_management::NumericTableIface::doAllocate, 0));
        _xtyTable = services::SharedPtr<data_management::NumericTable>(
            new data_management::HomogenNumericTable<modelFPType>(dimWithoutBeta, nrhs,
                                                                  data_management::NumericTableIface::doAllocate, 0));
    };

    /**
     * Empty constructor for deserialization
     */
    ModelNormEq() : Model() { }

    virtual ~ModelNormEq() { }

    /**
    * Initializes the linear regression model
    */
    void initialize()
    {
        Model::initialize();

        this->setToZero(_xtxTable.get());
        this->setToZero(_xtyTable.get());
    }

    /**
     * Returns a Numeric table that contains partial sums X'*X
     * \return Numeric table that contains partial sums X'*X
     */
    services::SharedPtr<data_management::NumericTable> getXTXTable() { return _xtxTable; }

    /**
     * Returns a Numeric table that contains partial sums X'*Y
     * \return Numeric table that contains partial sums X'*Y
         */
    services::SharedPtr<data_management::NumericTable> getXTYTable() { return _xtyTable; }

    /**
     * Returns the serialization tag of the linear regression model
     * \return         Serialization tag of the linear regression model
     */

    int getSerializationTag() { return SERIALIZATION_LINEAR_REGRESSION_MODELNORMEQ_ID; }
    /**
    *  Serializes an object
    *  \param[in]  arch  Storage for a serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch)
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes an object
    *  \param[in]  arch  Storage for a deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch)
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        Model::serialImpl<Archive, onDeserialize>(arch);

        arch->setSharedPtrObj(_xtxTable);
        arch->setSharedPtrObj(_xtyTable);
    }

private:
    services::SharedPtr<data_management::NumericTable> _xtxTable;        /* Table holding a partial sum of X'*X */
    services::SharedPtr<data_management::NumericTable> _xtyTable;        /* Table holding a partial sum of X'*Y */
};
} // namespace interface1
using interface1::ModelNormEq;

}
}
}
#endif
