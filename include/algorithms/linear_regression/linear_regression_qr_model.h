/* file: linear_regression_qr_model.h */
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
//  Declaration of the linear regression model class for the QR decomposition-based method
//--
*/

#ifndef __LINREG_QR_MODEL_H__
#define __LINREG_QR_MODEL_H__

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
 * <a name="DAAL-CLASS-LINEARREGRESSIONMODELQR"></a>
 * \brief %Model trained with the linear regression algorithm using the QR decomposition-based method
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
class ModelQR : public Model
{
public:
    /**
     * Constructs the linear regression model for the QR decomposition-based method
     * \param[in] featnum Number of features in the training data set
     * \param[in] nrhs    Number of responses in the training data
     * \param[in] par     Parameters of linear regression model-based training
     * \param[in] dummy   Dummy variable for the templated constructor
     */
    template <typename modelFPType>
    ModelQR(size_t featnum, size_t nrhs, const linear_regression::Parameter &par, modelFPType dummy):
        Model(featnum, nrhs, par, dummy)
    {
        size_t dimWithoutBeta = _coefdim;
        if(!_interceptFlag)
        {
            dimWithoutBeta--;
        };

        _rTable   = new data_management::HomogenNumericTable<modelFPType>(dimWithoutBeta, dimWithoutBeta,
                                                                          data_management::NumericTableIface::doAllocate, 0);
        _qtyTable = new data_management::HomogenNumericTable<modelFPType>(dimWithoutBeta, nrhs,
                                                                          data_management::NumericTableIface::doAllocate, 0);
    };

    /**
     * Empty constructor for deserialization
     */
    ModelQR() : Model()
    {
        _rTable   = 0;
        _qtyTable = 0;
    }

    virtual ~ModelQR()
    {
        if(_rTable ) { delete _rTable; }
        if(_qtyTable) { delete _qtyTable; }
    }

    /**
     * Initializes the linear regression model
     */
    virtual void initialize() DAAL_C11_OVERRIDE
    {
        Model::initialize();

        this->setToZero(_rTable);
        this->setToZero(_qtyTable);
    }

    /**
     * Returns a Numeric table that contains the R factor of QR decomposition
     * \return Numeric table that contains the R factor of QR decomposition
     */
    data_management::NumericTable *getRTable() { return _rTable; }

    /**
     * Returns a Numeric table that contains Q'*Y, where Q is the factor of QR decomposition for a data block,
     * Y is the respective block of the matrix of responses
     * \return Numeric table that contains partial sums Q'*Y
     */
    data_management::NumericTable *getQTYTable() { return _qtyTable; }

    /**
     * Returns the serialization tag of the linear regression model
     * \return         Serialization tag of the linear regression model
     */

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_LINEAR_REGRESSION_MODELQR_ID; }

    /**
    *  Serializes an object
    *  \param[in]  arch  Storage for a serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes an object
    *  \param[in]  arch  Storage for a deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        Model::serialImpl<Archive, onDeserialize>(arch);

        arch->setSingleObj((data_management::SerializationIface **)&_rTable);
        arch->setSingleObj((data_management::SerializationIface **)&_qtyTable);
    }

private:
    data_management::NumericTable *_rTable;        /* Table that contains matrix R */
    data_management::NumericTable *_qtyTable;      /* Table that contains matrix Q'*Y */
};
} // namespace interface1
using interface1::ModelQR;

}
}
} // namespace daal
#endif
