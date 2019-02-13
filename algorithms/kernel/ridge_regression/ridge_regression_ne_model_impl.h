/* file: ridge_regression_ne_model_impl.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Declaration of the ridge regression model class that implements the model
//  for the normal equations method
//--
*/

#ifndef __RIDGE_REGRESSION_NE_MODEL_IMPL_H__
#define __RIDGE_REGRESSION_NE_MODEL_IMPL_H__

#include "algorithms/ridge_regression/ridge_regression_ne_model.h"
#include "linear_model_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::services;
/**
 * \brief %Model trained with the ridge regression algorithm using the normal equations method
 */
class ModelNormEqInternal : public linear_model::internal::ModelInternal
{
public:
    typedef linear_model::internal::ModelInternal super;
    /**
     * Constructs the ridge regression model for the normal equations method
     * \param[in] featnum Number of features in the training data set
     * \param[in] nrhs    Number of responses in the training data
     * \param[in] par     Parameters of ridge regression model-based training
     * \param[in] dummy   Dummy variable for the templated constructor
     */
    template <typename modelFPType>
    ModelNormEqInternal(size_t featnum, size_t nrhs, const ridge_regression::Parameter &par, modelFPType dummy, services::Status &s);

    ModelNormEqInternal() {}

    virtual ~ModelNormEqInternal() {}

    /**
     * Initializes the ridge regression model
     */
    Status initialize();

    /**
     * Returns a Numeric table that contains partial sums X'*X
     * \return Numeric table that contains partial sums X'*X
     */
    NumericTablePtr getXTXTable();

    /**
     * Returns a Numeric table that contains partial sums X'*Y
     * \return Numeric table that contains partial sums X'*Y
     */
    NumericTablePtr getXTYTable();

protected:
    NumericTablePtr _xtxTable;        /* Table holding a partial sum of X'*X */
    NumericTablePtr _xtyTable;        /* Table holding a partial sum of X'*Y */

    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        super::serialImpl<Archive, onDeserialize>(arch);

        arch->setSharedPtrObj(_xtxTable);
        arch->setSharedPtrObj(_xtyTable);

        return services::Status();
    }
};

class ModelNormEqImpl : public ridge_regression::ModelNormEq,
                        public ModelNormEqInternal
{
public:
    typedef ModelNormEqInternal ImplType;

    /**
     * Constructs the ridge regression model for the normal equations method
     * \param[in] featnum Number of features in the training data set
     * \param[in] nrhs    Number of responses in the training data
     * \param[in] par     Parameters of ridge regression model-based training
     * \param[in] dummy   Dummy variable for the templated constructor
     */
    template <typename modelFPType>
    ModelNormEqImpl(size_t featnum, size_t nrhs, const ridge_regression::Parameter &par, modelFPType dummy, services::Status &s) :
        ImplType(featnum, nrhs, par, dummy, s)
    {}

    ModelNormEqImpl() {}

    virtual ~ModelNormEqImpl() {}

    /**
     * Initializes the ridge regression model
     */
    Status initialize() DAAL_C11_OVERRIDE { return ImplType::initialize(); }

    /**
     * Returns a Numeric table that contains partial sums X'*X
     * \return Numeric table that contains partial sums X'*X
     */
    NumericTablePtr getXTXTable() DAAL_C11_OVERRIDE { return ImplType::getXTXTable(); }

    /**
     * Returns a Numeric table that contains partial sums X'*Y
     * \return Numeric table that contains partial sums X'*Y
     */
    NumericTablePtr getXTYTable() DAAL_C11_OVERRIDE { return ImplType::getXTYTable(); }

    /**
     * Returns the number of regression coefficients
     * \return Number of regression coefficients
     */
    size_t getNumberOfBetas() const DAAL_C11_OVERRIDE  { return ImplType::getNumberOfBetas(); }

    /**
     * Returns the number of responses in the training data set
     * \return Number of responses in the training data set
     */
    size_t getNumberOfResponses() const DAAL_C11_OVERRIDE  { return ImplType::getNumberOfResponses(); }

    /**
     * Returns true if the regression model contains the intercept term, and false otherwise
     * \return True if the regression model contains the intercept term, and false otherwise
     */
    bool getInterceptFlag() const DAAL_C11_OVERRIDE { return ImplType::getInterceptFlag(); }

    /**
     * Returns the number of features in the training data set
     * \return Number of features in the training data set
     */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE { return ImplType::getNumberOfFeatures(); }

    /**
     * Returns the numeric table that contains regression coefficients
     * \return Table that contains regression coefficients
     */
    data_management::NumericTablePtr getBeta() DAAL_C11_OVERRIDE { return ImplType::getBeta(); }

protected:

    services::Status serializeImpl(InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {
        ImplType::serialImpl<InputDataArchive, false>(arch);

        return services::Status();
    }

    services::Status deserializeImpl(const OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {
        ImplType::serialImpl<const OutputDataArchive, true>(arch);

        return services::Status();
    }
};

}
}
}
}
#endif
