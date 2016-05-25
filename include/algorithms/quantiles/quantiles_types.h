/* file: quantiles_types.h */
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
//  Definition of common types of quantiles.
//--
*/

#ifndef __QUANTILES_TYPES_H__
#define __QUANTILES_TYPES_H__

#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes to run the quantile algorithms
 */
namespace quantiles
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__QUANTILES__METHOD"></a>
 * Available methods for quantiles computation
 */
enum Method
{
    defaultDense = 0    /*!< Default: performance-oriented method. Works with all types of input numeric tables */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__QUANTILES__INPUTID"></a>
 * Available identifiers of input objects for the quantiles algorithm
 */
enum InputId
{
    data = 0            /*!< %Input data table */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__QUANTILES__RESULTID"></a>
 * Available identifiers of results of the quantiles algorithm
 */
enum ResultId
{
    quantiles = 0       /*!< Values of quantiles */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__QUANTILES__PARAMETER"></a>
 * \brief Parameters of the quantiles algorithm
 */
struct Parameter : public daal::algorithms::Parameter
{
    Parameter(const services::SharedPtr<data_management::NumericTable> quantileOrders = services::SharedPtr<data_management::NumericTable>())
        : daal::algorithms::Parameter(), quantileOrders(quantileOrders)
    {
        if(quantileOrders.get() == NULL)
        {
            this->quantileOrders = services::SharedPtr<data_management::NumericTable>(
                                       new data_management::HomogenNumericTable<double>(1, 1, data_management::NumericTableIface::doAllocate, 0.5));
        }
    }

    services::SharedPtr<data_management::NumericTable> quantileOrders;    /*!< Numeric table with quantile orders. Default value is 0.5 (median) */
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QUANTILES__INPUT"></a>
 * \brief %Input objects for the quantiles algorithm
 */
class Input : public daal::algorithms::Input
{
public:
    Input() : daal::algorithms::Input(1)
    {}

    virtual ~Input() {}

    /**
     * Returns an input object for the quantiles algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the input object of the quantiles algorithm
     * \param[in] id    Identifier of the %input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Check the correctness of the %Input object
     * \param[in] parameter Pointer to the parameters structure
     * \param[in] method    Algorithm computation method
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);

        if (!data_management::checkNumericTable(algParameter->quantileOrders.get(), this->_errors.get(),
            strQuantileOrders(), 0, 0, 0, 1)) { return; }

        if (!data_management::checkNumericTable(get(data).get(), this->_errors.get(), strData())) { return; }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QUANTILES__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the
 *        quantiles algorithm in the batch processing mode
 */
class Result : public daal::algorithms::Result
{
public:
    Result() : daal::algorithms::Result(1)
    {}

    virtual ~Result() {};

    /**
     * Allocates memory to store final results of the quantile algorithms
     * \param[in] input     Input objects for the quantiles algorithm
     * \param[in] parameter Parameters of the quantiles algorithm
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const Input *in = static_cast<const Input *>(input);
        const Parameter *par = static_cast<const Parameter *>(parameter);

        size_t nFeatures = in->get(data)->getNumberOfColumns();
        size_t nQuantileOrders = par->quantileOrders->getNumberOfColumns();

        Argument::set(quantiles, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(nQuantileOrders, nFeatures,
                                                                                    data_management::NumericTable::doAllocate)));
    }

    /**
     * Returns the final result of the quantiles algorithm
     * \param[in] id   Identifier of the final result, \ref ResultId
     * \return         Final result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the Result object of the quantiles algorithm
     * \param[in] id        Identifier of the Result object
     * \param[in] value     Pointer to the Result object
     */
    void set(ResultId id, const services::SharedPtr<data_management::NumericTable> &value)
    {
        Argument::set(id, value);
    }

    /**
     * Checks the correctness of the Result object
     * \param[in] in     Pointer to the object
     * \param[in] par    Pointer to the parameters structure
     * \param[in] method Algorithm computation method
     */
    void check(const daal::algorithms::Input *in, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        const Input *input = static_cast<const Input *>(in);
        const Parameter *parameter = static_cast<const Parameter *>(par);

        if (!data_management::checkNumericTable(parameter->quantileOrders.get(), this->_errors.get(),
            strQuantileOrders(), 0, 0, 0, 1)) { return; }

        size_t nVectors  = input->get(data)->getNumberOfColumns();
        size_t nFeatures = parameter->quantileOrders->getNumberOfColumns();

        int unexpectedLayouts = (int)data_management::NumericTableIface::csrArray |
                                (int)data_management::NumericTableIface::upperPackedTriangularMatrix |
                                (int)data_management::NumericTableIface::lowerPackedTriangularMatrix |
                                (int)data_management::NumericTableIface::upperPackedSymmetricMatrix |
                                (int)data_management::NumericTableIface::lowerPackedSymmetricMatrix;

        if (!data_management::checkNumericTable(get(quantiles).get(), this->_errors.get(),
            strQuantiles(), unexpectedLayouts, 0, nFeatures, nVectors)) { return; }
    }

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_QUANTILES_RESULT_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for the serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes the object
    *  \param[in]  arch  Storage for the deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
} // namespace interface1
using interface1::Parameter;
using interface1::Input;
using interface1::Result;

} // namespace daal::algorithms::quantiles
} // namespace daal::algorithms
} // namespace daal
#endif
