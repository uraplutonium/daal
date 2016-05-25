/* file: correlation_distance_types.h */
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
//  Implementation of correlation distance algorithm interface.
//--
*/

#ifndef __CORDISTANCE_TYPES_H__
#define __CORDISTANCE_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/symmetric_matrix.h"

namespace daal
{
namespace algorithms
{
/**
* \brief Contains classes for computing the correlation distance
*/
namespace correlation_distance
{

/**
 * <a name="DAAL-ENUM-CORDISTANCE__METHOD"></a>
 * Available methods for computing the correlation distance
 */
enum Method
{
    defaultDense = 0       /*!< Default: performance-oriented method. */
};

/**
 * <a name="DAAL-ENUM-CORDISTANCE__INPUTID"></a>
 * Available identifiers of input objects for the correlation distance algorithm
 */
enum InputId
{
    data = 0           /*!< %Input data table */
};
/**
 * <a name="DAAL-ENUM-CORDISTANCE__RESULTID"></a>
 * Available identifiers of results for the correlation distance algorithm
 */
enum ResultId
{
    correlationDistance = 0          /*!< Table to store the result.*/
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-CORDISTANCE__INPUT"></a>
 * \brief %Input objects for the correlation distance algorithm
 */
class Input : public daal::algorithms::Input
{
public:
    Input() : daal::algorithms::Input(1) {}

    virtual ~Input() {}

    /**
    * Returns the input object of the correlation distance algorithm
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
    * Sets the input object of the correlation distance algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the object
    */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
    * Checks the parameters of the correlation distance algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if (!data_management::checkNumericTable(get(data).get(), this->_errors.get(), strData())) { return; }
    }
};

/**
 * <a name="DAAL-CLASS-CORDISTANCE__RESULT"></a>
 * \brief Results obtained with compute() method of the correlation distance algorithm in the batch processing mode
 */
class Result : public daal::algorithms::Result
{
public:
    Result() : daal::algorithms::Result(1) {}

    virtual ~Result() {};

    /**
     * Allocates memory to store the results of the correlation distance algorithm
     * \param[in] input  Pointer to input structure
     * \param[in] par    Pointer to parameter structure
     * \param[in] method Computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
    {
        Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
        size_t dim = algInput->get(data)->getNumberOfRows();
        Argument::set(correlationDistance, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::PackedSymmetricMatrix<data_management::NumericTableIface::lowerPackedSymmetricMatrix>(
                              dim, data_management::NumericTable::doAllocate)));
    }

    /**
     * Returns the result of the correlation distance algorithm
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the result of the correlation distance algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the value
     */
    void set(ResultId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
    * Checks the result of the correlation distance algorithm
    * \param[in] input   %Input of the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method
    */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
               int method) const DAAL_C11_OVERRIDE
    {
        const Input *algInput = static_cast<const Input *>(input);

        size_t nVectors  = algInput->get(data)->getNumberOfRows();
        int unexpectedLayouts = (int)data_management::NumericTableIface::csrArray |
                                (int)data_management::NumericTableIface::upperPackedTriangularMatrix |
                                (int)data_management::NumericTableIface::lowerPackedTriangularMatrix;

        if (!data_management::checkNumericTable(get(correlationDistance).get(), this->_errors.get(),
            strCorrelationDistance(), unexpectedLayouts, 0, nVectors, nVectors)) { return; }
    }

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_CORRELATION_DISTANCE_RESULT_ID; }

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
using interface1::Input;
using interface1::Result;

} // namespace correlation_distance
} // namespace algorithms
} // namespace daal
#endif
