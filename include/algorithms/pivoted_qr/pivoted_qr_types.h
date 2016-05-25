/* file: pivoted_qr_types.h */
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
//  Definition of Pivoted QR common types.
//--
*/


#ifndef __PIVOTED_QR_TYPES_H__
#define __PIVOTED_QR_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
/** \brief Contains classes for computing the pivoted QR decomposition */
namespace pivoted_qr
{
/**
 * <a name="DAAL-ENUM-PIVOTED_QR__METHOD"></a>
 * Available methods for computing the results of the pivoted QR algorithm
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
 * <a name="DAAL-ENUM-PIVOTED_QR__INPUTID"></a>
 * Available types of input objects for the pivoted QR algorithm
 */
enum InputId
{
    data = 0         /*!< Input data table */
};

/**
 * <a name="DAAL-ENUM-PIVOTED_QR__RESULTID"></a>
 * Available types of results of the pivoted QR algorithm
 */
enum ResultId
{
    matrixQ  = 0,            /*!< Orthogonal Matrix Q */
    matrixR  = 1,            /*!< Upper Triangular Matrix R */
    permutationMatrix = 2    /*!< The permutation matrix P overwritten by its details  */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-QRPARAMETERS"></a>
 * \brief Parameter for the pivoted QR computation method
 */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    Parameter(const services::SharedPtr<data_management::NumericTable> permutedColumns = services::SharedPtr<data_management::NumericTable>())
        : daal::algorithms::Parameter(), permutedColumns(permutedColumns)
    {}

    services::SharedPtr<data_management::NumericTable> permutedColumns;    /*!< On entry, if i-th element of permutedColumns != 0,
                                                                  * the i-th column of input matrix is moved  to the beginning of Data * P before
                                                                  * the computation, and fixed in place during the computation.
                                                                  * If i-th element of permutedColumns = 0, the i-th column of input data
                                                                  * is a free column (that is, it may be interchanged during the
                                                                  * computation with any other free column). */
};

/**
 * <a name="DAAL-CLASS-PIVOTED_QR__INPUT"></a>
 * \brief Input objects for the pivoted QR algorithm in the batch processing mode
 */
class Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    Input() : daal::algorithms::Input(1) {}
    /** Default destructor */
    virtual ~Input() {}

    /**
     * Returns input object for the pivoted QR algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets input object for the pivoted QR algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value Pointer to the input object
     */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &value)
    {
        Argument::set(id, value);
    }

    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if (!data_management::checkNumericTable(get(data).get(), this->_errors.get(), strData())) { return; }

        size_t nVectors = get(data)->getNumberOfRows();
        size_t nFeatures = get(data)->getNumberOfColumns();

        if (nVectors < nFeatures) {
            services::SharedPtr<services::Error> error = services::SharedPtr<services::Error>(new services::Error(services::ErrorIncorrectNumberOfRows));
            error->addStringDetail(services::ArgumentName, strData());
            this->_errors->add(error);
            return;
        }

        Parameter *parameter = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));

        if(parameter->permutedColumns.get() != NULL)
        {

            int unexpectedLayouts = (int)data_management::NumericTableIface::csrArray |
                                    (int)data_management::NumericTableIface::upperPackedTriangularMatrix |
                                    (int)data_management::NumericTableIface::lowerPackedTriangularMatrix |
                                    (int)data_management::NumericTableIface::upperPackedSymmetricMatrix |
                                    (int)data_management::NumericTableIface::lowerPackedSymmetricMatrix;

            if (!data_management::checkNumericTable(parameter->permutedColumns.get(), this->_errors.get(), strPermutedColumns(),
                unexpectedLayouts, 0, nFeatures, 1)) { return; }
        }
    }
};

/**
 * <a name="DAAL-CLASS-PIVOTED_QR__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the pivoted QR algorithm in the batch processing mode
 */
class Result : public daal::algorithms::Result
{
public:
    /** Default constructor */
    Result() : daal::algorithms::Result(3) {}
    /** Default destructor */
    virtual ~Result() {}

    /**
     * Allocates memory for storing final results of the pivoted QR algorithm
     * \param[in] input        Pointer to input object
     * \param[in] parameter    Pointer to parameter
     * \param[in] method       Computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        size_t m = static_cast<const Input *>(input)->get(data)->getNumberOfColumns();
        size_t n = static_cast<const Input *>(input)->get(data)->getNumberOfRows();

        Argument::set(matrixQ, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(m, n, data_management::NumericTable::doAllocate)));
        Argument::set(matrixR, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(m, m, data_management::NumericTable::doAllocate)));
        Argument::set(permutationMatrix, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<size_t>(m, 1, data_management::NumericTable::doAllocate, 0)));
    }

    /**
     * Returns result of the pivoted QR algorithm
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets data_management::NumericTable to store the result of the pivoted QR algorithm
     * \param[in] id    Identifier of the result
     * \param[in] value Pointer to the storage data_management::NumericTable
     */
    void set(ResultId id, const services::SharedPtr<data_management::NumericTable> &value)
    {
        Argument::set(id, value);
    }

    /**
    * Checks the correctness of the result object
    * \param[in] in     Pointer to the input objects structure
    * \param[in] par    Pointer to the structure of the algorithm parameters
    * \param[in] method Computation method
    */
    void check(const daal::algorithms::Input *in, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        const Input *input = static_cast<const Input *>(in);

        size_t nVectors = input->get(data)->getNumberOfRows();
        size_t nFeatures = input->get(data)->getNumberOfColumns();

        int unexpectedLayouts = (int)data_management::NumericTableIface::csrArray |
                                (int)data_management::NumericTableIface::upperPackedTriangularMatrix |
                                (int)data_management::NumericTableIface::lowerPackedTriangularMatrix |
                                (int)data_management::NumericTableIface::upperPackedSymmetricMatrix |
                                (int)data_management::NumericTableIface::lowerPackedSymmetricMatrix;

        if (!data_management::checkNumericTable(get(matrixQ).get(), this->_errors.get(), strMatrixQ(),
            unexpectedLayouts, 0, nFeatures, nVectors)) { return; }

        if (!data_management::checkNumericTable(get(permutationMatrix).get(), this->_errors.get(), strPermutationMatrix(),
            unexpectedLayouts, 0, nFeatures, 1)) { return; }

        unexpectedLayouts = (int)data_management::NumericTableIface::csrArray |
                            (int)data_management::NumericTableIface::lowerPackedTriangularMatrix |
                            (int)data_management::NumericTableIface::upperPackedSymmetricMatrix |
                            (int)data_management::NumericTableIface::lowerPackedSymmetricMatrix;

        if (!data_management::checkNumericTable(get(matrixR).get(), this->_errors.get(), strMatrixR(),
            unexpectedLayouts, 0, nFeatures, nFeatures)) { return; }
    }

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_PIVOTED_QR_RESULT_ID; }

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

} // namespace daal::algorithms::pivoted_qr
} // namespace daal::algorithms
} // namespace daal
#endif
