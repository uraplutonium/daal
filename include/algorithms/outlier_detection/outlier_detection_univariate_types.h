/* file: outlier_detection_univariate_types.h */
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
//  univariate outlier detection algorithm types
//--
*/

#ifndef __OUTLIERDETECTION_UNIVARIATE_TYPES_H__
#define __OUTLIERDETECTION_UNIVARIATE_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for computing results of the univariate outlier detection algorithm
 */
namespace univariate_outlier_detection
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__UNIVARIATE_OUTLIER_DETECTION__METHOD"></a>
 * Available methods for computing results of the univariate outlier detection algorithm
 */
enum Method
{
    defaultDense = 0       /*!< Default: performance-oriented method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__UNIVARIATE_OUTLIER_DETECTION__INPUTID"></a>
 * Available identifiers of input objects of the univariate outlier detection algorithm
 */
enum InputId
{
    data = 0            /*!< %Input data table */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__UNIVARIATE_OUTLIER_DETECTION__RESULTID"></a>
 * Available identifiers of results of the univariate outlier detection algorithm
 */
enum ResultId
{
    weights = 0          /*!< Table with results */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__UNIVARIATE_OUTLIER_DETECTION__INITIFACE"></a>
 * \brief Abstract class that provides a functor for the initial procedure
 */
struct InitIface
{
    /**
    * Returns the initial value for the univariate outlier detection algorithm
    * \param[in] data          Pointer to input values
    * \param[in] location      Vector of mean estimates
    * \param[in] scatter       Measure of spread, the array of standard deviations of size 1 x p
    * \param[in] threshold     Limit that defines the outlier region, the array of non-negative numbers of size 1 x p
    */
    virtual void operator()(data_management::NumericTable *data,
                            data_management::NumericTable *location,
                            data_management::NumericTable *scatter,
                            data_management::NumericTable *threshold) = 0;
    virtual ~InitIface() {}
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__UNIVARIATE_OUTLIER_DETECTION__PARAMETER"></a>
 * \brief Parameters of the univariate outlier detection algorithm
 *
 * \snippet outlier_detection/outlier_detection_univariate_types.h Parameter source code
 */
/* [Parameter source code] */
struct Parameter : public daal::algorithms::Parameter
{
    Parameter() : daal::algorithms::Parameter(), initializationProcedure() {}
    services::SharedPtr<InitIface>
    initializationProcedure;     /*!< Initialization procedure for setting initial parameters of the algorithm*/
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__UNIVARIATE_OUTLIER_DETECTION__INPUT"></a>
 * \brief %Input objects for the univariate outlier detection algorithm
 */
class Input : public daal::algorithms::Input
{
public:
    Input() : daal::algorithms::Input(1) {}

    virtual ~Input() {}

    /**
     * Returns an input object for the univariate outlier detection algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets an input object for the univariate outlier detection algorithm
     * \param[in] id    Identifier of the %input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks input objects for the univariate outlier detection algorithm
     * \param[in] par     Parameters of the algorithm
     * \param[in] method  univariate outlier detection computation method
          */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        services::SharedPtr<data_management::NumericTable> inTable = get(data);

        if(inTable.get() == 0)                 { this->_errors->add(services::ErrorNullInputNumericTable); return;         }
        if(inTable->getNumberOfRows() == 0)    { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(inTable->getNumberOfColumns() == 0) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;     }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__UNIVARIATE_OUTLIER_DETECTION__RESULT"></a>
 * \brief Results obtained with the compute() method of the univariate outlier detection algorithm in the %batch processing mode
 */
class Result : public daal::algorithms::Result
{
public:
    Result() : daal::algorithms::Result(1) {}

    virtual ~Result() {};

    /**
     * Registers user-allocated memory to store univariate outlier detection results
     * \param[in] input   Pointer to the %input objects for the algorithm
     * \param[in] parameter     Pointer to the parameters of the algorithm
     * \param[in] method  univariate outlier detection computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
        size_t nFeatures = algInput->get(data)->getNumberOfColumns();
        size_t nVectors  = algInput->get(data)->getNumberOfRows();
        Argument::set(weights, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(nFeatures, nVectors,
                                                                                    data_management::NumericTable::doAllocate)));
    }

    /**
     * Returns a result of the univariate outlier detection algorithm
     * \param[in] id   Identifier of the result
     * \return         Final result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets a result of the univariate outlier detection algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks the result object of the univariate outlier detection algorithm
     * \param[in] input   Pointer to the  %input objects for the algorithm
     * \param[in] par     Pointer to the parameters of the algorithm
     * \param[in] method  univariate outlier detection computation method
     * \return             Status of checking
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
               int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        services::SharedPtr<data_management::NumericTable> resTable = get(weights);

        if(resTable.get() == 0)                 { this->_errors->add(services::ErrorNullInputNumericTable); return;         }
        if(resTable->getNumberOfRows() == 0)    { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(resTable->getNumberOfColumns() == 0) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;     }

        Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));

        if(resTable->getNumberOfRows() != algInput->get(data)->getNumberOfRows() ||
           resTable->getNumberOfColumns() != algInput->get(data)->getNumberOfColumns())
        { this->_errors->add(services::ErrorIncorrectSizeOfOutputNumericTable); return; }
    }

    int getSerializationTag() { return SERIALIZATION_OUTLIER_DETECTION_UNIVARIATE_RESULT_ID; }

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
using interface1::InitIface;
using interface1::Parameter;
using interface1::Input;
using interface1::Result;

} // namespace univariate_outlier_detection
} // namespace algorithm
} // namespace daal
#endif
