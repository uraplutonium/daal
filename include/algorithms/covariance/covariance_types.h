/* file: covariance_types.h */
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
//  Definition of Covariance common types.
//--
*/

#ifndef __COVARIANCE_TYPES_H__
#define __COVARIANCE_TYPES_H__

#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for computing the correlation or variance-covariance matrix
 */
namespace covariance
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__COVARIANCE__METHOD"></a>
 * Available computation methods for variance-covariance or correlation matrix
 */
enum Method
{
    defaultDense    = 0,        /*!< Default: performance-oriented method. Works with all types of numeric tables */
    singlePassDense = 1,        /*!< Single-pass: implementation of the single-pass algorithm proposed by D.H.D. West.
                                     Works with all types of numeric tables */
    sumDense        = 2,        /*!< Precomputed sum: implementation of moments computation algorithm in the case of a precomputed sum.
                                     Works with all types of numeric tables */
    fastCSR         = 3,        /*!< Fast: performance-oriented method. Works with Compressed Sparse Rows (CSR) numeric tables */
    singlePassCSR   = 4,        /*!< Single-pass: implementation of the single-pass algorithm proposed by D.H.D. West.
                                     Works with CSR numeric tables */
    sumCSR          = 5         /*!< Precomputed sum: implementation of the algorithm in the case of a precomputed sum.
                                     Works with CSR numeric tables */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__COVARIANCE__INPUTID"></a>
 * Available identifiers of input objects for the correlation or variance-covariance matrix algorithm
 */
enum InputId
{
    data = 0                /*!< %Input data table */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__COVARIANCE__PARTIALRESULTID"></a>
 * Available identifiers of partial results of the correlation or variance-covariance matrix algorithm
 */
enum PartialResultId
{
    nObservations = 0,      /*!< Number of observations processed so far */
    crossProduct  = 1,      /*!< Cross-product matrix computed so far */
    sum           = 2       /*!< Vector of sums computed so far */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__COVARIANCE__RESULTID"></a>
 * \brief Available identifiers of results of the correlation or variance-covariance matrix algorithm
 */
enum ResultId
{
    covariance      = 0,    /*!< Variance-covariance matrix */
    correlation     = 0,    /*!< Correlation matrix */
    mean            = 1     /*!< Vector of means */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__COVARIANCE__OUTPUTMATRIXTYPE"></a>
 * Available types of the computed matrix for Covariance
 */
enum OutputMatrixType
{
    covarianceMatrix = 0,           /*!< Variance-Covariance matrix */
    correlationMatrix = 1           /*!< Correlation matrix */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__COVARIANCE__MASTERNODEINPUTID"></a>
 * \brief Available identifiers of master node input arguments of the Covariance algorithm
 */
enum MasterInputId
{
    partialResults = 0 /*!< Collection of partial results trained on local nodes */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__INPUT"></a>
 * \brief Abstract class that specifies interface for classes that declare input of the correlation or variance-covariance matrix algorithm
 */
class InputIface : public daal::algorithms::Input
{
public:
    InputIface(size_t nElements) : daal::algorithms::Input(nElements) {}
    virtual size_t getNumberOfFeatures() const = 0;
    virtual ~InputIface() {}
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__INPUT"></a>
 * \brief %Input objects of the correlation or variance-covariance matrix algorithm
 */
class Input : public InputIface
{
public:
    Input() : InputIface(1)
    {}

    virtual ~Input() {}

    /**
     * Returns number of columns in the input data set
     * \return Number of columns in the input data set
     */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE
    {
        services::SharedPtr<data_management::NumericTable> ntPtr =
            services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(get(data));
        if(ntPtr)
        {
            return ntPtr->getNumberOfColumns();
        }
        else
        {
            this->_errors->add(services::ErrorIncorrectSizeOfInputNumericTable);
            return 0;
        }
    }

    /**
     * Returns the input object of the correlation or variance-covariance matrix algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the input object of the correlation or variance-covariance matrix algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Checks algorithm parameters
     * \param[in] parameter Pointer to the structure of algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        int expectedLayouts = 0;

        if (method == fastCSR || method == singlePassCSR || method == sumCSR)
        {
            expectedLayouts = (int)data_management::NumericTableIface::csrArray;
        }

        if (!data_management::checkNumericTable(get(data).get(), this->_errors.get(), strData(), 0, expectedLayouts)) { return; }

        if (method == sumDense || method == sumCSR)
        {
            size_t nFeatures = get(data)->getNumberOfColumns();

            if (!data_management::checkNumericTable(get(data)->basicStatistics.get(data_management::NumericTableIface::sum).get(),
                this->_errors.get(), strSum(), 0, 0, nFeatures, 1)) { return; }
        }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__PARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 *        of the correlation or variance-covariance matrix algorithm
 *        in the online or distributed processing mode
 */
class PartialResult : public daal::algorithms::PartialResult
{
public:
    PartialResult() : daal::algorithms::PartialResult(3)
    {}

    virtual ~PartialResult()
    {}

    /**
     * Allocates memory to store partial results of the correlation or variance-covariance matrix algorithm
     * \param[in] input     %Input objects of the algorithm
     * \param[in] parameter Parameters of the algorithm
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const InputIface *algInput = static_cast<const InputIface *>(input);
        size_t nColumns = algInput->getNumberOfFeatures();

        Argument::set(nObservations, services::SharedPtr<data_management::NumericTable>(
                          new data_management::HomogenNumericTable<size_t>(1, 1, data_management::NumericTable::doAllocate)));
        Argument::set(crossProduct, services::SharedPtr<data_management::NumericTable>(
                          new data_management::HomogenNumericTable<algorithmFPType>(nColumns, nColumns, data_management::NumericTable::doAllocate)));
        Argument::set(sum, services::SharedPtr<data_management::NumericTable>(
                          new data_management::HomogenNumericTable<algorithmFPType>(nColumns, 1, data_management::NumericTable::doAllocate)));
    }

    /**
     * Gets the number of columns in the partial result of the correlation or variance-covariance matrix algorithm
     * \return Number of columns in the partial result
     */
    size_t getNumberOfFeatures() const
    {
        services::SharedPtr<data_management::NumericTable> ntPtr =
            services::dynamicPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(crossProduct));
        if(ntPtr)
        {
            return ntPtr->getNumberOfColumns();
        }
        else
        {
            this->_errors->add(services::ErrorIncorrectSizeOfInputNumericTable);
            return 0;
        }
    }

    /**
     * Returns the partial result of the correlation or variance-covariance matrix algorithm
     * \param[in] id   Identifier of the partial result, \ref PartialResultId
     * \return Partial result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(PartialResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the partial result of the correlation or variance-covariance matrix algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the partial result
     */
    void set(PartialResultId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Check correctness of the partial result
     * \param[in] input     Pointer to the structure with input objects
     * \param[in] parameter Pointer to the structure of algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        const InputIface *algInput = static_cast<const InputIface *>(input);
        size_t nFeatures = algInput->getNumberOfFeatures();
        checkImpl(nFeatures);
    }

    /**
    * Check the correctness of PartialResult object
    * \param[in] parameter Pointer to the structure of the parameters of the algorithm
    * \param[in] method    Computation method
    */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        size_t nFeatures = getNumberOfFeatures();
        checkImpl(nFeatures);
    }

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_COVARIANCE_PARTIAL_RESULT_ID; }

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

    void checkImpl(size_t nFeatures) const
    {
        services::SharedPtr<data_management::NumericTable> presTable = get(nObservations);
        if(!presTable)
        { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        if((presTable->getNumberOfRows() != 1) || (presTable->getNumberOfColumns() != 1))
        { this->_errors->add(services::ErrorIncorrectSizeOfInputNumericTable); return; }

        presTable = get(crossProduct);
        if(!presTable)
        { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        if((presTable->getNumberOfRows() != nFeatures) || (presTable->getNumberOfColumns() != nFeatures))
        { this->_errors->add(services::ErrorIncorrectSizeOfInputNumericTable); return; }

        presTable = get(sum);
        if(!presTable)
        { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        if((presTable->getNumberOfRows() != 1) || (presTable->getNumberOfColumns() != nFeatures))
        { this->_errors->add(services::ErrorIncorrectSizeOfInputNumericTable); return; }
    }

    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__PARAMETER"></a>
 * \brief Parameters of the correlation or variance-covariance matrix algorithm
 */
struct Parameter : public daal::algorithms::Parameter
{
    /** Default constructor */
    Parameter() : daal::algorithms::Parameter(), outputMatrixType(covarianceMatrix) {}
    OutputMatrixType outputMatrixType;      /*!< Type of the computed matrix */
};

/**
 * <a name="DAAL-CLASS-CLASS-ALGORITHMS__COVARIANCE__PARTIALRESULTINITIFACE"></a>
 * \brief Abstract interface class for initialization of partial results
 */
struct PartialResultsInitIface : public Base
{
    /**
     * Initializes partial results of the correlation or variance-covariance matrix algorithm
     * \param[in]       input     %Input objects of the algorithm
     * \param[in,out]   pres      Partial results of the algorithm
     */
    virtual void operator()(const Input &input, services::SharedPtr<PartialResult> &pres) = 0;

protected:
    void setToZero(data_management::NumericTable *table)
    {
        data_management::BlockDescriptor<double> block;
        size_t nCols = table->getNumberOfColumns();
        size_t nRows = table->getNumberOfRows();

        double *data;
        table->getBlockOfRows(0, nRows, data_management::writeOnly, block);
        data = block.getBlockPtr();

        for(size_t i = 0; i < nCols * nRows; i++)
        {
            data[i] = 0.0;
        };

        table->releaseBlockOfRows(block);
    };
};

/**
 * <a name="DAAL-CLASS-CLASS-ALGORITHMS__COVARIANCE__DEFAULTPARTIALRESULTINIT"></a>
 * \brief Class that specifies the default method for initialization of partial results
 */
struct DefaultPartialResultsInit : public PartialResultsInitIface
{
    /**
     * Initializes partial results of the correlation or variance-covariance matrix algorithm
     * \param[in]       input     %Input objects of the algorithm
     * \param[in,out]   pres      Partial results of the algorithm
     */
    virtual void operator()(const Input &input, services::SharedPtr<PartialResult> &pres)
    {
        setToZero(pres->get(nObservations).get());
        setToZero(pres->get(crossProduct).get());
        setToZero(pres->get(sum).get());
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__ONLINEPARAMETER"></a>
 * \brief Parameters of the correlation or variance-covariance matrix algorithm in the online processing mode
 */
struct OnlineParameter : public Parameter
{
    /** Default constructor */
    OnlineParameter() : Parameter(), initializationProcedure(new DefaultPartialResultsInit())
    {}

    services::SharedPtr<PartialResultsInitIface> initializationProcedure;         /**< Functor for partial results initialization */
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of
 *        the correlation or variance-covariance matrix algorithm in the batch processing mode
 */
class Result : public daal::algorithms::Result
{
public:
    Result() : daal::algorithms::Result(2)
    {}

    virtual ~Result() {};

    /**
     * Allocates memory to store final results of the correlation or variance-covariance matrix algorithm
     * \param[in] input     %Input objects of the algorithm
     * \param[in] parameter Parameters of the algorithm
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const Input *algInput = static_cast<const Input *>(input);
        size_t nColumns = algInput->getNumberOfFeatures();

        Argument::set(covariance, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(nColumns, nColumns, data_management::NumericTable::doAllocate)));
        Argument::set(mean, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(nColumns, 1, data_management::NumericTable::doAllocate)));
    }

    /**
     * Allocates memory for storing Covariance final results
     * \param[in] partialResult      Partial Results arguments of the covariance algorithm
     * \param[in] parameter          Parameters of the covariance algorithm
     * \param[in] method             Computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::PartialResult *partialResult, const daal::algorithms::Parameter *parameter, const int method)
    {
        const PartialResult *pres = static_cast<const PartialResult *>(partialResult);
        size_t nColumns = pres->getNumberOfFeatures();
        Argument::set(covariance, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(nColumns, nColumns, data_management::NumericTable::doAllocate)));
        Argument::set(mean, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(nColumns, 1, data_management::NumericTable::doAllocate)));
    }

    /**
     * Returns the final result of the correlation or variance-covariance matrix algorithm
     * \param[in] id   Identifier of the result, \ref ResultId
     * \return Final result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the result of the correlation or variance-covariance matrix algorithm
     * \param[in] id        Identifier of the result
     * \param[in] value     Pointer to the object
     */
    void set(ResultId id, const services::SharedPtr<data_management::NumericTable> &value)
    {
        Argument::set(id, value);
    }

    /**
     * Check correctness of the result
     * \param[in] partialResult     Pointer to the partial result arguments structure
     * \param[in] parameter         Pointer to the structure of the parameters of the algorithm
     * \param[in] method            Computation method
     */
    void check(const daal::algorithms::PartialResult *partialResult, const daal::algorithms::Parameter *parameter,
               int method) const DAAL_C11_OVERRIDE
    {
        const PartialResult *pres = static_cast<const PartialResult *>(partialResult);
        size_t nFeatures = pres->getNumberOfFeatures();

        services::SharedPtr<data_management::NumericTable> covTable = get(covariance);
        if(!covTable) { this->_errors->add(services::ErrorNullOutputNumericTable); return; }
        if(covTable->getNumberOfRows() != nFeatures || covTable->getNumberOfColumns() != nFeatures)
        { this->_errors->add(services::ErrorIncorrectSizeOfOutputNumericTable); return; }

        services::SharedPtr<data_management::NumericTable> meanTable = get(mean);
        if(!meanTable) { this->_errors->add(services::ErrorNullOutputNumericTable); return; }
        if(meanTable->getNumberOfRows() != 1 || meanTable->getNumberOfColumns() != nFeatures)
        { this->_errors->add(services::ErrorIncorrectSizeOfOutputNumericTable); return; }
    }

    /**
     * Check correctness of the result
     * \param[in] input     Pointer to the structure with input objects
     * \param[in] parameter Pointer to the structure of algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
               int method) const DAAL_C11_OVERRIDE
    {
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);

        size_t nFeatures = (static_cast<const InputIface *>(input))->getNumberOfFeatures();

        int unexpectedLayouts = (int)data_management::NumericTableIface::csrArray |
                                (int)data_management::NumericTableIface::upperPackedTriangularMatrix |
                                (int)data_management::NumericTableIface::lowerPackedTriangularMatrix;

        if (algParameter->outputMatrixType == covarianceMatrix)
        {
            /* Check covariance matrix */
            if (!data_management::checkNumericTable(get(covariance).get(), this->_errors.get(),
                strCovariance(), unexpectedLayouts, 0, nFeatures, nFeatures)) { return; }
        }
        else if (algParameter->outputMatrixType == correlationMatrix)
        {
            /* Check correlation matrix */
            if (!data_management::checkNumericTable(get(correlation).get(), this->_errors.get(),
                strCorrelation(), unexpectedLayouts, 0, nFeatures, nFeatures)) { return; }
        }

        unexpectedLayouts |= (int)data_management::NumericTableIface::upperPackedSymmetricMatrix |
                             (int)data_management::NumericTableIface::lowerPackedSymmetricMatrix;

        /* Check mean vector */
        if (!data_management::checkNumericTable(get(mean).get(), this->_errors.get(),
            strMean(), unexpectedLayouts, 0, nFeatures, 1)) { return; }
    }

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_COVARIANCE_RESULT_ID; }

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

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDINPUT"></a>
 * \brief Input parameters of the distributed Covariance algorithm.
 *
 * \tparam step             Step of the distributed computing algorithm, \ref ComputeStep
 */
template<ComputeStep step>
class DistributedInput {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDINPUT"></a>
 * \brief Input parameters of the distributed Covariance algorithm.
 *        Represents inputs of the algorithm on local node.
 */
template<>
class DAAL_EXPORT DistributedInput<step1Local> : public Input
{
public:
    DistributedInput() : Input()
    {}

    virtual ~DistributedInput()
    {}
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__COVARIANCE__DISTRIBUTEDINPUT"></a>
 * \brief Input parameters of the distributed Covariance algorithm.
 *        Represents inputs of the algorithm on master node.
 */
template<>
class DAAL_EXPORT DistributedInput<step2Master> : public InputIface
{
public:
    DistributedInput() : InputIface(1)
    {
        Argument::set(partialResults, services::SharedPtr<data_management::DataCollection>(new data_management::DataCollection()));
    }

    virtual ~DistributedInput()
    {}

    /**
     * Returns number of columns in the input data set
     * \return Number of columns in the input data set
     */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE
    {
        services::SharedPtr<data_management::DataCollection> collectionOfPartialResults =
            services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(partialResults));
        if(collectionOfPartialResults)
        {
            services::SharedPtr<PartialResult> onePartialResult =
                services::staticPointerCast<PartialResult, data_management::SerializationIface>((*collectionOfPartialResults)[0]);
            if(onePartialResult.get() != NULL)
            {
                services::SharedPtr<data_management::NumericTable> ntPtr = onePartialResult->get(sum);
                if(ntPtr)
                {
                    return ntPtr->getNumberOfColumns();
                }
            }
        }
        this->_errors->add(services::ErrorIncorrectSizeOfInputNumericTable);
        return 0;
    }

    /**
     * Adds partial result to the end of DataCollection of input arguments of the Distributed Covariance algorithm
     * \param[in] id            Input arguments's identifier
     * \param[in] partialResult Partial result obtained on the first step of the distributed algorithm
     */
    void add(MasterInputId id, const services::SharedPtr<PartialResult> &partialResult)
    {
        services::SharedPtr<data_management::DataCollection> collection =
            services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
        collection->push_back(services::staticPointerCast<data_management::SerializationIface, PartialResult>(partialResult));
    }

    /**
     * Returns collectionof inputs
     * \param[in] id   Partial result's identifier, \ref MasterInputId
     * \return Collection of distributed inputs
     */
    services::SharedPtr<data_management::DataCollection> get(MasterInputId id) const
    {
        return services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(partialResults));
    }

    /**
     * Check the correctness of DistributedInput<step2Master> object
     * \param[in] parameter Pointer to the structure of the parameters of the algorithm
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        services::SharedPtr<data_management::DataCollection> collectionPtr =
            services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(0));
        size_t nBlocks = collectionPtr->size();
        if(nBlocks == 0)
        { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        for(size_t j = 0; j < nBlocks; j++)
        {
            services::SharedPtr<PartialResult> partialResult =
                services::staticPointerCast<PartialResult, data_management::SerializationIface>((*collectionPtr)[j]);
            /* Check partial number of observations */
            data_management::NumericTable *presTable = static_cast<data_management::NumericTable *>(partialResult->get(nObservations).get());
            if(!presTable) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
            if((presTable->getNumberOfRows() != 1) || (presTable->getNumberOfColumns() != 1))
            { this->_errors->add(services::ErrorIncorrectSizeOfInputNumericTable); return; }

            size_t nFeatures = getNumberOfFeatures();
            if(nFeatures == 0)
            { this->_errors->add(services::ErrorIncorrectSizeOfInputNumericTable); return; }

            /* Check partial cross-products */
            presTable = static_cast<data_management::NumericTable *>(partialResult->get(crossProduct).get());
            if(!presTable) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
            if((presTable->getNumberOfRows() != nFeatures) || (presTable->getNumberOfColumns() != nFeatures))
            { this->_errors->add(services::ErrorIncorrectSizeOfInputNumericTable); return; }

            /* Check partial sums */
            presTable = static_cast<data_management::NumericTable *>(partialResult->get(sum).get());
            if(!presTable) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
            if((presTable->getNumberOfRows() != 1) || (presTable->getNumberOfColumns() != nFeatures))
            { this->_errors->add(services::ErrorIncorrectSizeOfInputNumericTable); return; }
        }
    }
};
} // namespace interface1
using interface1::InputIface;
using interface1::Input;
using interface1::PartialResult;
using interface1::Parameter;
using interface1::PartialResultsInitIface;
using interface1::DefaultPartialResultsInit;
using interface1::OnlineParameter;
using interface1::Result;
using interface1::DistributedInput;

} // namespace daal::algorithms::covariance
}
} // namespace daal
#endif // __COVARIANCE_TYPES_H__
