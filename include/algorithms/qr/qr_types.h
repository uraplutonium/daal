/* file: qr_types.h */
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
//  Definition of QR common types.
//--
*/


#ifndef __QR_TYPES_H__
#define __QR_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
/** \brief Contains classes for computing the results of the QR decomposition algorithm */
namespace qr
{
/**
 * <a name="DAAL-ENUM-QR__METHOD"></a>
 * Available methods for computing the QR decomposition algorithm
 */
enum Method
{
    defaultDense    = 0 /*!< Default method */
};

/**
 * <a name="DAAL-ENUM-QR__INPUTID"></a>
 * Available types of input objects for the QR decomposition algorithm
 */
enum InputId
{
    data = 0      /*!< Input data table */
};

/**
 * <a name="DAAL-ENUM-QR__RESULTID"></a>
 * Available types of results of the QR decomposition algorithm
 */
enum ResultId
{
    matrixQ = 0,   /*!< Orthogonal Matrix Q */
    matrixR = 1    /*!< Upper Triangular Matrix R */
};

/**
 * <a name="DAAL-ENUM-QR__PARTIALRESULTID"></a>
 * Available types of partial results of the QR decomposition algorithm in the online processing mode and of the first step of the
 * QR decomposition algorithm in the distributed processing mode
 */
enum PartialResultId
{
    outputOfStep1ForStep3 = 0,   /*!< Partial results of the QR decomposition algorithms computed on the first step and to be transferred
                                    * to the third step in the distributed processing mode */
    outputOfStep1ForStep2 = 1    /*!< Partial results of the QR decomposition algorithms computed on the first step and to be transferred
                                    * to the second step in the distributed processing mode */
};

/**
 * <a name="DAAL-ENUM-QR__DISTRIBUTEDPARTIALRESULTCOLLECTIONID"></a>
 * Available types of partial results of the second step of the QR decomposition algorithm stored in DataCollection object in the distributed
 * processing mode
 */
enum DistributedPartialResultCollectionId
{
    outputOfStep2ForStep3 = 0    /*!< Partial results of the QR decomposition algorithms to be transferred  to the third step in the distributed
                                    * processing mode */
};

/**
 * <a name="DAAL-ENUM-QR__DISTRIBUTEDPARTIALRESULTID"></a>
 * Available types of partial results of the second step of the QR decomposition algorithm stored in Result object in the distributed processing mode
 */
enum DistributedPartialResultId
{
    finalResultFromStep2Master = 1 /*!< Result object with R matrix */
};

/**
 * <a name="DAAL-ENUM-QR__DISTRIBUTEDPARTIALRESULTSTEP3ID"></a>
 * Available types of partial results of the second step of the QR decomposition algorithm stored in Result object in the distributed processing mode
 */
enum DistributedPartialResultStep3Id
{
    finalResultFromStep3 = 0 /*!< Result object with Q matrix */
};

/**
 * <a name="DAAL-ENUM-QR__MASTERNODEINPUTID"></a>
 * Partial results from the previous steps in the distributed processing mode required by the second distributed step of the algorithm
 */
enum MasterInputId
{
    inputOfStep2FromStep1 = 0  /*!< Partial results of the QR decomposition algorithms computed on the first step and to be transferred  to the
                                  * second step in the distributed processing mode */
};

/**
 * <a name="DAAL-ENUM-QR__FINALIZEONLOCALINPUTID"></a>
 * Partial results from the previous steps in the distributed processing mode required by the third distributed step
 */
enum FinalizeOnLocalInputId
{
    inputOfStep3FromStep1 = 0, /*!< Partial results of the QR decomposition algorithms computed on the first step and to be transferred
                                  * to the third step in the distributed processing mode */
    inputOfStep3FromStep2 = 1  /*!< Partial results of the QR decomposition algorithms computed on the second step and to be transferred
                                  * to the third step in the distributed processing mode */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-QR__INPUT"></a>
 * \brief Input objects for the QR decomposition algorithm in the batch and online processing modes and for the first distributed step of the
 * algorithm.
 */
class Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    Input() : daal::algorithms::Input(1) {}
    /** Default destructor */
    virtual ~Input() {}

    /**
     * Returns input object of the QR decomposition algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets input object for the QR decomposition algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value Pointer to the input object
     */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &value)
    {
        Argument::set(id, value);
    }

    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        services::SharedPtr<data_management::NumericTable> inTable = get(data);
        if(!inTable || inTable->getDataMemoryStatus() == data_management::NumericTable::notAllocated)
        { this->_errors->add(services::ErrorNullInputNumericTable); return; }

        size_t nFeatures = inTable->getNumberOfColumns();
        if(nFeatures == 0) { this->_errors->add(services::ErrorIncorrectSizeOfInputNumericTable); return; }
        if(inTable->getNumberOfRows() == 0) { this->_errors->add(services::ErrorIncorrectSizeOfInputNumericTable); return; }
        if(inTable->getNumberOfRows() < nFeatures) { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }
    }
};

/**
 * <a name="DAAL-CLASS-QR__DISTRIBUTEDSTEP2INPUT"></a>
 * \brief Input objects for the second step of the QR decomposition algorithm in the distributed processing mode.
 */
class DistributedStep2Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedStep2Input() : daal::algorithms::Input(1)
    {
        Argument::set(inputOfStep2FromStep1,
                      services::SharedPtr<data_management::KeyValueDataCollection>(new data_management::KeyValueDataCollection()));
    }

    /**
     * Sets input object for the QR decomposition algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Input object that corresponds to the given identifier
     */
    void set(MasterInputId id, const services::SharedPtr<data_management::KeyValueDataCollection> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Returns input object for the QR decomposition algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::KeyValueDataCollection> get(MasterInputId id) const
    {
        return services::staticPointerCast<data_management::KeyValueDataCollection, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Adds input object to KeyValueDataCollection  of the QR decomposition algorithm
     * \param[in] id    Identifier of input object
     * \param[in] key   Key to use to retrieve data
     * \param[in] value Pointer to the input object value
     */
    void add(MasterInputId id, size_t key, const services::SharedPtr<data_management::DataCollection> &value)
    {
        services::SharedPtr<data_management::KeyValueDataCollection> collection =
            services::staticPointerCast<data_management::KeyValueDataCollection, data_management::SerializationIface>(Argument::get(id));
        (*collection)[key] = value;
    }

    /**
    * Returns the number of blocks in the input data set
    * \return Number of blocks in the input data set
    */
    size_t getNBlocks()
    {
        services::SharedPtr<data_management::KeyValueDataCollection> kvDC = get(inputOfStep2FromStep1);
        size_t nNodes = kvDC->size();
        size_t nBlocks = 0;
        for(size_t i = 0 ; i < nNodes ; i++)
        {
            services::SharedPtr<data_management::DataCollection> nodeCollection =
                services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>((*kvDC).getValueByIndex((int)i));
            size_t nodeSize = nodeCollection->size();
            nBlocks += nodeSize;
        }

        return nBlocks;
    }

    /**
    * Checks parameters of the algorithm
    * \param[in] parameter Pointer to the parameters
    * \param[in] method Computation method
    */
    void check(const daal::algorithms::Parameter *parameter,
               int method) const DAAL_C11_OVERRIDE
    {
        services::SharedPtr<data_management::KeyValueDataCollection> kvDC = get(inputOfStep2FromStep1);
        if(!kvDC) { this->_errors->add(services::ErrorNullInput); return; }

        size_t nNodes = kvDC->size();
        if(nNodes == 0) { this->_errors->add(services::ErrorIncorrectNumberOfElementsInInputCollection); return; }

        size_t m = 0;

        size_t nBlocks = 0;
        for(size_t i = 0 ; i < nNodes ; i++)
        {
            services::SharedPtr<data_management::DataCollection> nodeCollection =
                services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>((*kvDC).getValueByIndex((int)i));
            if(!nodeCollection) { this->_errors->add(services::ErrorNullInputNumericTable);  return; } /* Wrong error? */

            size_t nodeSize = nodeCollection->size();
            if(nodeSize == 0) { this->_errors->add(services::ErrorIncorrectNumberOfElementsInInputCollection); return; }

            for(size_t j = 0 ; j < nodeSize ; j++)
            {
                services::SharedPtr<data_management::NumericTable> rNT =
                    services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>((*nodeCollection)[j]);

                if(!rNT) { this->_errors->add(services::ErrorNullInputNumericTable); return; }

                if(m == 0)
                {
                    m = rNT->getNumberOfColumns();
                    if(m == 0) { this->_errors->add(services::ErrorEmptyInputNumericTable); return; }
                }

                if(m != rNT->getNumberOfColumns() || m != rNT->getNumberOfRows())
                { this->_errors->add(services::ErrorInconsistentNumberOfRows); return; }
            }

            nBlocks += nodeSize;
        }

        if(nBlocks == 0) { this->_errors->add(services::ErrorIncorrectNumberOfElementsInInputCollection); return; }
    }
};

/**
 * <a name="DAAL-CLASS-QR__DISTRIBUTEDSTEP3INPUT"></a>
 * \brief Input objects for the third step of the QR decomposition algorithm in the distributed processing mode
 */
class DistributedStep3Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedStep3Input() : daal::algorithms::Input(2) {}

    /**
     * Returns input object for the QR decomposition algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::DataCollection> get(FinalizeOnLocalInputId id) const
    {
        return services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets input object for the QR decomposition algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value Pointer to the input object value
     */
    void set(FinalizeOnLocalInputId id, const services::SharedPtr<data_management::DataCollection> &value)
    {
        Argument::set(id, value);
    }

    /**
    * Checks parameters of the algorithm
    * \param[in] parameter Pointer to the parameters
    * \param[in] method Computation method
    */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        services::SharedPtr<data_management::DataCollection> qDC = get(inputOfStep3FromStep1);
        services::SharedPtr<data_management::DataCollection> rDC = get(inputOfStep3FromStep2);
        if(!rDC) { this->_errors->add(services::ErrorNullInput); return; }
        if(!qDC) { this->_errors->add(services::ErrorNullInput); return; }

        size_t nodeSize = qDC->size();
        if(nodeSize == 0) { this->_errors->add(services::ErrorIncorrectNumberOfElementsInInputCollection); return; }
        if(nodeSize != rDC->size()) { this->_errors->add(services::ErrorIncorrectNumberOfElementsInInputCollection); return; }

        size_t m = 0;

        for(size_t i = 0 ; i < nodeSize ; i++)
        {
            services::SharedPtr<data_management::NumericTable> qNT =
                services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>((*qDC)[i]);
            services::SharedPtr<data_management::NumericTable> rNT =
                services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>((*rDC)[i]);

            if(!qNT) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
            if(!rNT) { this->_errors->add(services::ErrorNullInputNumericTable); return; }

            if(m == 0)
            {
                m = rNT->getNumberOfColumns();
                if(m == 0) { this->_errors->add(services::ErrorEmptyInputNumericTable); return; }
            }

            if(m != rNT->getNumberOfColumns() || m != rNT->getNumberOfRows() ||
               m != qNT->getNumberOfColumns())
            {
                this->_errors->add(services::ErrorInconsistentNumberOfRows); return;
            }

            if(qNT->getNumberOfRows() == 0) { this->_errors->add(services::ErrorEmptyInputNumericTable); return; }
        }
    }
};

/**
 * <a name="DAAL-CLASS-QR__ONLINEPARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method of the QR decomposition algorithm
 *        in the online processing mode or on the first step of the algorithm in the distributed processing mode
 */
class OnlinePartialResult : public daal::algorithms::PartialResult
{
public:
    /** Default constructor */
    OnlinePartialResult() : daal::algorithms::PartialResult(2) {}
    /** Default destructor */
    virtual ~OnlinePartialResult() {}

    /**
     * Allocates memory for storing partial results of the QR decomposition algorithm
     * \param[in] input     Pointer to input object
     * \param[in] parameter Pointer to parameter
     * \param[in] method    Algorithm method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        set(outputOfStep1ForStep3, services::SharedPtr<data_management::DataCollection>(new data_management::DataCollection()));
        set(outputOfStep1ForStep2, services::SharedPtr<data_management::DataCollection>(new data_management::DataCollection()));
    }

    /**
     * Allocates additional memory for storing partial results of the QR decomposition algorithm for each subsequent call to compute method
     * \tparam     algorithmFPType  Data type to be used for storage in resulting HomogenNumericTable
     * \param[in]  m  Number of columns in the input data set
     * \param[in]  n  Number of rows in the input data set
     */
    template <typename algorithmFPType>
    void addPartialResultStorage(size_t m, size_t n)
    {
        services::SharedPtr<data_management::DataCollection> qCollection = get(outputOfStep1ForStep3);
        services::SharedPtr<data_management::DataCollection> rCollection = get(outputOfStep1ForStep2);

        qCollection->push_back(services::SharedPtr<data_management::SerializationIface>(
                                   new data_management::HomogenNumericTable<algorithmFPType>(m, n, data_management::NumericTable::doAllocate)));
        rCollection->push_back(services::SharedPtr<data_management::SerializationIface>(
                                   new data_management::HomogenNumericTable<algorithmFPType>(m, m, data_management::NumericTable::doAllocate)));
    }

    /**
     * Returns partial result of the QR decomposition algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Partial result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::DataCollection> get(PartialResultId id) const
    {
        return services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    }

    /**
    * Sets an input object for the QR decomposition algorithm
    * \param[in] id    Identifier of partial result
    * \param[in] value Pointer to the partial result
    */
    void set(PartialResultId id, const services::SharedPtr<data_management::DataCollection> &value)
    {
        Argument::set(id, value);
    }

    /**
    * Checks parameters of the algorithm
    * \param[in] input Input of the algorithm
    * \param[in] parameter Pointer to the parameters
    * \param[in] method Computation method
    */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
    {
        if(Argument::size() != 2)
        { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        services::SharedPtr<data_management::DataCollection> s3Collection = get(outputOfStep1ForStep3);
        services::SharedPtr<data_management::DataCollection> s2Collection = get(outputOfStep1ForStep2);
        if(s2Collection.get()==0 || s3Collection.get()==0) { this->_errors->add(services::ErrorNullOutputDataCollection); return; }

        if(s2Collection->size() != s3Collection->size()) { this->_errors->add(services::ErrorIncorrectNumberOfElementsInResultCollection); return; }

        size_t dcSize = s2Collection->size();

        for(size_t i=0;i<dcSize;i++)
        {
            services::SharedPtr<data_management::SerializationIface> s3si = (*s3Collection)[i];
            services::SharedPtr<data_management::SerializationIface> s2si = (*s2Collection)[i];

            if(!s2si || !s3si) { this->_errors->add(services::ErrorNullOutputNumericTable); return; }

            services::SharedPtr<data_management::NumericTable> s3nt =
                services::dynamicPointerCast<data_management::NumericTable, data_management::SerializationIface>(s3si);
            services::SharedPtr<data_management::NumericTable> s2nt =
                services::dynamicPointerCast<data_management::NumericTable, data_management::SerializationIface>(s2si);

            if(!s2nt || !s3nt) { this->_errors->add(services::ErrorIncorrectClassLabels); return; }
        }

    }

    /**
    * Checks parameters of the algorithm
    * \param[in] parameter Pointer to the parameters
    * \param[in] method Computation method
    */
    void check(const daal::algorithms::Parameter *parameter, int method) const
    {
        if(Argument::size() != 2)
        { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }
    }

    /**
     * Returns the number of columns in the input data set
     * \return Number of columns in the input data set
     */

    size_t getNumberOfColumns() const
    {
        data_management::DataCollection *rCollection = get(outputOfStep1ForStep2).get();
        return static_cast<data_management::NumericTable *>((*rCollection)[0].get())->getNumberOfColumns();
    }

    /**
    * Returns the number of rows in the input data set
    * \return Number of rows in the input data set
    */
    size_t getNumberOfRows() const
    {
        data_management::DataCollection *qCollection = get(outputOfStep1ForStep3).get();
        size_t np = qCollection->size();

        size_t n = 0;
        if(1 /* we need V matrices */)
        {
            for(size_t i = 0; i < np; i++)
            {
                n += static_cast<data_management::NumericTable *>((*qCollection)[i].get())->getNumberOfRows();
            }
        }

        return n;
    }

    int getSerializationTag() { return SERIALIZATION_QR_ONLINE_PARTIAL_RESULT_ID; }

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
        daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
 * <a name="DAAL-CLASS-QR__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the QR decomposition algorithm
 *        in the batch processing mode or finalizeCompute() method of algorithm in the online processing mode
 *        or on the second and third steps of the algorithm in the distributed processing mode
 */
class Result : public daal::algorithms::Result
{
public:
    /** Default constructor */
    Result() : daal::algorithms::Result(2) {}
    /** Default destructor */
    virtual ~Result() {}

    /**
     * Returns the result of the QR decomposition algorithm
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Allocates memory for storing final results of the QR decomposition algorithm
     * \param[in] input     Pointer to input object
     * \param[in] parameter Pointer to parameter
     * \param[in] method    Algorithm method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const Input *in = static_cast<const Input *>(input);
        allocateImpl<algorithmFPType>(in->get(data)->getNumberOfColumns(), in->get(data)->getNumberOfRows());
    }

    /**
     * Allocates memory for storing final results of the QR decomposition algorithm
     * \param[in] partialResult  Pointer to partial result
     * \param[in] parameter      Pointer to the result
     * \param[in] method         Algorithm method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::PartialResult *partialResult, daal::algorithms::Parameter *parameter,
                  const int method)
    {
        const OnlinePartialResult *in = static_cast<const OnlinePartialResult *>(partialResult);
        allocateImpl<algorithmFPType>(in->getNumberOfColumns(), in->getNumberOfRows());
    }

    /**
    * Sets an input object for the QR decomposition algorithm
    * \param[in] id    Identifier of the result
    * \param[in] value Pointer to the result
    */
    void set(ResultId id, const services::SharedPtr<data_management::NumericTable> &value)
    {
        Argument::set(id, value);
    }

    /**
    * Checks parameters of the algorithm
    * \param[in] input Input of the algorithm
    * \param[in] par Pointer to the parameters
    * \param[in] method Computation method
    */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
               int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 2)
        { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));

        size_t m = algInput->get(data)->getNumberOfColumns();
        size_t n = algInput->get(data)->getNumberOfRows();

        services::SharedPtr<data_management::NumericTable> q = get(matrixQ);
        if(q.get() == 0  || q->getDataMemoryStatus() == data_management::NumericTable::notAllocated)
        { this->_errors->add(services::ErrorNullOutputNumericTable); return; }
        if(q->getNumberOfRows() != n)    { this->_errors->add(services::ErrorIncorrectNumberOfRowsInOutputNumericTable); return; }
        if(q->getNumberOfColumns() != m) { this->_errors->add(services::ErrorIncorrectNumberOfColumnsInOutputNumericTable); return; }

        services::SharedPtr<data_management::NumericTable> r = get(matrixR);
        if(r.get() == 0 || r->getDataMemoryStatus() == data_management::NumericTable::notAllocated)
        { this->_errors->add(services::ErrorNullOutputNumericTable); return; }
        if(r->getNumberOfRows() != m)    { this->_errors->add(services::ErrorIncorrectNumberOfRowsInOutputNumericTable); return; }
        if(r->getNumberOfColumns() != m) { this->_errors->add(services::ErrorIncorrectNumberOfColumnsInOutputNumericTable); return; }
    }

    /**
    * Checks the result of the QR decomposition algorithm
    * \param[in] pres    Partial result of the algorithm
    * \param[in] par     Algorithm %parameter
    * \param[in] method  Computation method
    */
    void check(const daal::algorithms::PartialResult *pres, const daal::algorithms::Parameter *par, int method) const
    {
        if(Argument::size() != 2)
        { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }
    }

    /**
     * Allocates memory for storing final results of the QR decomposition algorithm
     * \tparam     algorithmFPType  Data type to be used for storage in resulting HomogenNumericTable
     * \param[in]  m  Number of columns in the input data set
     * \param[in]  n  Number of rows in the input data set
     */
    template <typename algorithmFPType>
    void allocateImpl(size_t m, size_t n)
    {
        if(n == 0)
        {
            Argument::set(matrixQ, services::SharedPtr<data_management::SerializationIface>());
        }
        else
        {
            Argument::set(matrixQ, services::SharedPtr<data_management::SerializationIface>(
                              new data_management::HomogenNumericTable<algorithmFPType>(m, n, data_management::NumericTable::doAllocate)));
        }
        Argument::set(matrixR, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(m, m, data_management::NumericTable::doAllocate)));
    }

    int getSerializationTag() { return SERIALIZATION_QR_RESULT_ID; }

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
 * <a name="DAAL-CLASS-QR__DISTRIBUTEDPARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method of the
 *        second step of the QR decomposition algorithm in the distributed processing mode
 */
class DistributedPartialResult : public daal::algorithms::PartialResult
{
public:
    /** Default constructor */
    DistributedPartialResult() : daal::algorithms::PartialResult(2) {}
    /** Default destructor */
    virtual ~DistributedPartialResult() {}

    /**
     * Allocates memory for storing partial results of the QR decomposition algorithm
     * \param[in] input  Pointer to input object
     * \param[in] parameter    Pointer to parameter
     * \param[in] method Computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        Argument::set(outputOfStep2ForStep3,
                      services::SharedPtr<data_management::KeyValueDataCollection>(new data_management::KeyValueDataCollection()));
        Argument::set(finalResultFromStep2Master, services::SharedPtr<Result>(new Result()));

        services::SharedPtr<data_management::KeyValueDataCollection> inCollection = static_cast<const DistributedStep2Input *>(input)->get(
                                                                                        inputOfStep2FromStep1);

        size_t nBlocks = 0;
        setPartialResultStorage<algorithmFPType>(inCollection.get(), nBlocks);
    }

    /**
     * Allocates memory for storing partial results of the QR decomposition algorithm based on known structure of partial results from the
     * first steps of the algorithm in the distributed processing mode.
     * KeyValueDataCollection under outputOfStep2ForStep3 is structured the same as KeyValueDataCollection under
     * inputOfStep2FromStep1 id of the algorithm input
     * \tparam     algorithmFPType             Data type to be used for storage in resulting HomogenNumericTable
     * \param[in]  inCollection  KeyValueDataCollection of all partial results from the first steps of the algorithm in the distributed
     * processing mode
     * \param[out] nBlocks  Number of rows in the input data set
     */
    template <typename algorithmFPType>
    void setPartialResultStorage(data_management::KeyValueDataCollection *inCollection, size_t &nBlocks)
    {
        services::SharedPtr<data_management::KeyValueDataCollection> partialCollection =
            services::staticPointerCast<data_management::KeyValueDataCollection, data_management::SerializationIface>(Argument::get(
                                                                                                                          outputOfStep2ForStep3));
        services::SharedPtr<Result> result =
            services::staticPointerCast<Result, data_management::SerializationIface>(Argument::get(finalResultFromStep2Master));

        size_t inSize = inCollection->size();

        data_management::DataCollection *fisrtNodeCollection =
            static_cast<data_management::DataCollection *>((*inCollection).getValueByIndex(0).get());
        data_management::NumericTable   *firstNumericTable   = static_cast<data_management::NumericTable *>((*fisrtNodeCollection)[0].get());

        size_t m = firstNumericTable->getNumberOfColumns();

        if(result->get(matrixR).get() == NULL)
        {
            result->allocateImpl<algorithmFPType>(m, 0);
        }

        nBlocks = 0;
        for(size_t i = 0 ; i < inSize ; i++)
        {
            data_management::DataCollection   *nodeCollection
                = static_cast<data_management::DataCollection *>((*inCollection).getValueByIndex((int)i).get());
            size_t            nodeKey        = (*inCollection).getKeyByIndex((int)i);
            size_t nodeSize = nodeCollection->size();
            nBlocks += nodeSize;

            services::SharedPtr<data_management::DataCollection> nodePartialResult(new data_management::DataCollection());

            for(size_t j = 0 ; j < nodeSize ; j++)
            {
                nodePartialResult->push_back(
                    services::SharedPtr<data_management::SerializationIface>(
                        new data_management::HomogenNumericTable<algorithmFPType>(m, m, data_management::NumericTable::doAllocate)));
            }

            (*partialCollection)[ nodeKey ] = nodePartialResult;
        }
    }

    /**
     * Returns partial result of the QR decomposition algorithm.
     * KeyValueDataCollection under outputOfStep2ForStep3 id is structured the same as KeyValueDataCollection under
     * inputOfStep2FromStep1 id of the algorithm input
     * \param[in] id    Identifier of the partial result
     * \return          Partial result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::KeyValueDataCollection> get(DistributedPartialResultCollectionId id) const
    {
        return services::staticPointerCast<data_management::KeyValueDataCollection, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Returns the result of the QR decomposition algorithm with the matrix R calculated
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    services::SharedPtr<Result> get(DistributedPartialResultId id) const
    {
        return services::staticPointerCast<Result, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets KeyValueDataCollection to store partial result of the QR decomposition algorithm
     * \param[in] id    Identifier of partial result
     * \param[in] value Pointer to the Result object
     */
    void set(DistributedPartialResultCollectionId id, const services::SharedPtr<data_management::KeyValueDataCollection> &value)
    {
        Argument::set(id, value);
    }

    /**
     * Sets Result object to store the result of the QR decomposition algorithm
     * \param[in] id    Identifier of the result
     * \param[in] value Pointer to the Result object
     */
    void set(DistributedPartialResultId id, const services::SharedPtr<Result> &value)
    {
        Argument::set(id, value);
    }

    /**
    * Checks parameters of the algorithm
    * \param[in] parameter Pointer to the parameters
    * \param[in] method Computation method
    */
    void check(const daal::algorithms::Parameter *parameter, int method) const
    {
        if(Argument::size() != 2)
        { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }
    }

    int getSerializationTag() { return SERIALIZATION_QR_DISTRIBUTED_PARTIAL_RESULT_ID; }

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
        daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
 * <a name="DAAL-CLASS-QR__DISTRIBUTEDPARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method of the third step of the QR decomposition algorithm
 *        in the distributed processing mode
 */
class DistributedPartialResultStep3 : public daal::algorithms::PartialResult
{
public:
    /** Default constructor */
    DistributedPartialResultStep3() : daal::algorithms::PartialResult(1) {}
    /** Default destructor */
    virtual ~DistributedPartialResultStep3() {}

    /**
     * Allocates memory for storing partial results of the QR decomposition algorithm
     * \param[in] input     Pointer to input object
     * \param[in] parameter Pointer to parameter
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        Argument::set(finalResultFromStep3, services::SharedPtr<Result>(new Result()));
    }

    /**
     * Allocates memory for storing partial results of the third step of the QR decomposition algorithm in the distributed processing mode
     * \tparam     algorithmFPType            Data type to be used for storage in resulting HomogenNumericTable
     * \param[in]  qCollection  KeyValueDataCollection of all partial results from the first steps of the algorithm in the distributed
     * processing mode
     */
    template <typename algorithmFPType>
    void setPartialResultStorage(data_management::DataCollection *qCollection)
    {
        size_t qSize = qCollection->size();
        size_t m = 0;
        size_t n = 0;
        for(size_t i = 0 ; i < qSize ; i++)
        {
            data_management::NumericTable  *qNT = static_cast<data_management::NumericTable *>((*qCollection)[i].get());
            m  = qNT->getNumberOfColumns();
            n += qNT->getNumberOfRows();
        }
        services::SharedPtr<Result> result =
            services::staticPointerCast<Result, data_management::SerializationIface>(Argument::get(finalResultFromStep3));

        result->allocateImpl<algorithmFPType>(m, n);
    }

    /**
     * Returns the result of the QR decomposition algorithm with the matrix Q calculated
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    services::SharedPtr<Result> get(DistributedPartialResultStep3Id id) const
    {
        return services::staticPointerCast<Result, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets Result object to store the result of the QR decomposition algorithm
     * \param[in] id    Identifier of the result
     * \param[in] value Pointer to the Result object
     */
    void set(DistributedPartialResultStep3Id id, const services::SharedPtr<Result> &value)
    {
        Argument::set(id, services::staticPointerCast<data_management::SerializationIface, Result>(value));
    }

    /**
    * Checks parameters of the algorithm
    * \param[in] parameter Pointer to the parameters
    * \param[in] method Computation method
    */
    void check(const daal::algorithms::Parameter *parameter, int method) const
    {
        if(Argument::size() != 1)
        { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }
    }

    int getSerializationTag() { return SERIALIZATION_QR_DISTRIBUTED_PARTIAL_RESULT_STEP3_ID; }

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
        daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
 * <a name="DAAL-CLASS-QRPARAMETERS"></a>
 * \brief Parameters for the QR decomposition compute method
 */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /**
     *  Default constructor
     */
    Parameter() {}
};
} // namespace interface1
using interface1::Input;
using interface1::DistributedStep2Input;
using interface1::DistributedStep3Input;
using interface1::OnlinePartialResult;
using interface1::Result;
using interface1::DistributedPartialResult;
using interface1::DistributedPartialResultStep3;
using interface1::Parameter;

} // namespace daal::algorithms::qr
} // namespace daal::algorithms
} // namespace daal
#endif
