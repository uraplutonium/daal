/* file: svd_types.h */
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
//  Definition of the SVD common types.
//--
*/

#ifndef __SVD_TYPES_H__
#define __SVD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
/** \brief Contains classes to run the singular-value decomposition (SVD) algorithm */
namespace svd
{
/**
 * <a name="DAAL-ENUM-SVD__METHOD"></a>
 * Available methods to compute results of the SVD algorithm
 */
enum Method
{
    defaultDense    = 0 /*!< Default method */
};

/**
 * <a name="DAAL-ENUM-SVD-SVDRESULTFORMAT"></a>
 * Available options to return result matrices
 */
enum SVDResultFormat
{
    notRequired = 0,          /*!< Matrix is not required */
    requiredInPackedForm = 1  /*!< Matrix in the packed format is required */
};

/**
 * <a name="DAAL-ENUM-SVD__INPUTID"></a>
 * \brief Available types of input objects for the SVD algorithm
 */
enum InputId
{
    data = 0      /*!< %Input data table */
};

/**
 * <a name="DAAL-ENUM-SVD__RESULTID"></a>
 * \brief Available types of results of the SVD algorithm
 */
enum ResultId
{
    singularValues      = 0, /*!< Singular values         */
    leftSingularMatrix  = 1, /*!< Left orthogonal matrix  */
    rightSingularMatrix = 2  /*!< Right orthogonal matrix */
};

/**
 * <a name="DAAL-ENUM-SVD__PARTIALRESULTID"></a>
 * \brief Available types of partial results of the SVD algorithm obtained in the online processing mode and in the first step in the
 * distributed processing mode
 */
enum PartialResultId
{
    outputOfStep1ForStep3 = 0,   /*!< DataCollection with data computed in the first step to be transferred to the third step in the distributed
                                    * processing mode */
    outputOfStep1ForStep2 = 1    /*!< DataCollection with data computed in the first step to be transferred to the second step in the distributed
                                    * processing mode  */
};

/**
 * <a name="DAAL-ENUM-SVD__DISTRIBUTEDPARTIALRESULTCOLLECTIONID"></a>
 * \brief Available types of partial results obtained in the second step of the SVD algorithm in the distributed processing mode, stored in the
 * DataCollection object
 */
enum DistributedPartialResultCollectionId
{
    outputOfStep2ForStep3 = 0    /*!< DataCollection with data to be transferred to the third step in the distributed processing mode */
};

/**
 * <a name="DAAL-ENUM-SVD__DISTRIBUTEDPARTIALRESULTID"></a>
 * \brief Available types of partial results obtained in the second step of the SVD algorithm in the distributed processing mode, stored in the
 *  Result object
 */
enum DistributedPartialResultId
{
    finalResultFromStep2Master = 1 /*!< Result object with singular values and the right orthogonal matrix */
};

/**
 * <a name="DAAL-ENUM-SVD__DISTRIBUTEDPARTIALRESULTSTEP3ID"></a>
 * \brief Available types of partial results obtained in the third step of the SVD algorithm in the distributed processing mode, stored in the
 * Result object
 */
enum DistributedPartialResultStep3Id
{
    finalResultFromStep3 = 0 /*!< Result object with singular values and the left orthogonal matrix */
};

/**
 * <a name="DAAL-ENUM-SVD__MASTERINPUTID"></a>
 * \brief Partial results from previous steps in the distributed processing mode, required by the second step
 */
enum MasterInputId
{
    inputOfStep2FromStep1 = 0  /*!< DataCollection with data transferred from the first step to the second step in the distributed processing mode*/
};

/**
 * <a name="DAAL-ENUM-SVD__FINALIZEONLOCALINPUTID"></a>
 * \brief Partial results from previous steps in the distributed processing mode, required by the third step
 */
enum FinalizeOnLocalInputId
{
    inputOfStep3FromStep1 = 0, /*!< DataCollection with data transferred from the first step to the third step in the distributed processing mode */
    inputOfStep3FromStep2 = 1  /*!< DataCollection with data transferred from the second step to the third step in the distributed processing mode */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-SVDPARAMETERS"></a>
 * \brief Parameters for the computation method of the SVD algorithm
 */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /**
     *  Default constructor
     *  \param[in] _leftSingularMatrix  Format of the matrix of left singular vectors
     *  \param[in] _rightSingularMatrix Format of the matrix of right singular vectors
     */
    Parameter(SVDResultFormat _leftSingularMatrix  = requiredInPackedForm,
              SVDResultFormat _rightSingularMatrix = requiredInPackedForm) :
        leftSingularMatrix(_leftSingularMatrix), rightSingularMatrix(_rightSingularMatrix) {}

    SVDResultFormat leftSingularMatrix;  /*!< Format of the matrix of left singular vectors  >*/
    SVDResultFormat rightSingularMatrix; /*!< Format of the matrix of right singular vectors >*/
};

/**
 * <a name="DAAL-CLASS-SVD__INPUT"></a>
 * \brief Input objects for the SVD algorithm in the batch processing and online processing modes, and the first step in the distributed
 * processing mode
 */
class Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    Input() : daal::algorithms::Input(1) {}
    /** Default destructor */
    virtual ~Input() {}

    /**
     * Returns an input object for the SVD algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets an input object for the SVD algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value Pointer to the new input object value
     */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &value)
    {
        Argument::set(id, value);
    }

    /**
     * Checks parameters of the algorithm
     * \param[in] parameter Pointer to the parameters
    * \param[in] method Computation method
    */

    void check(const daal::algorithms::Parameter *parameter,
               int method) const DAAL_C11_OVERRIDE
    {
        services::SharedPtr<data_management::NumericTable> inTable = get(data);
        if(!inTable) { this->_errors->add(services::ErrorNullInputNumericTable); return; }
        size_t nFeatures = inTable->getNumberOfColumns();
        if(nFeatures == 0) { this->_errors->add(services::ErrorIncorrectSizeOfInputNumericTable); return; }
        if(inTable->getNumberOfRows() == 0) { this->_errors->add(services::ErrorIncorrectSizeOfInputNumericTable); return; }
    }
};

/**
 * <a name="DAAL-CLASS-SVD__DISTRIBUTEDSTEP2INPUT"></a>
 * \brief %Input objects for the second step of  the SVD algorithm in the distributed processing mode
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
     * Sets an input object for the SVD algorithm
     * \param[in] id   Identifier of the input object
     * \param[in] ptr  Input object that corresponds to the given identifier
     */
    void set(MasterInputId id, const services::SharedPtr<data_management::KeyValueDataCollection> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
     * Returns an input object for the SVD algorithm
     * \param[in] id   Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::KeyValueDataCollection> get(MasterInputId id) const
    {
        return services::staticPointerCast<data_management::KeyValueDataCollection, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Adds the value to KeyValueDataCollection of the input object for the SVD algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] key   Key to use to retrieve data
     * \param[in] value Pointer to the new input object value
     */
    void add(MasterInputId id, size_t key, const services::SharedPtr<data_management::DataCollection> &value)
    {
        services::SharedPtr<data_management::KeyValueDataCollection> collection =
            services::staticPointerCast<data_management::KeyValueDataCollection, data_management::SerializationIface>(Argument::get(id));
        (*collection)[key] = value;
    }

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
            if(!nodeCollection) { this->_errors->add(services::ErrorNullInputNumericTable); /* Wrong error? */ return; }

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
 * <a name="DAAL-CLASS-SVD__DISTRIBUTEDSTEP3INPUT"></a>
 * \brief %Input objects for the third step of the SVD algorithm in the distributed processing mode
 */
class DistributedStep3Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedStep3Input() : daal::algorithms::Input(2) {}

    /**
     * Returns an input object for the SVD algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    services::SharedPtr<data_management::DataCollection> get(FinalizeOnLocalInputId id) const
    {
        return services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets an input object for the SVD algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value Pointer to the new input object value
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

    void check(const daal::algorithms::Parameter *parameter,
               int method) const DAAL_C11_OVERRIDE
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
               m != qNT->getNumberOfColumns()) { this->_errors->add(services::ErrorInconsistentNumberOfRows); return; }

            if(qNT->getNumberOfRows() == 0) { this->_errors->add(services::ErrorEmptyInputNumericTable); return; }
        }
    }
};

/**
 * <a name="DAAL-CLASS-SVD__ONLINEPARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method of  the SVD algorithm in the online processing mode or
 * the first step in the distributed processing mode
 */
class OnlinePartialResult : public daal::algorithms::PartialResult
{
public:
    /** Default constructor */
    OnlinePartialResult() : daal::algorithms::PartialResult(2) {}
    /** Default destructor */
    virtual ~OnlinePartialResult() {}

    /**
     * Allocates memory to store final results of the SVD algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        Argument::set(outputOfStep1ForStep3, services::SharedPtr<data_management::DataCollection>(new data_management::DataCollection()));
        Argument::set(outputOfStep1ForStep2, services::SharedPtr<data_management::DataCollection>(new data_management::DataCollection()));
    }

    /**
     * Allocates additional memory to store partial results of the SVD algorithm for each subsequent compute() method
     * \tparam     algorithmFPType    Data type to use for storage in the resulting HomogenNumericTable
     * \param[in]  m    Number of columns in the input data set
     * \param[in]  n    Number of rows in the input data set
     * \param[in]  par  Reference to the object with the algorithm parameters
     */
    template <typename algorithmFPType>
    void addPartialResultStorage(size_t m, size_t n, Parameter &par)
    {
        services::SharedPtr<data_management::DataCollection> rCollection =
            services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(outputOfStep1ForStep2));
        rCollection->push_back(services::SharedPtr<data_management::SerializationIface>(
                                   new data_management::HomogenNumericTable<algorithmFPType>(m, m, data_management::NumericTable::doAllocate)));

        if(par.leftSingularMatrix != notRequired)
        {
            services::SharedPtr<data_management::DataCollection> qCollection =
                services::staticPointerCast<data_management::DataCollection,
                data_management::SerializationIface>(Argument::get(outputOfStep1ForStep3));
            qCollection->push_back(services::SharedPtr<data_management::SerializationIface>(
                                       new data_management::HomogenNumericTable<algorithmFPType>(m, n, data_management::NumericTable::doAllocate)));
        }
    }

    /**
     * Returns partial results of the SVD algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    services::SharedPtr<data_management::DataCollection> get(PartialResultId id) const
    {
        return services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets partial results of the SVD algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] value Value that corresponds to the given identifier
     */
    void set(PartialResultId id, const services::SharedPtr<data_management::DataCollection> &value)
    {
        Argument::set(id, value);
    }

    /**
     * Checks partial results of the algorithm
     * \param[in] parameter Pointer to parameters
    * \param[in] method Computation method
    */

    void check(const daal::algorithms::Parameter *parameter, int method) const
    {
        if(Argument::size() != 2)
        { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }
    }

    size_t getNumberOfColumns() const
    {
        data_management::DataCollection *rCollection = get(outputOfStep1ForStep2).get();
        return static_cast<data_management::NumericTable *>((*rCollection)[0].get())->getNumberOfColumns();
    }

    size_t getNumberOfRows() const
    {
        data_management::DataCollection *qCollection = get(outputOfStep1ForStep3).get();
        size_t np = qCollection->size();

        size_t n = 0;

        for(size_t i = 0; i < np; i++)
        {
            n += static_cast<data_management::NumericTable *>((*qCollection)[i].get())->getNumberOfRows();
        }

        return n;
    }

    int getSerializationTag() { return SERIALIZATION_SVD_ONLINE_PARTIAL_RESULT_ID; }

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
 * <a name="DAAL-CLASS-SVD__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the SVD algorithm in the batch processing mode
 *        or with the finalizeCompute() method in the online processing mode or steps 2 and 3 in the distributed processing mode
 */
class Result : public daal::algorithms::Result
{
public:
    /** Default constructor */
    Result() : daal::algorithms::Result(3) {}
    /** Default destructor */
    virtual ~Result() {}

    /**
     * Returns a result of the SVD algorithm
     * \param[in] id    Identifier of the parameter
     * \return          Parameter that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Allocates memory to store final results of the SVD algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const Input *in = static_cast<const Input *>(input);
        allocateImpl<algorithmFPType>(in->get(data)->getNumberOfColumns(), in->get(data)->getNumberOfRows());
    }

    /**
     * Allocates memory to store final results of the SVD algorithm
     * \param[in] partialResult  Pointer to the partial result
     * \param[in] parameter      Pointer to the parameter
     * \param[in] method         Algorithm computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::PartialResult *partialResult, daal::algorithms::Parameter *parameter,
                  const int method)
    {
        const OnlinePartialResult *in = static_cast<const OnlinePartialResult *>(partialResult);
        allocateImpl<algorithmFPType>(in->getNumberOfColumns(), in->getNumberOfRows());
    }

    /**
     * Sets the final result of the SVD algorithm
     * \param[in] id    Identifier of the final result
     * \param[in] value Value that corresponds to the given identifier
     */
    void set(ResultId id, const services::SharedPtr<data_management::NumericTable> &value)
    {
        Argument::set(id, value);
    }

    /**
       * Checks final results of the algorithm
      * \param[in] input  Pointer to input objects
      * \param[in] par    Pointer to parameters
      * \param[in] method Computation method
      */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
               int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 3)
        { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        Input     *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
        Parameter *svdPar   = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par  ));

        size_t m = algInput->get(data)->getNumberOfColumns();
        size_t n = algInput->get(data)->getNumberOfRows();

        services::SharedPtr<data_management::NumericTable> s = get(singularValues);
        if(s.get() == 0)                 { this->_errors->add(services::ErrorNullOutputNumericTable); return;      }
        if(s->getNumberOfRows() != 1)    { this->_errors->add(services::ErrorInconsistentNumberOfRows); return;    }
        if(s->getNumberOfColumns() != m) { this->_errors->add(services::ErrorInconsistentNumberOfColumns); return; }

        if(svdPar->rightSingularMatrix == requiredInPackedForm)
        {
            services::SharedPtr<data_management::NumericTable> r = get(rightSingularMatrix);
            if(r.get() == 0)                 { this->_errors->add(services::ErrorNullOutputNumericTable); return;      }
            if(r->getNumberOfRows() != m)    { this->_errors->add(services::ErrorInconsistentNumberOfRows); return;    }
            if(r->getNumberOfColumns() != m) { this->_errors->add(services::ErrorInconsistentNumberOfColumns); return; }
        }

        if(svdPar->leftSingularMatrix == requiredInPackedForm)
        {
            services::SharedPtr<data_management::NumericTable> l = get(leftSingularMatrix);
            if(l.get() == 0)                 { this->_errors->add(services::ErrorNullOutputNumericTable); return;      }
            if(l->getNumberOfRows() != n)    { this->_errors->add(services::ErrorInconsistentNumberOfRows); return;    }
            if(l->getNumberOfColumns() != m) { this->_errors->add(services::ErrorInconsistentNumberOfColumns); return; }
        }
    }

    /**
    * Checks the result parameter of the SVD algorithm
    * \param[in] pres    Partial result of the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method
    */
    void check(const daal::algorithms::PartialResult *pres, const daal::algorithms::Parameter *par, int method) const
    {
        if(Argument::size() != 3)
        { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }
    }

    /**
     * Allocates memory to store final results of the SVD algorithm
     * \tparam     algorithmFPType  Data type to use for storage in the resulting HomogenNumericTable
     * \param[in]  m  Number of columns in the input data set
     * \param[in]  n  Number of rows in the input data set
     */
    template <typename algorithmFPType>
    void allocateImpl(size_t m, size_t n)
    {
        Argument::set(singularValues, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(m, 1, data_management::NumericTable::doAllocate)));
        Argument::set(rightSingularMatrix, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(m, m, data_management::NumericTable::doAllocate)));
        if(n != 0)
        {
            Argument::set(leftSingularMatrix, services::SharedPtr<data_management::SerializationIface>(
                              new data_management::HomogenNumericTable<algorithmFPType>(m, n, data_management::NumericTable::doAllocate)));
        }
    }

    int getSerializationTag() { return SERIALIZATION_SVD_RESULT_ID; }

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
 * <a name="DAAL-CLASS-SVD__DISTRIBUTEDPARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method of the SVD algorithm in the second step in the
 * distributed processing mode
 */
class DistributedPartialResult : public daal::algorithms::PartialResult
{
public:
    /** Default constructor */
    DistributedPartialResult() : daal::algorithms::PartialResult(2) {}
    /** Default destructor */
    virtual ~DistributedPartialResult() {}

    /**
     * Allocates memory to store partial results of the SVD algorithm
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
     * Allocates memory to store partial results of the SVD algorithm based on the known structure of partial results from step 1 in the
     * distributed processing mode.
     * KeyValueDataCollection under outputOfStep2ForStep3 id is structured the same as KeyValueDataCollection under
     * inputOfStep2FromStep1 id of the algorithm input
     * \tparam     algorithmFPType Data type to use for storage in the resulting HomogenNumericTable
     * \param[in]  inCollection    KeyValueDataCollection of all partial results from the first step of  the SVD algorithm in the distributed
     *                             processing mode
     * \param[out] nBlocks         Number of rows in the input data set
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
        data_management::NumericTable *fisrtNumericTable
            = static_cast<data_management::NumericTable *>((*fisrtNodeCollection)[0].get());

        size_t m = fisrtNumericTable->getNumberOfColumns();

        if(result->get(singularValues).get() == NULL)
        {
            result->allocateImpl<algorithmFPType>(m, 0);
        }

        nBlocks = 0;
        for(size_t i = 0 ; i < inSize ; i++)
        {
            data_management::DataCollection   *nodeCollection =
                static_cast<data_management::DataCollection *>((*inCollection).getValueByIndex((int)i).get());
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
     * Returns partial results of the SVD algorithm.
     * KeyValueDataCollection under outputOfStep2ForStep3 id is structured the same as KeyValueDataCollection under
     * inputOfStep2FromStep1 id of the algorithm input
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    services::SharedPtr<data_management::KeyValueDataCollection> get(DistributedPartialResultCollectionId id) const
    {
        return services::staticPointerCast<data_management::KeyValueDataCollection, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Returns results of the SVD algorithm with singular values and the left orthogonal matrix calculated
     * \param[in] id    Identifier of the parameter
     * \return          Parameter that corresponds to the given identifier
     */
    services::SharedPtr<Result> get(DistributedPartialResultId id) const
    {
        return services::staticPointerCast<Result, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets KeyValueDataCollection to store partial results of the SVD algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] value Value that corresponds to the given identifier
     */
    void set(DistributedPartialResultCollectionId id, const services::SharedPtr<data_management::KeyValueDataCollection> &value)
    {
        Argument::set(id, value);
    }

    /**
     * Sets the Result object to store results of the SVD algorithm
     * \param[in] id    Identifier of the parameter
     * \param[in] value Pointer to the Result object
     */
    void set(DistributedPartialResultId id, const services::SharedPtr<Result> &value)
    {
        Argument::set(id, value);
    }

    /**
     * Checks partial results of the algorithm
     * \param[in] parameter Pointer to parameters
    * \param[in] method Computation method
    */

    void check(const daal::algorithms::Parameter *parameter, int method) const
    {
        if(Argument::size() != 2)
        { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }
    }

    int getSerializationTag() { return SERIALIZATION_SVD_DISTRIBUTED_PARTIAL_RESULT_ID; }

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
 * <a name="DAAL-CLASS-SVD__DISTRIBUTEDPARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method of the SVD algorithm
 *        in the third step in the distributed processing mode
 */
class DistributedPartialResultStep3 : public daal::algorithms::PartialResult
{
public:
    /** Default constructor */
    DistributedPartialResultStep3() : daal::algorithms::PartialResult(1) {}
    /** Default destructor */
    virtual ~DistributedPartialResultStep3() {}

    /**
     * Allocates memory to store partial results of the SVD algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        Argument::set(finalResultFromStep3, services::SharedPtr<Result>(new Result()));
    }

    /**
     * Allocates memory to store partial results of the SVD algorithm obtained in the third step in the distributed processing mode
     * \tparam     algorithmFPType            Data type to use for storage in the resulting HomogenNumericTable
     * \param[in]  qCollection  DataCollection of all partial results from step 1 of the SVD algorithm in the distributed processing mode
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
     * Returns results of the SVD algorithm with singular values and the left orthogonal matrix calculated
     * \param[in] id    Identifier of the parameter
     * \return          Parameter that corresponds to the given identifier
     */
    services::SharedPtr<Result> get(DistributedPartialResultStep3Id id) const
    {
        return services::staticPointerCast<Result, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the Result object to store results of the SVD algorithm
     * \param[in] id    Identifier of the parameter
     * \param[in] value Pointer to the Result object
     */
    void set(DistributedPartialResultStep3Id id, const services::SharedPtr<Result> &value)
    {
        Argument::set(id, value);
    }

    /**
     * Checks partial results of the algorithm
     * \param[in] parameter Pointer to parameters
     * \param[in] method Computation method
     */

    void check(const daal::algorithms::Parameter *parameter, int method) const
    {
        if(Argument::size() != 1)
        { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }
    }

    int getSerializationTag() { return SERIALIZATION_SVD_DISTRIBUTED_PARTIAL_RESULT_STEP3_ID; }

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
} // namespace interface1
using interface1::Parameter;
using interface1::Input;
using interface1::DistributedStep2Input;
using interface1::DistributedStep3Input;
using interface1::OnlinePartialResult;
using interface1::Result;
using interface1::DistributedPartialResult;
using interface1::DistributedPartialResultStep3;

} // namespace daal::algorithms::svd
} // namespace daal::algorithms
} // namespace daal
#endif
