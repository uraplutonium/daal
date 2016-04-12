/* file: kmeans_init_types.h */
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
//  Implementation of the K-Means algorithm interface.
//--
*/

#ifndef __KMEANS_INIT_TYPES_H__
#define __KMEANS_INIT_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
/** \brief Contains classes for computing initial clusters for the K-Means algorithm */
namespace init
{
/**
 * <a name="DAAL-ENUM-KMEANS-INIT__METHOD"></a>
 * Available methods for computing initial clusters for the K-Means algorithm
 */
enum Method
{
    deterministicDense = 0, /*!< Default: uses first nClusters points as initial clusters */
    defaultDense       = 0, /*!< Synonym of deterministicDense */
    randomDense        = 1, /*!< Uses random nClusters points as initial clusters */
    deterministicCSR   = 2, /*!< Uses first nClusters points as initial clusters for data in a CSR numeric table */
    randomCSR          = 3  /*!< Uses random nClusters points as initial clusters for data in a CSR numeric table */
};

/**
 * <a name="DAAL-ENUM-KMEANS-INIT__INPUTID"></a>
 * \brief Available identifiers of input objects for computing initial clusters for the K-Means algorithm
 */
enum InputId
{
    data = 0 /*!< %Input data table */
};

/**
 * <a name="DAAL-ENUM-KMEANS-INIT__DISTRIBUTEDSTEP2MASTERINPUTID"></a>
 * \brief Available identifiers of input objects for computing initial clusters for the K-Means algorithm in the distributed processing mode
 */
enum DistributedStep2MasterInputId
{
    partialResults = 0   /*!< Collection of partial results computed on local nodes */
};

/**
 * <a name="DAAL-ENUM-KMEANS-INIT__PARTIALRESULTID"></a>
 * \brief Available identifiers of partial results of computing initial clusters for the K-Means algorithm in the distributed processing mode
 */
enum PartialResultId
{
    partialClustersNumber = 0, /*!< Table with the number of observations assigned to centroids */
    partialClusters       = 1  /*!< Table with the sum of observations assigned to centroids */
};

/**
 * <a name="DAAL-ENUM-KMEANS-INIT__RESULTID"></a>
 * \brief Available identifiers of the results of computing initial clusters for the K-Means algorithm
 */
enum ResultId
{
    centroids = 0 /*!< Table for cluster centroids */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-KMEANS-INIT__PARAMETER"></a>
 * \brief Parameters for computing initial clusters for the K-Means algorithm
 *
 * \snippet kmeans/kmeans_init_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /**
     *  Main constructor
     *  \param[in] _nClusters     Number of clusters
     *  \param[in] _offset        Offset in the total data set specifying the start of a block stored on a given local node
     *  \param[in] seed           Seed for generating random numbers for the initialization
     */
    Parameter(size_t _nClusters, size_t _offset = 0, size_t seed = 777777) : nClusters(_nClusters), offset(_offset), nRowsTotal(0), seed(seed) {}

    /**
     * Constructs parameters of the algorithm that computes initial clusters for the K-Means algorithm
     * by copying another parameters object
     * \param[in] other    Parameters of the K-Means algorithm
     */
    Parameter(const Parameter &other) : nClusters(other.nClusters), offset(other.offset), nRowsTotal(other.nRowsTotal), seed(other.seed) {}

    size_t nClusters;     /*!< Number of clusters */
    size_t nRowsTotal;    /*!< Total number of rows in the data set  */
    size_t offset;        /*!< Offset in the total data set specifying the start of a block stored on a given local node */
    size_t seed;          /*!< Seed for generating random numbers for the initialization */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-KMEANS-INIT__INPUTIFACE"></a>
 * \brief Interface for the K-Means initialization batch and distributed Input classes
 */
class InputIface : public daal::algorithms::Input
{
public:
    InputIface(size_t nElements) : daal::algorithms::Input(nElements) {};

    virtual size_t getNumberOfFeatures() const = 0;
};

/**
 * <a name="DAAL-CLASS-KMEANS-INIT__INPUT"></a>
 * \brief %Input objects for computing initial clusters for the K-Means algorithm
 */
class Input : public InputIface
{
public:
    Input() : InputIface(1) {}
    virtual ~Input() {}

    /**
    * Returns input objects for computing initial clusters for the K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
    * Sets an input object for computing initial clusters for the K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the input object
    */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
    * Returns the number of features in the Input data table
    * \return Number of features in the Input data table
    */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE
    {
        services::SharedPtr<data_management::NumericTable> inTable = get(data);
        return inTable->getNumberOfColumns();
    }

    /**
    * Checks an input object for computing initial clusters for the K-Means algorithm
    * \param[in] par     %Input object
    * \param[in] method  Method of the algorithm
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
 * <a name="DAAL-CLASS-KMEANS-INIT__PARTIALRESULT"></a>
 * \brief Partial results obtained with the compute() method of the K-Means algorithm in the batch processing mode
 */
class PartialResult : public daal::algorithms::PartialResult
{
public:
    PartialResult() : daal::algorithms::PartialResult(2) {}

    virtual ~PartialResult() {};

    /**
     * Allocates memory to store partial results of computing initial clusters for the K-Means algorithm
     * \param[in] input     Pointer to the input structure
     * \param[in] parameter Pointer to the parameter structure
     * \param[in] method    Computation method of the algorithm
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const Parameter *kmPar = static_cast<const Parameter *>(parameter);

        size_t nFeatures = static_cast<const InputIface *>(input)->getNumberOfFeatures();
        size_t nClusters = kmPar->nClusters;

        Argument::set(partialClusters, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(nFeatures, nClusters,
                                                                                    data_management::NumericTable::doAllocate)));
        Argument::set(partialClustersNumber, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<int>( 1, 1, data_management::NumericTable::doAllocate)));
    }

    /**
     * Returns a partial result of computing initial clusters for the K-Means algorithm
     * \param[in] id   Identifier of the partial result
     * \return         Partial result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(PartialResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets a partial result of computing initial clusters for the K-Means algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the object
     */
    void set(PartialResultId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
    * Returns the number of features in the result table of the K-Means algorithm
    * \return Number of features in the result table of the K-Means algorithm
    */
    size_t getNumberOfFeatures() const
    {
        services::SharedPtr<data_management::NumericTable> clusters = get(partialClusters);
        return clusters->getNumberOfColumns();
    }

    /**
    * Checks a partial result of computing initial clusters for the K-Means algorithm
    * \param[in] input   %Input object for the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
               int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 2)
        { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        size_t inputFeatures = static_cast<const InputIface *>(input)->getNumberOfFeatures();
        const Parameter *kmPar = static_cast<const Parameter *>(par);

        services::SharedPtr<data_management::NumericTable> pClusters = get(partialClusters);
        if(pClusters.get() == 0)                             { this->_errors->add(services::ErrorNullOutputNumericTable); return;        }
        if(pClusters->getNumberOfRows() != kmPar->nClusters) { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(pClusters->getNumberOfColumns() != inputFeatures) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;     }

        services::SharedPtr<data_management::NumericTable> pClustersNumber = get(partialClustersNumber);
        if(pClustersNumber.get() == 0)                 { this->_errors->add(services::ErrorNullOutputNumericTable); return;        }
        if(pClustersNumber->getNumberOfRows() != 1)    { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(pClustersNumber->getNumberOfColumns() != 1) { this->_errors->add(services::ErrorInconsistentNumberOfColumns); return;   }
    }

    /**
    * Checks a partial result of computing initial clusters for the K-Means algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    void check(const daal::algorithms::Parameter *par, int method) const
    {
        if(Argument::size() != 2)
        { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        const Parameter *kmPar = static_cast<const Parameter *>(par);

        services::SharedPtr<data_management::NumericTable> pClusters = get(partialClusters);
        if(pClusters.get() == 0)                             { this->_errors->add(services::ErrorNullOutputNumericTable); return;        }
        if(pClusters->getNumberOfRows() != kmPar->nClusters) { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(pClusters->getNumberOfColumns() == 0)             { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;     }

        services::SharedPtr<data_management::NumericTable> pClustersNumber = get(partialClustersNumber);
        if(pClustersNumber.get() == 0)                 { this->_errors->add(services::ErrorNullOutputNumericTable); return;        }
        if(pClustersNumber->getNumberOfRows() != 1)    { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(pClustersNumber->getNumberOfColumns() != 1) { this->_errors->add(services::ErrorInconsistentNumberOfColumns); return;   }
    }

     /**
     * Returns the serialization tag of a partial result
     * \return         Serialization tag of the partial result
     */
    int getSerializationTag() { return SERIALIZATION_KMEANS_INIT_PARTIAL_RESULT_ID; }

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
        daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
 * <a name="DAAL-CLASS-KMEANS-INIT__RESULT"></a>
 * \brief Results obtained with the compute() method that computes initial clusters
 *  for the K-Means algorithm in the batch processing mode
 */
class Result : public daal::algorithms::Result
{
public:
    Result() : daal::algorithms::Result(1) {}

    virtual ~Result() {};

    /**
     * Allocates memory to store the results of computing initial clusters for the K-Means algorithm
     * \param[in] input        Pointer to the input structure
     * \param[in] parameter    Pointer to the parameter structure
     * \param[in] method       Computation method of the algorithm
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const Parameter *kmPar = static_cast<const Parameter *>(parameter);

        size_t nClusters = kmPar->nClusters;
        size_t nFeatures = (static_cast<const Input *>(input))->get(data)->getNumberOfColumns();

        Argument::set(centroids, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(nFeatures, nClusters,
                                                                                    data_management::NumericTable::doAllocate)));
    }

    /**
     * Allocates memory to store the results of computing initial clusters for the K-Means algorithm
     * \param[in] partialResult Pointer to the partial result structure
     * \param[in] parameter     Pointer to the parameter structure
     * \param[in] method        Computation method of the algorithm
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::PartialResult *partialResult, const daal::algorithms::Parameter *parameter,
                  const int method)
    {
        const Parameter *kmPar = static_cast<const Parameter *>(parameter);

        size_t nClusters = kmPar->nClusters;
        size_t nFeatures = (static_cast<const PartialResult *>(partialResult))->get(partialClusters)->getNumberOfColumns();

        Argument::set(centroids, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(nFeatures, nClusters,
                                                                                    data_management::NumericTable::doAllocate)));
    }

    /**
     * Returns the result of computing initial clusters for the K-Means algorithm
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the result of computing initial clusters for the K-Means algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the object
     */
    void set(ResultId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
    * Checks the result of computing initial clusters for the K-Means algorithm
    * \param[in] input   %Input object for the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
               int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() == 0 || Argument::size() > 1)
        { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        size_t inputFeatures = static_cast<const InputIface *>(input)->getNumberOfFeatures();
        const Parameter *kmPar = static_cast<const Parameter *>(par);

        services::SharedPtr<data_management::NumericTable> centroidsTable = get(centroids);
        if(centroidsTable.get() == 0)                             { this->_errors->add(services::ErrorNullInputNumericTable); return;         }
        if(centroidsTable->getNumberOfRows() != kmPar->nClusters) { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(centroidsTable->getNumberOfColumns() != inputFeatures) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;     }
    }

    /**
    * Checks the result of computing initial clusters for the K-Means algorithm
    * \param[in] pres    Partial results of the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    void check(const daal::algorithms::PartialResult *pres, const daal::algorithms::Parameter *par, int method) const
    {
        if(Argument::size() == 0 || Argument::size() > 1)
        { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        size_t inFeatures = const_cast<PartialResult *>(static_cast<const PartialResult *>(pres))->get(
                                partialClusters)->getNumberOfColumns();
        const Parameter *kmPar = static_cast<const Parameter *>(par);

        services::SharedPtr<data_management::NumericTable> centroidsTable = get(centroids);
        if(centroidsTable.get() == 0)                             { this->_errors->add(services::ErrorNullInputNumericTable); return;         }
        if(centroidsTable->getNumberOfRows() != kmPar->nClusters) { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(centroidsTable->getNumberOfColumns() != inFeatures)    { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;     }
    }

     /**
     * Returns the serialization tag of the result
     * \return         Serialization tag of the result
     */
    int getSerializationTag() { return SERIALIZATION_KMEANS_INIT_RESULT_ID; }

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
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
 * <a name="DAAL-CLASS-KMEANS-INIT__DISTRIBUTEDSTEP2MASTERINPUT"></a>
 * \brief %Input objects for computing initials clusters for the K-Means
 *  algorithm in the second step of the distributed processing mode
 */
class DistributedStep2MasterInput : public InputIface
{
public:
    DistributedStep2MasterInput() : InputIface(1)
    {
        Argument::set(partialResults, services::SharedPtr<data_management::DataCollection>(new data_management::DataCollection()));
    }

    virtual ~DistributedStep2MasterInput() {}

    /**
    * Returns an input object for computing initial clusters for the K-Means algorithm
    * in the second step of the distributed processing mode
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    services::SharedPtr<data_management::DataCollection> get(DistributedStep2MasterInputId id) const
    {
        return services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    }

    /**
    * Sets an input object for computing initial clusters for the K-Means algorithm
    * in the second step of the distributed processing mode
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the object
    */
    void set(DistributedStep2MasterInputId id, const services::SharedPtr<data_management::DataCollection> &ptr)
    {
        Argument::set(id, services::staticPointerCast<data_management::SerializationIface, data_management::DataCollection>(ptr));
    }

    /**
     * Adds a value to the data collection of input objects for computing initial clusters for the K-Means algorithm
     * in the second step of the distributed processing mode
     * \param[in] id    Identifier of the parameter
     * \param[in] value Pointer to the new parameter value
     */
    void add(DistributedStep2MasterInputId id, const services::SharedPtr<PartialResult> &value)
    {
        services::SharedPtr<data_management::DataCollection> collection
            = services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
        collection->push_back( value );
    }

    /**
    * Returns the number of features in the Input data table in the second step of the distributed processing mode
    * \return Number of features in the Input data table
    */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE
    {
        services::SharedPtr<data_management::DataCollection> collection = get(partialResults);
        services::SharedPtr<PartialResult> pres = services::staticPointerCast<PartialResult, data_management::SerializationIface>((*collection)[0]);
        return pres->getNumberOfFeatures();
    }

    /**
    * Checks an input object for computing initial clusters for the K-Means algorithm
    * in the second step of the distributed processing mode
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        services::SharedPtr<data_management::DataCollection> collection = get(partialResults);

        if(collection.get() == 0) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        size_t n = collection->size();
        if(n == 0) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        for(size_t i = 0; i < n; i++)
        {
            services::SharedPtr<PartialResult> pres =
                services::staticPointerCast<PartialResult, data_management::SerializationIface>((*collection)[i]);
            if( pres.get() == 0 ) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

            pres->check(par, method);
        }
    }
};
} // namespace interface1
using interface1::Parameter;
using interface1::InputIface;
using interface1::Input;
using interface1::PartialResult;
using interface1::Result;
using interface1::DistributedStep2MasterInput;

} // namespace daal::algorithms::kmeans::init
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
#endif
