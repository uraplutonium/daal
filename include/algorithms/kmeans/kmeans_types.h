/* file: kmeans_types.h */
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

#ifndef __KMEANS_TYPES_H__
#define __KMEANS_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
/** \brief Contains classes of the K-Means algorithm */
namespace kmeans
{
/**
 * <a name="DAAL-ENUM-KMEANS__METHOD"></a>
 * Available methods of the K-Means algorithm
 */
enum Method
{
    lloydDense = 0,     /*!< Default: performance-oriented method, synonym of defaultDense */
    defaultDense = 0,   /*!< Default: performance-oriented method, synonym of lloydDense */
    lloydCSR = 1        /*!< Implementation of the Lloyd algorithm for CSR numeric tables */
};

/**
 * <a name="DAAL-ENUM-KMEANS__DISTANCETYPE"></a>
 * Supported distance types
 */
enum DistanceType
{
    euclidean /*!< Euclidean distance */
};

/**
 * <a name="DAAL-ENUM-KMEANS__INPUTID"></a>
 * \brief Available identifiers of input objects for the K-Means algorithm
 */
enum InputId
{
    data = 0,            /*!< %Input data table */
    inputCentroids = 1 /*!< Initial centroids for the algorithm */
};

/**
 * <a name="DAAL-ENUM-KMEANS__MASTERINPUTID"></a>
 * \brief Available identifiers of input objects for the K-Means algorithm in the distributed processing mode
 */
enum MasterInputId
{
    partialResults = 0   /*!< Collection of partial results computed on local nodes  */
};

/**
 * <a name="DAAL-ENUM-KMEANS__PARTIALRESULTID"></a>
 * \brief Available identifiers of partial results of the K-Means algorithm in the distributed processing mode
 */
enum PartialResultId
{
    nObservations       = 0,  /*!< Table containing the number of observations assigned to centroids */
    partialSums         = 1,  /*!< Table containing the sum of observations assigned to centroids */
    partialGoalFunction = 2,  /*!< Table containing a goal function value */
    partialAssignments  = 3   /*!< Table containing assignments of observations to particular clusters */
};

/**
 * <a name="DAAL-ENUM-KMEANS__RESULTID"></a>
 * \brief Available identifiers of results of the K-Means algorithm
 */
enum ResultId
{
    centroids    = 0, /*!< Table containing cluster centroids */
    assignments  = 1, /*!< Table containing assignments of observations to particular clusters */
    goalFunction = 2, /*!< Table containing a goal function value */
    nIterations  = 3  /*!< Table containing the number of executed iterations */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-KMEANS__PARAMETER"></a>
 * \brief Parameters for the K-Means algorithm
 * \par Enumerations
 *      - \ref DistanceType Methods for distance computation
 *
 * \snippet kmeans/kmeans_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /**
     *  Constructs parameters of the K-Means algorithm
     *  \param[in] _nClusters   Number of clusters
     *  \param[in] _maxIterations Number of iterations
     */
    Parameter(size_t _nClusters, size_t _maxIterations) :
        nClusters(_nClusters), maxIterations(_maxIterations), accuracyThreshold(0.0), gamma(1.0),
        distanceType(euclidean), assignFlag(true) {}

    /**
     *  Constructs parameters of the K-Means algorithm by copying another parameters of the K-Means algorithm
     *  \param[in] other    Parameters of the K-Means algorithm
     */
    Parameter(const Parameter &other) :
        nClusters(other.nClusters), maxIterations(other.maxIterations),
        accuracyThreshold(other.accuracyThreshold), gamma(other.gamma),
        distanceType(other.distanceType), assignFlag(other.assignFlag)
    {}

    size_t nClusters;                                      /*!< Number of clusters */
    size_t maxIterations;                                  /*!< Number of iterations */
    double accuracyThreshold;                              /*!< Threshold for the termination of the algorithm */
    double gamma;                                          /*!< Weight used in distance computation for categorical features */
    DistanceType distanceType;                             /*!< Distance used in the algorithm */
    bool assignFlag;                                       /*!< Do data points assignment */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-KMEANS__INPUTIFACE"></a>
 * \brief Interface for input objects for the the K-Means algorithm in the batch and distributed processing modes
 */
class InputIface : public daal::algorithms::Input
{
public:
    InputIface(size_t nElements) : daal::algorithms::Input(nElements) {};

    virtual size_t getNumberOfFeatures() const = 0;
};

/**
 * <a name="DAAL-CLASS-KMEANS__INPUT"></a>
 * \brief %Input objects for the K-Means algorithm
 */
class Input : public InputIface
{
public:
    Input() : InputIface(2) {}
    virtual ~Input() {}

    /**
    * Returns an input object for the K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    services::SharedPtr<data_management::NumericTable> get(InputId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
    * Sets an input object for the K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the object
    */
    void set(InputId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }


    /**
    * Returns the number of features in the input object
    * \return Number of features in the input object
    */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE
    {
        services::SharedPtr<data_management::NumericTable> inTable = get(data);
        return inTable->getNumberOfColumns();
    }

    /**
    * Checks input objects for the K-Means algorithm
    * \param[in] par     Algorithm parameter
    * \param[in] method  Computation method of the algorithm
    */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

        services::SharedPtr<data_management::NumericTable> inData = get(data);
        if(inData.get() == 0)                 { this->_errors->add(services::ErrorNullInputNumericTable); return;         }
        if(inData->getNumberOfRows() == 0)    { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(inData->getNumberOfColumns() == 0) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;     }

        services::SharedPtr<data_management::NumericTable> inClusters = get(inputCentroids);
        if(inClusters.get() == 0)                 { this->_errors->add(services::ErrorNullInputNumericTable); return;         }
        if(inClusters->getNumberOfRows() == 0)    { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(inClusters->getNumberOfColumns() == 0) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;     }

        if(inData->getNumberOfColumns() != inClusters->getNumberOfColumns())
        { this->_errors->add(services::ErrorInconsistentNumberOfColumns); return; }
    }
};

/**
 * <a name="DAAL-CLASS-KMEANS__PARTIALRESULT"></a>
 * \brief Partial results obtained with the compute() method of the K-Means algorithm in the batch processing mode
 */
class PartialResult : public daal::algorithms::PartialResult
{
public:
    PartialResult() : daal::algorithms::PartialResult(4) {}

    virtual ~PartialResult() {};

    /**
     * Allocates memory to store partial results of the K-Means algorithm
     * \param[in] input        Pointer to the structure of the input objects
     * \param[in] parameter    Pointer to the structure of the algorithm parameters
     * \param[in] method       Computation method of the algorithm
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const Parameter *kmPar = static_cast<const Parameter *>(parameter);

        size_t nFeatures = static_cast<const InputIface *>(input)->getNumberOfFeatures();
        size_t nClusters = kmPar->nClusters;

        Argument::set(nObservations, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(1, nClusters, data_management::NumericTable::doAllocate)));
        Argument::set(partialSums, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(nFeatures, nClusters,
                                                                                    data_management::NumericTable::doAllocate)));
        Argument::set(partialGoalFunction, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>(1, 1, data_management::NumericTable::doAllocate)));

        if( kmPar->assignFlag )
        {
            Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
            size_t nRows = algInput->get(data)->getNumberOfRows();

            Argument::set(partialAssignments, services::SharedPtr<data_management::SerializationIface>(
                              new data_management::HomogenNumericTable<int>(1, nRows, data_management::NumericTable::doAllocate)));
        }
    }

    /**
     * Returns a partial result of the K-Means algorithm
     * \param[in] id   Identifier of the partial result
     * \return         Partial result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(PartialResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets a partial result of the K-Means algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the object
     */
    void set(PartialResultId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
    * Returns the number of features in the Input data table
    * \return Number of features in the Input data table
    */

    size_t getNumberOfFeatures() const
    {
        services::SharedPtr<data_management::NumericTable> sums = get(partialSums);
        return sums->getNumberOfColumns();
    }

    /**
    * Checks partial results of the K-Means algorithm
    * \param[in] input   %Input object of the algorithm
    * \param[in] par     Algorithm parameter
    * \param[in] method  Computation method
    */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
               int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 4)
        { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        size_t inputFeatures = static_cast<const InputIface *>(input)->getNumberOfFeatures();
        const Parameter *kmPar = static_cast<const Parameter *>(par);

        services::SharedPtr<data_management::NumericTable> pCounter = get(nObservations);
        if(pCounter.get() == 0)                             { this->_errors->add(services::ErrorNullOutputNumericTable); return;        }
        if(pCounter->getNumberOfRows() != kmPar->nClusters) { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(pCounter->getNumberOfColumns() != 1)             { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;     }

        services::SharedPtr<data_management::NumericTable> pSums = get(partialSums);
        if(pSums.get() == 0)                             { this->_errors->add(services::ErrorNullOutputNumericTable); return;        }
        if(pSums->getNumberOfRows() != kmPar->nClusters) { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(pSums->getNumberOfColumns() != inputFeatures) { this->_errors->add(services::ErrorInconsistentNumberOfColumns); return;   }

        services::SharedPtr<data_management::NumericTable> pGoal   = get(partialGoalFunction  );
        if(pGoal  .get() == 0)                 { this->_errors->add(services::ErrorNullOutputNumericTable); return;        }
        if(pGoal  ->getNumberOfRows() != 1)    { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(pGoal  ->getNumberOfColumns() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;     }

        if( kmPar->assignFlag )
        {
            Input *algInput = dynamic_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
            if( algInput == 0 ) { this->_errors->add(services::ErrorNullInput); return; }
            size_t nRows = algInput->get(data)->getNumberOfRows();

            services::SharedPtr<data_management::NumericTable> pAssignments = get( partialAssignments );
            if(pAssignments.get() == 0)                  { this->_errors->add(services::ErrorNullOutputNumericTable); return;        }
            if(pAssignments->getNumberOfRows() != nRows) { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
            if(pAssignments->getNumberOfColumns() != 1)  { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;     }
        }
    }

    /**
    * Checks partial results of the K-Means algorithm
    * \param[in] par     Algorithm parameter
    * \param[in] method  Computation method
    */
    void check(const daal::algorithms::Parameter *par, int method) const
    {
        if(Argument::size() != 4)
        { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        const Parameter *kmPar = static_cast<const Parameter *>(par);

        services::SharedPtr<data_management::NumericTable> pCounter = get(nObservations);
        if(pCounter.get() == 0)                             { this->_errors->add(services::ErrorNullOutputNumericTable); return;        }
        if(pCounter->getNumberOfRows() != kmPar->nClusters) { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(pCounter->getNumberOfColumns() != 1)             { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;     }

        services::SharedPtr<data_management::NumericTable> pSums = get(partialSums);
        if(pSums.get() == 0)                             { this->_errors->add(services::ErrorNullOutputNumericTable); return;        }
        if(pSums->getNumberOfRows() != kmPar->nClusters) { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(pSums->getNumberOfColumns() == 0)             { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;     }

        services::SharedPtr<data_management::NumericTable> pGoal   = get( partialGoalFunction );
        if(pGoal  .get() == 0)                 { this->_errors->add(services::ErrorNullOutputNumericTable); return;        }
        if(pGoal  ->getNumberOfRows() != 1)    { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(pGoal  ->getNumberOfColumns() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;     }

        if( kmPar->assignFlag )
        {
            services::SharedPtr<data_management::NumericTable> pAssignments = get( partialAssignments );
            if(pAssignments.get() == 0)                  { this->_errors->add(services::ErrorNullOutputNumericTable); return;        }
            if(pAssignments->getNumberOfRows() == 0)     { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
            if(pAssignments->getNumberOfColumns() != 1)  { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;     }
        }
    }

     /**
     * Returns the serialization tag of a partial result
     * \return         Serialization tag of the partial result
     */
    int getSerializationTag() { return SERIALIZATION_KMEANS_PARTIAL_RESULT_ID; }

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
 * <a name="DAAL-CLASS-KMEANS__RESULT"></a>
 * \brief Results obtained with the compute() method of the K-Means algorithm in the batch processing mode
 */
class Result : public daal::algorithms::Result
{
public:
    Result() : daal::algorithms::Result(4) {}

    virtual ~Result() {};

    /**
     * Allocates memory to store the results of the K-Means algorithm
     * \param[in] input     Pointer to the structure of the input objects
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const Parameter *kmPar = static_cast<const Parameter *>(parameter);

        Input *algInput  = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
        size_t nFeatures = algInput->getNumberOfFeatures();
        size_t nClusters = kmPar->nClusters;

        Argument::set(centroids, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>
                          (nFeatures, nClusters, data_management::NumericTable::doAllocate)));
        Argument::set(goalFunction, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>
                          (        1,         1, data_management::NumericTable::doAllocate)));
        Argument::set(nIterations, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<int>
                          (        1,         1, data_management::NumericTable::doAllocate)));

        if(kmPar->assignFlag)
        {
            size_t nRows = algInput->get(data)->getNumberOfRows();

            Argument::set(assignments, services::SharedPtr<data_management::SerializationIface>(
                              new data_management::HomogenNumericTable<int>(1, nRows, data_management::NumericTable::doAllocate)));
        }
    }

    /**
     * Allocates memory to store the results of the K-Means algorithm
     * \param[in] partialResult Pointer to the partial result structure
     * \param[in] parameter     Pointer to the structure of the algorithm parameters
     * \param[in] method        Computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::PartialResult *partialResult, const daal::algorithms::Parameter *parameter,
                  const int method)
    {
        size_t nClusters = (static_cast<const Parameter *>(parameter))->nClusters;
        size_t nFeatures = (static_cast<const PartialResult *>(partialResult))->getNumberOfFeatures();

        Argument::set(centroids, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>
                          (nFeatures, nClusters, data_management::NumericTable::doAllocate)));
        Argument::set(goalFunction, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<algorithmFPType>
                          (        1,         1, data_management::NumericTable::doAllocate)));
        Argument::set(nIterations, services::SharedPtr<data_management::SerializationIface>(
                          new data_management::HomogenNumericTable<int>
                          (        1,         1, data_management::NumericTable::doAllocate)));
    }

    /**
     * Returns the result of the K-Means algorithm
     * \param[in] id   Result identifier
     * \return         Result that corresponds to the given identifier
     */
    services::SharedPtr<data_management::NumericTable> get(ResultId id) const
    {
        return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
    }

    /**
     * Sets the result of the K-Means algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the object
     */
    void set(ResultId id, const services::SharedPtr<data_management::NumericTable> &ptr)
    {
        Argument::set(id, ptr);
    }

    /**
    * Checks the result of the K-Means algorithm
    * \param[in] input   %Input objects for the algorithm
    * \param[in] par     Algorithm parameter
    * \param[in] method  Computation method
    */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par,
               int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 4)
        { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));

        size_t inputFeatures   = algInput->getNumberOfFeatures();
        const Parameter *kmPar = static_cast<const Parameter *>(par);

        services::SharedPtr<data_management::NumericTable> centroidsTable = get(centroids);
        if(centroidsTable.get() == 0)                             { this->_errors->add(services::ErrorNullOutputNumericTable); return;        }
        if(centroidsTable->getNumberOfRows() != kmPar->nClusters) { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(centroidsTable->getNumberOfColumns() != inputFeatures) { this->_errors->add(services::ErrorInconsistentNumberOfColumns); return;   }

        services::SharedPtr<data_management::NumericTable> goalTable = get(goalFunction);
        if(goalTable.get() == 0)                 { this->_errors->add(services::ErrorNullOutputNumericTable); return;        }
        if(goalTable->getNumberOfRows() != 1)    { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(goalTable->getNumberOfColumns() != 1) { this->_errors->add(services::ErrorInconsistentNumberOfColumns); return;   }

        services::SharedPtr<data_management::NumericTable> iterationsTable = get(nIterations);
        if(iterationsTable.get() == 0)                 { this->_errors->add(services::ErrorNullOutputNumericTable); return;        }
        if(iterationsTable->getNumberOfRows() != 1)    { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(iterationsTable->getNumberOfColumns() != 1) { this->_errors->add(services::ErrorInconsistentNumberOfColumns); return;   }

        if(kmPar->assignFlag)
        {
            services::SharedPtr<data_management::NumericTable> assignmentsTable = get(assignments);

            size_t inputRows = algInput->get(data)->getNumberOfRows();

            if(assignmentsTable.get() == 0)                      { this->_errors->add(services::ErrorNullOutputNumericTable); return;            }
            if(assignmentsTable->getNumberOfRows() != inputRows) { this->_errors->add(services::ErrorInconsistentNumberOfRows); return;          }
            if(assignmentsTable->getNumberOfColumns() != 1)      { this->_errors->add(services::ErrorIncorrectSizeOfOutputNumericTable); return; }
        }
    }

    /**
    * Checks the results of the K-Means algorithm
    * \param[in] pres    Partial results of the algorithm
    * \param[in] par     Algorithm parameter
    * \param[in] method  Computation method
    */
    void check(const daal::algorithms::PartialResult *pres, const daal::algorithms::Parameter *par, int method) const
    {
        if(Argument::size() != 4)
        { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }

        PartialResult *algPres = static_cast<PartialResult *>(const_cast<daal::algorithms::PartialResult *>(pres));
        const Parameter *kmPar = static_cast<const Parameter *>(par);

        size_t presFeatures = algPres->get(partialSums)->getNumberOfColumns();

        services::SharedPtr<data_management::NumericTable> centroidsTable = get(centroids);
        if(centroidsTable.get() == 0)                             { this->_errors->add(services::ErrorNullOutputNumericTable); return;        }
        if(centroidsTable->getNumberOfRows() != kmPar->nClusters) { this->_errors->add(services::ErrorIncorrectNumberOfObservations); return; }
        if(centroidsTable->getNumberOfColumns() != presFeatures)  { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return;     }
    }

     /**
     * Returns the serialization tag of the result
     * \return         Serialization tag of the result
     */
    int getSerializationTag() { return SERIALIZATION_KMEANS_RESULT_ID; }

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
 * <a name="DAAL-CLASS-KMEANS__DISTRIBUTEDSTEP2MASTERINPUT"></a>
 * \brief %Input objects for the K-Means algorithm in the distributed processing mode
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
    * Returns an input object for the K-Means algorithm in the second step of the distributed processing mode
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    services::SharedPtr<data_management::DataCollection> get(MasterInputId id) const
    {
        return services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    }

    /**
    * Sets an input object for the K-Means algorithm in the second step of the distributed processing mode
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the object
    */
    void set(MasterInputId id, const services::SharedPtr<data_management::DataCollection> &ptr)
    {
        Argument::set(id, services::staticPointerCast<data_management::SerializationIface, data_management::DataCollection>(ptr));
    }

    /**
     * Adds partial results computed on local nodes to the input for the K-Means algorithm
     * in the second step of the distributed processing mode
     * \param[in] id    Identifier of the input object
     * \param[in] value Pointer to the object
     */
    void add(MasterInputId id, const services::SharedPtr<PartialResult> &value)
    {
        services::SharedPtr<data_management::DataCollection> collection =
            services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
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
    * Checks an input object for the K-Means algorithm in the second step of the distributed processing mode
    * \param[in] par     Algorithm parameter
    * \param[in] method  Computation method
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

} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
#endif
