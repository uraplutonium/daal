/* file: graph.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation
* All Rights Reserved.
*
* If this  software was obtained  under the  Intel Simplified  Software License,
* the following terms apply:
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*
*
* If this  software  was obtained  under the  Apache License,  Version  2.0 (the
* "License"), the following terms apply:
*
* You may  not use this  file except  in compliance  with  the License.  You may
* obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
*
*
* Unless  required  by   applicable  law  or  agreed  to  in  writing,  software
* distributed under the License  is distributed  on an  "AS IS"  BASIS,  WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*
* See the   License  for the   specific  language   governing   permissions  and
* limitations under the License.
*******************************************************************************/

#ifndef __DATA_MANAGEMENT_DATA_GRAPH_H__
#define __DATA_MANAGEMENT_DATA_GRAPH_H__

#include "services/base.h"
#include "services/allocators.h"
#include "services/error_handling.h"

#include "algorithms/algorithm_types.h"
#include "data_management/data/data_collection.h"
#include "data_management/data/data_dictionary.h"
#include "data_management/data/internal/indexers.h"

#include "data_management/data/numeric_types.h"

namespace daal
{
namespace data_management
{
namespace interface1
{

class GraphFeature
{
public:
    GraphFeature() :
        indexType(features::DAAL_OTHER_T),
        featureType(features::DAAL_CONTINUOUS),
        typeSize(0) { }

    explicit GraphFeature(features::IndexNumType indexType,
                          features::FeatureType featureType,
                          size_t typeSize) :
        indexType(indexType),
        featureType(featureType),
        typeSize(typeSize) { }

    features::IndexNumType  indexType;
    features::FeatureType   featureType;
    size_t                  typeSize;
};

class GraphTraits
{
public:
    typedef size_t VertexId;
    typedef size_t EdgeId;

private:
    /* Disable constructor */
    GraphTraits();
};

template<typename DataType = GraphTraits::VertexId>
class GraphDataDescriptor : public Base
{
public:
    void setBuffer(const services::BufferView<DataType> &bufferView)
    {
        _data.setPtr(bufferView.data(), 1, bufferView.size());
    }

    void setIndexer(const internal::Indexer<size_t> &indexer)
    {
        _indexer = services::SharedPtr<internal::Indexer<size_t> >(indexer.copy());
    }

    void setRWFlag(ReadWriteMode rwFlag)
    {
        _data.setRWFlag(rwFlag);
    }

    DataType *getBlockPtr() const
    {
        return _data.getBlockPtr();
    }

    size_t size() const
    {
        return _data.getNumberOfRows();
    }

    const internal::Indexer<size_t> &getIndexer() const
    {
        return *_indexer;
    }

    ReadWriteMode getRWFlag() const
    {
        return (ReadWriteMode)_data.getRWFlag();
    }

    void resizeBuffer(size_t bufferSize)
    {
        _data.resizeBuffer(1, bufferSize);
    }

    void reset()
    {
        _data.reset();
    }

private:
    BlockDescriptor<DataType> _data;
    services::SharedPtr<internal::Indexer<size_t> > _indexer;
};

class AdjacencyListGraphIface
{
public:
    virtual ~AdjacencyListGraphIface() { }

    virtual services::Status getAdjacentVertices(const GraphTraits::VertexId &vertexId,
                                                 ReadWriteMode rwflag,
                                                 GraphDataDescriptor<> &descriptor) = 0;

    virtual services::Status releaseAdjacentVertices(GraphDataDescriptor<> &descriptor) = 0;
};

class Graph : public Base, public AdjacencyListGraphIface
{
public:
    template <typename T>
    services::Status getVertexFeatures(const std::vector<GraphTraits::VertexId> &container,
                                       ReadWriteMode rwflag, GraphDataDescriptor<T> &descriptor)
    { return getVertexFeaturesIdx<T>(internal::idx::many(container), rwflag, descriptor); }

    template <typename T>
    services::Status getVertexFeatures(const services::Collection<GraphTraits::VertexId> &container,
                                       ReadWriteMode rwflag, GraphDataDescriptor<T> &descriptor)
    { return getVertexFeaturesIdx<T>(internal::idx::many(container), rwflag, descriptor); }

    template <typename T>
    services::Status getVertexFeatures(ReadWriteMode rwflag, GraphDataDescriptor<T> &descriptor)
    { return getVertexFeaturesIdx<T>(internal::idx::range((size_t)0, getNumberOfVertices()), rwflag, descriptor); }

    template<typename T>
    services::Status releaseVertexFeatures(GraphDataDescriptor<T> &descriptor)
    { return releaseVertexFeaturesIdx<T>(descriptor.getIndexer(), descriptor.getRWFlag(), descriptor); }

    size_t getNumberOfVertices() const
    { return _numberOfVertices; }

    size_t getNumberOfEdges() const
    { return _numberOfEdges; }

protected:
    explicit Graph(size_t numberOfVertices = 0, size_t numberOfEdges = 0,
                   const GraphFeature &vertexFeaturesInfo = GraphFeature(),
                   const GraphFeature &edgeFeaturesInfo = GraphFeature())
    { initialize(numberOfVertices, numberOfEdges, vertexFeaturesInfo, edgeFeaturesInfo); }

    void setNumberOfEdges(size_t numberOfEdges)
    { _numberOfEdges = numberOfEdges; }

private:
    void initialize(size_t numberOfVertices, size_t numberOfEdges,
                    const GraphFeature &vertexFeaturesInfo,
                    const GraphFeature &edgeFeaturesInfo)
    {
        _numberOfVertices   = numberOfVertices;
        _numberOfEdges      = numberOfEdges;
        _vertexFeaturesInfo = vertexFeaturesInfo;

        if (numberOfVertices && _vertexFeaturesInfo.typeSize)
        {
            const size_t typeSize = _vertexFeaturesInfo.typeSize;
            _verticesData.reallocate(typeSize * numberOfVertices);
        }
    }

    template<typename T>
    services::Status getVertexFeaturesIdx(const internal::Indexer<GraphTraits::VertexId> &indexer,
                                          ReadWriteMode rwflag, GraphDataDescriptor<T> &descriptor)
    {
        if (!_verticesData.size())
        { return services::Status(); }

        descriptor.setRWFlag(rwflag);
        descriptor.setIndexer(indexer);
        descriptor.resizeBuffer(indexer.size());

        if (rwflag == writeOnly)
        { return services::Status(); }

        const bool conversionRequired =
            (_vertexFeaturesInfo.indexType != features::internal::getIndexNumType<T>()) ||
            (_vertexFeaturesInfo.typeSize  != sizeof(T));

        T *featuresBuffer = descriptor.getBlockPtr();
        const size_t typeSize = _vertexFeaturesInfo.typeSize;

        if (conversionRequired)
        {
            services::internal::Buffer<daal::byte> gatherBuffer(indexer.size() * typeSize);

            /* Gather the raw bytes from _verticesData to the gatherBuffer
             * in order to apply data conversion function to that buffer */
            gather(indexer, _verticesData, gatherBuffer, typeSize);

            /* Convert features from internal type to requested
             * type and write down to GraphDataDescriptor buffer */
            internal::getVectorUpCast(_vertexFeaturesInfo.indexType,
                                      internal::getConversionDataType<T>())
            (indexer.size(), (void *)gatherBuffer.data(), (void *)featuresBuffer);
        }
        else
        {
            daal::byte *rawVerticesData = _verticesData.data();
            for (size_t i = 0; i < indexer.size(); ++i)
            {
                DAAL_ASSERT( indexer[i] * typeSize < _verticesData.size() );
                daal::byte *elementPtr = rawVerticesData + indexer[i] * typeSize;
                featuresBuffer[i] = *((T *)elementPtr);
            }
        }

        return services::Status();
    }

    template<typename T>
    services::Status releaseVertexFeaturesIdx(const internal::Indexer<GraphTraits::VertexId> &indexer,
                                              ReadWriteMode rwflag, GraphDataDescriptor<T> &descriptor)
    {
        if (!_verticesData.size() || rwflag == readOnly)
        {
            descriptor.reset();
            return services::Status();
        }

        const bool conversionRequired =
            (_vertexFeaturesInfo.indexType != features::internal::getIndexNumType<T>()) ||
            (_vertexFeaturesInfo.typeSize  != sizeof(T));

        T *featuresBuffer = descriptor.getBlockPtr();
        const size_t typeSize = _vertexFeaturesInfo.typeSize;

        if (conversionRequired)
        {
            services::internal::Buffer<daal::byte> conversionBuffer(indexer.size() * typeSize);

            /*  */
            internal::getVectorDownCast(_vertexFeaturesInfo.indexType,
                                        internal::getConversionDataType<T>())
            (indexer.size(), (void *)featuresBuffer, (void *)conversionBuffer.data());

            /*  */
            scatter(indexer, conversionBuffer, _verticesData, typeSize);
        }
        else
        {
            daal::byte *rawVerticesData = _verticesData.data();
            for (size_t i = 0; i < indexer.size(); ++i)
            {
                DAAL_ASSERT( indexer[i] * typeSize < _verticesData.size() );
                daal::byte *elementPtr = rawVerticesData + indexer[i] * typeSize;
                *((T *)elementPtr) = featuresBuffer[i];
            }
        }

        descriptor.reset();
        return services::Status();
    }

    static void gather(const internal::Indexer<GraphTraits::VertexId> &indexer,
                       const services::internal::Buffer<daal::byte> &source,
                             services::internal::Buffer<daal::byte> &dest,
                             size_t typeSize)
    {
        DAAL_ASSERT( dest.data() );
        DAAL_ASSERT( source.data() );
        DAAL_ASSERT( dest.size() >= indexer.size() * typeSize );

        daal::byte *rawsource = source.data();
        daal::byte *rawdest = dest.data();

        for (size_t i = 0; i < indexer.size(); ++i)
        {
            DAAL_ASSERT( indexer[i] * typeSize < source.size() );

            daal::byte *ss = rawsource + indexer[i] * typeSize;
            daal::byte *ds = rawdest + i * typeSize;

            for (size_t j = 0; j < typeSize; j++)
            { ds[j] = ss[j]; }
        }
    }

    static void scatter(const internal::Indexer<GraphTraits::VertexId> &indexer,
                        const services::internal::Buffer<daal::byte> &source,
                              services::internal::Buffer<daal::byte> &dest,
                              size_t typeSize)
    {
        DAAL_ASSERT( dest.data() );
        DAAL_ASSERT( source.data() );
        DAAL_ASSERT( source.size() >= indexer.size() * typeSize );

        daal::byte *rawsource = source.data();
        daal::byte *rawdest = dest.data();

        for (size_t i = 0; i < indexer.size(); ++i)
        {
            DAAL_ASSERT( indexer[i] * typeSize < dest.size() );

            daal::byte *ss = rawsource + i * typeSize;
            daal::byte *ds = rawdest + indexer[i] * typeSize;

            for (size_t j = 0; j < typeSize; j++)
            { ds[j] = ss[j]; }
        }
    }

    size_t _numberOfVertices;
    size_t _numberOfEdges;
    GraphFeature _vertexFeaturesInfo;
    services::internal::Buffer<daal::byte> _verticesData;
};
typedef services::SharedPtr<Graph> GraphPtr;

class AdjacencyListGraph : public Graph
{
private:
    typedef Graph super;
    typedef services::internal::Buffer<GraphTraits::VertexId> AdjacentVerticesBuffer;

public:
    static services::SharedPtr<AdjacencyListGraph> create(size_t numberOfVertices)
    {
        return services::SharedPtr<AdjacencyListGraph>(new AdjacencyListGraph(numberOfVertices));
    }

    template<typename VertexFeatureType>
    static services::SharedPtr<AdjacencyListGraph> create(size_t numberOfVertices)
    {
        const GraphFeature vertexFeatureType = getGraphFeature<VertexFeatureType>();
        return services::SharedPtr<AdjacencyListGraph>(new AdjacencyListGraph(numberOfVertices,
                                                                              vertexFeatureType));
    }

    virtual services::Status getAdjacentVertices(const GraphTraits::VertexId &vertexId,
                                                 ReadWriteMode rwflag,
                                                 GraphDataDescriptor<> &descriptor) DAAL_C11_OVERRIDE
    {
        descriptor.setBuffer(getAdjacentVerticesBuffer(vertexId).view());
        return services::Status();
    }

    virtual services::Status releaseAdjacentVertices(GraphDataDescriptor<> &descriptor) DAAL_C11_OVERRIDE
    {
        descriptor.reset();
        return services::Status();
    }

    size_t getNumberOfAdjacentVertices(const GraphTraits::VertexId &vertexId)
    {
        DAAL_CHECK(vertexId < _adjacentVertices.size(), services::ErrorIncorrectIndex);
        return getAdjacentVerticesBuffer(vertexId).size();
    }

    services::Status setNumberOfAdjacentVertices(const GraphTraits::VertexId &vertexId,
                                                 size_t numberOfVertices)
    {
        /* Number of adjacent vertices cannot be larger than the total number of vertices */
        DAAL_CHECK(numberOfVertices <= _adjacentVertices.size(), services::ErrorIncorrectIndex);
        DAAL_CHECK(vertexId < _adjacentVertices.size(), services::ErrorIncorrectIndex);

        AdjacentVerticesBuffer &buffer = getAdjacentVerticesBuffer(vertexId);
        setNumberOfEdges(getNumberOfEdges() - buffer.size() + numberOfVertices);
        return buffer.reallocate(numberOfVertices, /* copy */ true);
    }

protected:
    explicit AdjacencyListGraph(size_t numberOfVertices,
                                const GraphFeature &vertexFeatureType = GraphFeature()) :
        super(numberOfVertices, 0, vertexFeatureType),
        _edgesCounter(0),
        _adjacentVertices(numberOfVertices)
    { initializeAdjacentVertices(); }

private:
    AdjacentVerticesBuffer &getAdjacentVerticesBuffer(const GraphTraits::VertexId &vertexId)
    {
        AdjacentVerticesBuffer *&buffer = _adjacentVertices[vertexId];
        if (!buffer)
        { buffer = new AdjacentVerticesBuffer(); }

        return *buffer;
    }

    void initializeAdjacentVertices()
    {
        for (size_t i = 0; i < _adjacentVertices.size(); i++)
        { _adjacentVertices[i] = NULL; }
    }


    template<typename T>
    static GraphFeature getGraphFeature()
    {
        return GraphFeature(features::internal::getIndexNumType<T>(),
                            features::DAAL_CONTINUOUS, sizeof(T));
    }

    size_t _edgesCounter;
    services::internal::Buffer<AdjacentVerticesBuffer *> _adjacentVertices;
};
typedef services::SharedPtr<AdjacencyListGraph> AdjacencyListGraphPtr;

} // namespace interface1

using interface1::GraphFeature;
using interface1::GraphTraits;
using interface1::GraphDataDescriptor;
using interface1::AdjacencyListGraphIface;
using interface1::Graph;
using interface1::GraphPtr;
using interface1::AdjacencyListGraph;
using interface1::AdjacencyListGraphPtr;

} // namespace data_management
} // namespace daal

#endif
