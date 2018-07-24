/* file: graph.cpp */
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

/*
!  Content:
!    C++ example of using a structure of arrays (SOA)
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-GRAPHS_GRAPH">
 * \example graph.cpp
 */
#include <vector>
#include <iostream>

#include "daal.h"

using namespace daal;
using namespace daal::data_management;

void writeFeaturesWithDescriptorOfSameType(const GraphPtr &graph)
{
    std::cout << "writeFeaturesWithDescriptorOfSameType():" << std::endl;

    GraphDataDescriptor<float> gdd;
    std::vector<size_t> idx(2); // Ids of vertices
    idx[0] = 1; idx[1] = 3;

    graph->getVertexFeatures(idx, writeOnly, gdd);
    float *writeOnlyBlockPtr = gdd.getBlockPtr();
    writeOnlyBlockPtr[0] = 0.5;
    writeOnlyBlockPtr[1] = 2.5;
    graph->releaseVertexFeatures(gdd);

    graph->getVertexFeatures(idx, readOnly, gdd);
    std::cout << "gdd.size() = " << gdd.size() << std::endl;
    std::cout << "gdd[0]     = " << gdd.getBlockPtr()[0] << std::endl;
    std::cout << "gdd[1]     = " << gdd.getBlockPtr()[1] << std::endl;
    graph->releaseVertexFeatures(gdd);

    std::cout << std::endl;
}

void writeFeaturesWithDescriptorOfOtherType(const GraphPtr &graph)
{
    std::cout << "writeFeaturesWithDescriptorOfOtherType():" << std::endl;

    GraphDataDescriptor<int> gdd;
    std::vector<size_t> idx(3); // Ids of vertices
    idx[0] = 8; idx[1] = 5; idx[2] = 7;

    graph->getVertexFeatures(idx, writeOnly, gdd);
    int *writeOnlyBlockPtr = gdd.getBlockPtr();
    writeOnlyBlockPtr[0] = 5;
    writeOnlyBlockPtr[1] = 6;
    writeOnlyBlockPtr[2] = 11;
    graph->releaseVertexFeatures(gdd);

    graph->getVertexFeatures(idx, readOnly, gdd);
    std::cout << "gdd.size() = " << gdd.size() << std::endl;
    std::cout << "gdd[0]     = " << gdd.getBlockPtr()[0] << std::endl;
    std::cout << "gdd[1]     = " << gdd.getBlockPtr()[1] << std::endl;
    std::cout << "gdd[2]     = " << gdd.getBlockPtr()[2] << std::endl;
    graph->releaseVertexFeatures(gdd);

    std::cout << std::endl;
}

int main(int argc, char const *argv[])
{
    const size_t numberOfVertices = 10;
    AdjacencyListGraphPtr graph = AdjacencyListGraph::create<float>(numberOfVertices);

    graph->setNumberOfAdjacentVertices(0, 5);
    graph->setNumberOfAdjacentVertices(3, 4);

    std::cout << "graph->getNumberOfVertices() = " << graph->getNumberOfVertices() << std::endl;
    std::cout << "graph->getNumberOfEdges()    = " << graph->getNumberOfEdges() << std::endl;

    writeFeaturesWithDescriptorOfSameType(graph);
    writeFeaturesWithDescriptorOfOtherType(graph);

    return 0;
}
