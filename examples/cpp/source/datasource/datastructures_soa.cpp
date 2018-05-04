/* file: datastructures_soa.cpp */
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
 * <a name="DAAL-EXAMPLE-CPP-DATASTRUCTURES_SOA">
 * \example datastructures_soa.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;

const char *toString(data_feature_utils::FeatureType v);
const char *toString(data_feature_utils::InternalNumType v);

int main()
{
    std::cout << "Structure of array (SOA) numeric table example" << std::endl << std::endl;

    const size_t firstReadRow = 0;
    const size_t nRead = 3;
    size_t readFeatureIdx;

    /*Example of using an SOA numeric table*/
    const size_t nObservations = 10;
    const size_t nFeatures = 4;
    double dDataSOA[nObservations] = {1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8};
    float  fDataSOA[nObservations] = {3.1f, 3.2f, 3.3f, 3.4f, 3.5f, 3.6f, 3.7f, 3.8f, 3.9f, 4.0f};
    int    iDataSOA[nObservations] = { -10, -20, -30, -40, -50, -60, -70, -80, -90, -100};
    int    cDataSOA[nObservations] = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5};

    /* Construct an SOA numeric table with nObservations rows and nFeatures columns */
    SOANumericTablePtr dataTable = SOANumericTable::create(nFeatures, nObservations);
    checkPtr(dataTable.get());
    dataTable->setArray<int>   (cDataSOA, 0);
    dataTable->setArray<float> (fDataSOA, 1);
    dataTable->setArray<double>(dDataSOA, 2);
    dataTable->setArray<int>   (iDataSOA, 3);

    /* Read a block of rows */
    BlockDescriptor<double> doubleBlock;
    dataTable->getBlockOfRows(firstReadRow, nRead, readOnly, doubleBlock);
    printArray<double>(doubleBlock.getBlockPtr(), nFeatures, doubleBlock.getNumberOfRows(), "Print SOA data structures as double:");
    dataTable->releaseBlockOfRows(doubleBlock);

    /* Read a feature (column) and write a new value into it */
    readFeatureIdx = 0;
    BlockDescriptor<int> intBlock;
    dataTable->getBlockOfColumnValues(readFeatureIdx, firstReadRow, nObservations, readOnly, intBlock);
    printArray<int>(intBlock.getBlockPtr(), 1, intBlock.getNumberOfRows(), "Print the first feature of SOA:");
    dataTable->releaseBlockOfColumnValues(intBlock);

    /* Get the dictionary and the number of features */
    NumericTableDictionaryPtr pDictionary = dataTable->getDictionarySharedPtr();
    std::cout << "Number of features in table: " << pDictionary->getNumberOfFeatures() << std::endl;
    std::cout << std::endl;

    std::cout << "Default type in autogenerated dictionary:" << std::endl;
    for(size_t i = 0; i < nFeatures; i++)
    {
        data_feature_utils::FeatureType featureType = (*pDictionary)[i].featureType;
        std::cout << "Type of " << i << " feature: " ;
        std::cout << toString(featureType) << std::endl;
    }
    std::cout << std::endl;

    /* Modify the dictionary information about data */
    NumericTableFeature &categoricalFeature = (*pDictionary)[0];
    categoricalFeature.featureType = data_feature_utils::DAAL_CATEGORICAL;

    std::cout << "Modified type in the dictionary:" << std::endl;
    for(size_t i = 0; i < nFeatures; i++)
    {
        data_feature_utils::FeatureType featureType = (*pDictionary)[i].featureType;
        std::cout << "Type of " << i << " feature: " ;
        std::cout << toString(featureType) << std::endl;
    }
    std::cout << std::endl;

    return 0;
}

const char *toString(data_feature_utils::FeatureType v)
{
    switch (v)
    {
    case data_feature_utils::DAAL_CATEGORICAL:   return "DAAL_CATEGORICAL";
    case data_feature_utils::DAAL_ORDINAL:       return "DAAL_ORDINAL";
    case data_feature_utils::DAAL_CONTINUOUS:    return "DAAL_CONTINUOUS";
    default: return "[Unknown FeatureType]";
    }
}
