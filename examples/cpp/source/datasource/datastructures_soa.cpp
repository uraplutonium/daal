/* file: datastructures_soa.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
