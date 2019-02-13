/* file: kdtree_knn_classification_model_impl.h */
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
//++
//  Implementation of the class defining the K-Nearest Neighbors (kNN) model
//--
*/

#ifndef __KDTREE_KNN_CLASSIFICATION_MODEL_IMPL_
#define __KDTREE_KNN_CLASSIFICATION_MODEL_IMPL_

#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_model.h"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace interface1
{

struct KDTreeNode
{
    size_t dimension;
    size_t leftIndex;
    size_t rightIndex;
    double cutPoint;
};

class KDTreeTable : public data_management::AOSNumericTable
{
public:
    KDTreeTable(size_t rowCount, services::Status &st) : data_management::AOSNumericTable(sizeof(KDTreeNode), 4, rowCount, st)
    {
        setFeature<size_t> (0, DAAL_STRUCT_MEMBER_OFFSET(KDTreeNode, dimension));
        setFeature<size_t> (1, DAAL_STRUCT_MEMBER_OFFSET(KDTreeNode, leftIndex));
        setFeature<size_t> (2, DAAL_STRUCT_MEMBER_OFFSET(KDTreeNode, rightIndex));
        setFeature<double> (3, DAAL_STRUCT_MEMBER_OFFSET(KDTreeNode, cutPoint));
        st |= allocateDataMemory();
    }
    KDTreeTable(services::Status &st) : KDTreeTable(0, st) {}
};
typedef services::SharedPtr<KDTreeTable> KDTreeTablePtr;
typedef services::SharedPtr<const KDTreeTable> KDTreeTableConstPtr;

class Model::ModelImpl
{
public:
    /**
     * Empty constructor for deserialization
     */
    ModelImpl(size_t nFeatures = 0) : _kdTreeTable(), _rootNodeIndex(0), _lastNodeIndex(0), _data(), _labels(), _nFeatures(nFeatures) {}

    /**
     * Returns the KD-tree table
     * \return KD-tree table
     */
    KDTreeTablePtr getKDTreeTable() { return _kdTreeTable; }

    /**
     * Returns the KD-tree table
     * \return KD-tree table
     */
    KDTreeTableConstPtr getKDTreeTable() const { return _kdTreeTable; }

    /**
     * Sets a KD-tree table
     * \param[in]  value  KD-tree table
     */
    void setKDTreeTable(const KDTreeTablePtr & value) { _kdTreeTable = value; }

    /**
     * Returns the index of KD-tree root node
     * \return Index of KD-tree root node
     */
    size_t getRootNodeIndex() const { return _rootNodeIndex; }

    /**
     * Sets a index of KD-tree root node
     * \param[in]  value  Index of KD-tree root node
     */
    void setRootNodeIndex(size_t value) { _rootNodeIndex = value; }

    /**
     * Returns the index of first part KD-tree last node
     * \return Index of first part KD-tree last node
     */
    size_t getLastNodeIndex() const { return _lastNodeIndex; }

    /**
    *  Sets a index of first part KD-tree last node
    *  \param[in]  value  Index of first part KD-tree last node
    */
    void setLastNodeIndex(size_t value) { _lastNodeIndex = value; }

    /**
     * Returns training data
     * \return Training data
     */
    data_management::NumericTableConstPtr getData() const { return _data; }

    /**
     * Returns training data
     * \return Training data
     */
    data_management::NumericTablePtr getData() { return _data; }

    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        arch->set(_nFeatures);
        arch->set(_rootNodeIndex);
        arch->set(_lastNodeIndex);
        arch->setSharedPtrObj(_kdTreeTable);
        arch->setSharedPtrObj(_data);
        arch->setSharedPtrObj(_labels);

        return services::Status();
    }

    /**
     * Sets a training data
     * \param[in]  value  Training data
     * \param[in]  copy   Flag indicating necessary of data deep copying to avoid direct usage and modification of input data.
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void setData(const data_management::NumericTablePtr & value, bool copy)
    {
        if (!copy)
        {
            _data = value;
        }
        else
        {
            data_management::SOANumericTablePtr tbl(new data_management::SOANumericTable(value->getNumberOfColumns(),
                                                                                         value->getNumberOfRows(),
                                                                                         data_management::DictionaryIface::equal));
            tbl->getDictionary()->setAllFeatures<algorithmFPType>(); // Just to set type of all features. Also, no way to use featuresEqual flag.
            tbl->resize(value->getNumberOfRows());

            tbl->allocateDataMemory();
            data_management::BlockDescriptor<algorithmFPType> destBD, srcBD;
            tbl->getBlockOfRows(0, tbl->getNumberOfRows(), data_management::writeOnly, destBD);
            value->getBlockOfRows(0, value->getNumberOfRows(), data_management::readOnly, srcBD);
            services::daal_memcpy_s(destBD.getBlockPtr(), destBD.getNumberOfColumns() * destBD.getNumberOfRows() * sizeof(algorithmFPType),
                                    srcBD.getBlockPtr(), srcBD.getNumberOfColumns() * srcBD.getNumberOfRows() * sizeof(algorithmFPType));
            tbl->releaseBlockOfRows(destBD);
            value->releaseBlockOfRows(srcBD);
            _data = tbl;
        }
    }

    /**
     * Returns training labels
     * \return Training labels
     */
    data_management::NumericTableConstPtr getLabels() const { return _labels; }

    /**
     * Returns training labels
     * \return Training labels
     */
    data_management::NumericTablePtr getLabels() { return _labels; }

    /**
     * Sets a training data
     * \param[in]  value  Training labels
     * \param[in]  copy   Flag indicating necessary of data deep copying to avoid direct usage and modification of input labels.
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void setLabels(const data_management::NumericTablePtr & value, bool copy)
    {
        if (!copy)
        {
            _labels = value;
        }
        else
        {
            data_management::SOANumericTablePtr tbl(new data_management::SOANumericTable(value->getNumberOfColumns(),
                                                                                         value->getNumberOfRows()));
            tbl->setArray(static_cast<algorithmFPType *>(0), 0); // Just to create the dictionary.
            tbl->getDictionary()->setNumberOfFeatures(value->getNumberOfColumns()); // Sadly, setArray() hides number of features from the dictionary.
            data_management::NumericTableFeature temp;
            temp.setType<algorithmFPType>();
            tbl->getDictionary()->setAllFeatures(temp); // Just to set type of all features. Also, no way to use featuresEqual flag.
            tbl->allocateDataMemory();
            data_management::BlockDescriptor<algorithmFPType> destBD, srcBD;
            tbl->getBlockOfRows(0, tbl->getNumberOfRows(), data_management::writeOnly, destBD);
            value->getBlockOfRows(0, value->getNumberOfRows(), data_management::readOnly, srcBD);
            services::daal_memcpy_s(destBD.getBlockPtr(), destBD.getNumberOfColumns() * destBD.getNumberOfRows() * sizeof(algorithmFPType),
                                    srcBD.getBlockPtr(), srcBD.getNumberOfColumns() * srcBD.getNumberOfRows() * sizeof(algorithmFPType));
            tbl->releaseBlockOfRows(destBD);
            value->releaseBlockOfRows(srcBD);
            _labels = tbl;
        }
    }

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  \return Number of features in the dataset was used on the training stage
     */
    size_t getNumberOfFeatures() const { return _nFeatures; }
private:
    size_t _nFeatures;
    KDTreeTablePtr _kdTreeTable;
    size_t _rootNodeIndex;
    size_t _lastNodeIndex;
    data_management::NumericTablePtr _data;
    data_management::NumericTablePtr _labels;
};

} // namespace interface1

using interface1::KDTreeTable;
using interface1::KDTreeTablePtr;
using interface1::KDTreeTableConstPtr;
using interface1::KDTreeNode;

} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
