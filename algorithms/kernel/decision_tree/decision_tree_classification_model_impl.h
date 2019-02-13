/* file: decision_tree_classification_model_impl.h */
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
//  Implementation of the class defining the Decision tree model
//--
*/

#ifndef __DECISION_TREE_CLASSIFICATION_MODEL_IMPL_
#define __DECISION_TREE_CLASSIFICATION_MODEL_IMPL_

#include "algorithms/decision_tree/decision_tree_classification_model.h"
#include "classifier_model_impl.h"
#include "service_defines.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace classification
{
namespace interface1
{

struct DecisionTreeNode
{
    size_t dimension;
    size_t leftIndexOrClass;
    double cutPoint;
};

class DecisionTreeTable : public data_management::AOSNumericTable
{
public:
    DecisionTreeTable(size_t rowCount, services::Status &st) : data_management::AOSNumericTable(sizeof(DecisionTreeNode), 3, rowCount, st)
    {
        setFeature<size_t> (0, DAAL_STRUCT_MEMBER_OFFSET(DecisionTreeNode, dimension));
        setFeature<size_t> (1, DAAL_STRUCT_MEMBER_OFFSET(DecisionTreeNode, leftIndexOrClass));
        setFeature<double> (2, DAAL_STRUCT_MEMBER_OFFSET(DecisionTreeNode, cutPoint));
        st |= allocateDataMemory();
    }
    DecisionTreeTable(services::Status &st) : DecisionTreeTable(0, st) {}
};

typedef services::SharedPtr<DecisionTreeTable> DecisionTreeTablePtr;
typedef services::SharedPtr<const DecisionTreeTable> DecisionTreeTableConstPtr;

class Model::ModelImpl : public classifier::internal::ModelImpl
{
    typedef classifier::internal::ModelImpl super;
    typedef services::Collection<size_t> NodeIdxArray;

    static bool visitSplit(size_t iRowInTable, size_t level, const DecisionTreeNode* aNode, classifier::TreeNodeVisitor& visitor)
    {
        const DecisionTreeNode& n = aNode[iRowInTable];
        return visitor.onSplitNode(level, n.dimension, n.cutPoint);
    }

    static bool visitLeaf(size_t iRowInTable, size_t level, const DecisionTreeNode* aNode, classifier::TreeNodeVisitor& visitor)
    {
        const DecisionTreeNode& n = aNode[iRowInTable];
        return visitor.onLeafNode(level, n.leftIndexOrClass);
    }

    static bool visitSplit(size_t iRowInTable, size_t level, tree_utils::SplitNodeDescriptor& descSplit, const DecisionTreeNode* aNode, const double *imp,
        const int *nodeSamplesCount, tree_utils::classification::TreeNodeVisitor& visitor)
    {
        const DecisionTreeNode& n                       = aNode[iRowInTable];
        if(imp)              descSplit.impurity         = imp[iRowInTable];
        if(nodeSamplesCount) descSplit.nNodeSampleCount = (size_t)(nodeSamplesCount[iRowInTable]);
        descSplit.featureIndex                          = n.dimension;
        descSplit.featureValue                          = n.cutPoint;
        descSplit.level                                 = level;
        return visitor.onSplitNode(descSplit);
    }

    static bool visitLeaf(size_t iRowInTable, size_t level, tree_utils::classification::LeafNodeDescriptor& descLeaf, const DecisionTreeNode* aNode, const double *imp,
        const int *nodeSamplesCount, tree_utils::classification::TreeNodeVisitor& visitor)
    {
        const DecisionTreeNode& n                      = aNode[iRowInTable];
        if(imp)              descLeaf.impurity         = imp[iRowInTable];
        if(nodeSamplesCount) descLeaf.nNodeSampleCount = (size_t)(nodeSamplesCount[iRowInTable]);
        descLeaf.level                                 = level;
        descLeaf.label                                 = n.leftIndexOrClass;
        return visitor.onLeafNode(descLeaf);
    }

public:
    /**
     * Empty constructor for deserialization
     */
    ModelImpl(size_t nFeatures = 0) : super(nFeatures), _TreeTable() {}

    /**
     * Returns the decision tree table
     * \return decision tree table
     */
    DecisionTreeTablePtr getTreeTable() { return _TreeTable; }

    /**
     * Returns the decision tree table
     * \return decision tree table
     */
    DecisionTreeTableConstPtr getTreeTable() const { return _TreeTable; }

    /**
     * Sets a decision tree table
     * \param[in]  value  decision tree table
     */
    void setTreeTable(const DecisionTreeTablePtr & value) { _TreeTable = value; }

    void setImpTable(const services::SharedPtr<data_management::HomogenNumericTable<double> > & value) { _impTable = value; }

    void setNodeSmplCntTable(const services::SharedPtr<data_management::HomogenNumericTable<int> > & value) { _nodeSampleCountTable = value; }

    const double* getImpVals() const
    {
        return _impTable ? _impTable->getArray() : nullptr;
    }

    const int* getNodeSampleCount() const
    {
        return _nodeSampleCountTable ? _nodeSampleCountTable->getArray() : nullptr;
    }

    void traverseDF(classifier::TreeNodeVisitor& visitor) const
    {
        const DecisionTreeNode* aNode = (const DecisionTreeNode*)_TreeTable->getArray();
        if(aNode)
        {
            auto onSplitNodeFunc = [&aNode, &visitor](size_t iRowInTable, size_t level) -> bool {
                return visitSplit(iRowInTable, level, aNode, visitor);
            };

            auto onLeafNodeFunc = [&aNode, &visitor](size_t iRowInTable, size_t level) -> bool {
                return visitLeaf(iRowInTable, level, aNode, visitor);
            };

            traverseNodesDF(0, 0, aNode, onSplitNodeFunc, onLeafNodeFunc);
        }
    }

    void traverseBF(classifier::TreeNodeVisitor& visitor) const
    {
        const DecisionTreeNode* aNode = (const DecisionTreeNode*)_TreeTable->getArray();
        NodeIdxArray aCur;//nodes of current layer
        NodeIdxArray aNext;//nodes of next layer
        if(aNode)
        {
            aCur.push_back(0);
            auto onSplitNodeFunc = [&aNode, &visitor](size_t iRowInTable, size_t level) -> bool {
                return visitSplit(iRowInTable, level, aNode, visitor);
            };

            auto onLeafNodeFunc = [&aNode, &visitor](size_t iRowInTable, size_t level) -> bool {
                return visitLeaf(iRowInTable, level, aNode, visitor);
            };

            traverseNodesBF(0, aCur, aNext, aNode, onSplitNodeFunc, onLeafNodeFunc);
        }
    }

    void traverseDFS(tree_utils::classification::TreeNodeVisitor& visitor) const
    {
        const DecisionTreeNode* aNode = (const DecisionTreeNode*)_TreeTable->getArray();
        const double *imp = getImpVals();
        const int *nodeSamplesCount = getNodeSampleCount();
        if(aNode)
        {
            tree_utils::SplitNodeDescriptor descSplit;
            tree_utils::classification::LeafNodeDescriptor descLeaf;

            auto onSplitNodeFunc = [&descSplit, &aNode, &imp, &nodeSamplesCount, &visitor](size_t iRowInTable, size_t level) -> bool {
                return visitSplit(iRowInTable, level, descSplit, aNode, imp, nodeSamplesCount, visitor);
            };

            auto onLeafNodeFunc = [&descLeaf, &aNode, &imp, &nodeSamplesCount, &visitor](size_t iRowInTable, size_t level) -> bool {
                return visitLeaf(iRowInTable, level, descLeaf, aNode, imp, nodeSamplesCount, visitor);
            };

            traverseNodesDF(0, 0, aNode, onSplitNodeFunc, onLeafNodeFunc);
        }
    }

    void traverseBFS(tree_utils::classification::TreeNodeVisitor& visitor) const
    {
        const DecisionTreeNode* aNode = (const DecisionTreeNode*)_TreeTable->getArray();
        const double *imp = getImpVals();
        const int *nodeSamplesCount = getNodeSampleCount();
        NodeIdxArray aCur;//nodes of current layer
        NodeIdxArray aNext;//nodes of next layer
        if(aNode)
        {
            tree_utils::SplitNodeDescriptor descSplit;
            tree_utils::classification::LeafNodeDescriptor descLeaf;

            auto onSplitNodeFunc = [&descSplit, &aNode, &imp, &nodeSamplesCount, &visitor](size_t iRowInTable, size_t level) -> bool {
                return visitSplit(iRowInTable, level, descSplit, aNode, imp, nodeSamplesCount, visitor);
            };

            auto onLeafNodeFunc = [&descLeaf, &aNode, &imp, &nodeSamplesCount, &visitor](size_t iRowInTable, size_t level) -> bool {
                return visitLeaf(iRowInTable, level, descLeaf, aNode, imp, nodeSamplesCount, visitor);
            };

            aCur.push_back(0);
            traverseNodesBF(0, aCur, aNext, aNode, onSplitNodeFunc, onLeafNodeFunc);
        }
    }

    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch, int daalVersion = INTEL_DAAL_VERSION)
    {
        super::serialImpl<Archive, onDeserialize>(arch);
        arch->setSharedPtrObj(_TreeTable);

        if((daalVersion >= COMPUTE_DAAL_VERSION(2019, 0, 0)))
        {
            arch->setSharedPtrObj(_impTable);
            arch->setSharedPtrObj(_nodeSampleCountTable);
        }

        return services::Status();
    }

private:
    DecisionTreeTablePtr _TreeTable;
    services::SharedPtr<data_management::HomogenNumericTable<double> > _impTable;
    services::SharedPtr<data_management::HomogenNumericTable<int> >    _nodeSampleCountTable;

    template <typename OnSplitFunctor, typename OnLeafFunctor>
    bool traverseNodesDF(size_t level, size_t iRowInTable, const DecisionTreeNode* aNode, OnSplitFunctor &visitSplit, OnLeafFunctor &visitLeaf) const
    {
        const DecisionTreeNode& n = aNode[iRowInTable];
        if(n.dimension != static_cast<size_t>(-1))
        {
            if(!visitSplit(iRowInTable, level))
                return false; //do not continue traversing
            ++level;
            size_t leftIdx = n.leftIndexOrClass;
            size_t rightIdx = leftIdx + 1;
            if(!traverseNodesDF(level, leftIdx, aNode, visitSplit, visitLeaf))
                return false;
            return traverseNodesDF(level, rightIdx, aNode, visitSplit, visitLeaf);
        }
        return visitLeaf(iRowInTable, level);
    }

    template <typename OnSplitFunctor, typename OnLeafFunctor>
    bool traverseNodesBF(size_t level, NodeIdxArray& aCur,
        NodeIdxArray& aNext, const DecisionTreeNode* aNode, OnSplitFunctor &visitSplit, OnLeafFunctor &visitLeaf) const
    {
        for(size_t i = 0; i < aCur.size(); ++i)
        {
            for(size_t j = 0; j < (level ? 2 : 1); ++j)
            {
                size_t iRowInTable = aCur[i] + j;
                const DecisionTreeNode& n = aNode[iRowInTable];
                if(n.dimension != static_cast<size_t>(-1))
                {
                    if(!visitSplit(iRowInTable, level))
                        return false; //do not continue traversing
                    aNext.push_back(n.leftIndexOrClass);
                }
                else
                {
                    if(!visitLeaf(iRowInTable, level))
                        return false; //do not continue traversing
                }
            }
        }
        aCur.clear();
        if(!aNext.size())
            return true;//done
        return traverseNodesBF(level + 1, aNext, aCur, aNode, visitSplit, visitLeaf);
    }
};

} // namespace interface1

using interface1::DecisionTreeTable;
using interface1::DecisionTreeNode;
using interface1::DecisionTreeTablePtr;
using interface1::DecisionTreeTableConstPtr;

} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal

#endif
