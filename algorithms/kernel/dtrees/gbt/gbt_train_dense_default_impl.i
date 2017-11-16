/* file: gbt_train_dense_default_impl.i */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//++
//  Implementation of auxiliary functions for gradient boosted trees training
//  (defaultDense) method.
//--
*/

#ifndef __GBT_TRAIN_DENSE_DEFAULT_IMPL_I__
#define __GBT_TRAIN_DENSE_DEFAULT_IMPL_I__

#include "dtrees_model_impl.h"
#include "dtrees_train_data_helper.i"
#include "dtrees_predict_dense_default_impl.i"
#include "gbt_internal.h"
#include "threading.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace training
{
namespace internal
{

using namespace daal::algorithms::dtrees::training::internal;
using namespace daal::algorithms::gbt::internal;

template <CpuType cpu>
void deleteTables(dtrees::internal::DecisionTreeTable** aTbl, size_t n)
{
    for(size_t i = 0; i < n; ++i)
        delete aTbl[i];
}

//////////////////////////////////////////////////////////////////////////////////////////
// Data helper class for regression
//////////////////////////////////////////////////////////////////////////////////////////
template<typename algorithmFPType, CpuType cpu>
class OrderedRespHelper : public DataHelper<algorithmFPType, algorithmFPType, cpu>
{
public:
    typedef algorithmFPType TResponse;
    typedef DataHelper<algorithmFPType, algorithmFPType, cpu> super;

public:
    OrderedRespHelper(const dtrees::internal::SortedFeaturesHelper* sortedFeatHelper, size_t dummy) : super(sortedFeatHelper){}
};

//////////////////////////////////////////////////////////////////////////////////////////
// Service class, pair (gradient, hessian) of algorithmFPType values
//////////////////////////////////////////////////////////////////////////////////////////
template<typename algorithmFPType, CpuType cpu>
struct gh
{
    algorithmFPType g; //gradient
    algorithmFPType h; //hessian
    gh() : g(0), h(0){}
    gh(algorithmFPType _g, algorithmFPType _h) : g(_g), h(_h){}
    gh(const gh& o) : g(o.g), h(o.h){}
    gh(const gh& total, const gh& part) : g(total.g - part.g), h(total.h - part.h){}
    gh& operator =(const gh& o) { g = o.g;  h = o.h; return *this; }
    void reset(algorithmFPType _g, algorithmFPType _h) { g = _g;  h = _h; }
    void add(const gh& o) { g += o.g;  h += o.h; }
    algorithmFPType value(algorithmFPType regLambda) const { return (g / (h + regLambda))*g; }
};

template<typename algorithmFPType, CpuType cpu>
gh<algorithmFPType, cpu> operator -(const gh<algorithmFPType, cpu>& a, const gh<algorithmFPType, cpu>& b)
{
    return gh<algorithmFPType, cpu>(a.g - b.g, a.h - b.h);
}

//////////////////////////////////////////////////////////////////////////////////////////
// Impurity data
//////////////////////////////////////////////////////////////////////////////////////////
template<typename algorithmFPType, CpuType cpu>
using ImpurityData = gh<algorithmFPType, cpu>;

//////////////////////////////////////////////////////////////////////////////////////////
// Base class for loss function L(y,f), where y is a response value,
// f is its current approximation
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class LossFunction : public Base
{
public:
    virtual void getGradients(size_t n,
        const algorithmFPType* y, const algorithmFPType* f,
        const IndexType* sampleInd,
        algorithmFPType* gh) = 0;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Squared loss function, L(y,f)=1/2([y-f(x)]^2)
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class SquaredLoss : public LossFunction<algorithmFPType, cpu>
{
public:
    virtual void getGradients(size_t n, const algorithmFPType* y, const algorithmFPType* f,
        const IndexType* sampleInd,
        algorithmFPType* gh) DAAL_C11_OVERRIDE
    {
        if(sampleInd)
        {
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for(size_t i = 0; i < n; ++i)
            {
                gh[2 * i] = f[sampleInd[i]] - y[i]; //gradient
                gh[2 * i + 1] = 1; //hessian
            }
        }
        else
        {
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for(size_t i = 0; i < n; ++i)
            {
                gh[2 * i] = f[i] - y[i]; //gradient
                gh[2 * i + 1] = 1; //hessian
            }
        }
    }
};

//////////////////////////////////////////////////////////////////////////////////////////
// Base class for algo-specific data
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class AlgoBase : public Base
{
public:
    typedef OrderedRespHelper<algorithmFPType, cpu> DataHelperType;
    typedef ImpurityData<algorithmFPType, cpu> ImpurityType;
    virtual void getInitialImpurity(ImpurityType& val, size_t nSamples) = 0;
    virtual double computeLeafWeightUpdateF(const IndexType* idx, size_t n, algorithmFPType* f,
        const IndexType* sampleInd, const ImpurityType& impBase, const Parameter& par) = 0;
    virtual services::Status init(const DataHelperType& inputData) = 0;
    virtual void step(size_t n, const algorithmFPType* y, const algorithmFPType* f,
        const IndexType* aSample, LossFunction<algorithmFPType, cpu>& func) = 0;

    void setCurrentTree(size_t iTree) { _curTree = iTree; }
    size_t nTrees() const { return _nTrees; }

protected:
    AlgoBase(size_t n) : _curTree(0), _nTrees(n){}

protected:
    size_t _nTrees;
    size_t _curTree;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Base memory helper class
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class MemHelperBase : public Base
{
protected:
    MemHelperBase(size_t nFeaturesIdx) : _nFeaturesIdx(nFeaturesIdx){}

public:
    typedef gh<algorithmFPType, cpu> ghType;
    typedef TVector<IndexType, cpu, DefaultAllocator<cpu> > IndexTypeVector;
    typedef TVector<ghType, cpu, DefaultAllocator<cpu> > ghTypeVector;
    typedef TVector<algorithmFPType, cpu, DefaultAllocator<cpu> > algorithmFPTypeVector;

    virtual bool init() = 0;
    //get buffer for the indices of features to be used for the split at the current level
    virtual IndexType* getFeatureSampleBuf() = 0;
    //release the buffer
    virtual void releaseFeatureSampleBuf(IndexType* buf) = 0;

    //get buffer for the indices of an indexed feature values to be used for the split at the current level
    virtual IndexTypeVector* getIndexedFeatureCountsBuf(size_t size) = 0;
    //release the buffer
    virtual void releaseIndexedFeatureCountsBuf(IndexTypeVector*) = 0;

    //get buffer for gh values to be used in split finding (sorted features)
    virtual ghTypeVector* getGradBuf(size_t size) = 0;
    //release the buffer
    virtual void releaseGradBuf(ghTypeVector*) = 0;

    //get buffer for the feature values to be used for the split at the current level
    virtual algorithmFPTypeVector* getFeatureValueBuf(size_t size) = 0;
    //release the buffer
    virtual void releaseFeatureValueBuf(algorithmFPTypeVector* buf) = 0;

    //get buffer for the indexes of the sorted feature values to be used for the split at the current level
    virtual IndexTypeVector* getSortedFeatureIdxBuf(size_t size) = 0;
    //release the buffer
    virtual void releaseSortedFeatureIdxBuf(IndexTypeVector* p) = 0;

protected:
    const size_t _nFeaturesIdx;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Implementation of memory helper for sequential version
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class MemHelperSeq : public MemHelperBase<algorithmFPType, cpu>
{
public:
    typedef MemHelperBase<algorithmFPType, cpu> super;
    MemHelperSeq(size_t nFeaturesIdx, size_t nDiffFeaturesMax, size_t nFeatureValuesMax) :
        super(nFeaturesIdx), _featureSample(nFeaturesIdx),
        _aIndexedFeatureCounts(nDiffFeaturesMax), _aGH(nDiffFeaturesMax), _aFeatureValue(nFeatureValuesMax){}

    virtual bool init() DAAL_C11_OVERRIDE
    {
        return (!_featureSample.size() || _featureSample.get()) && //not required to allocate or allocated
            (!_aIndexedFeatureCounts.size() || (_aIndexedFeatureCounts.get() && _aGH.get())) &&
            (!_aFeatureValue.size() || _aFeatureValue.get());
    }

    virtual IndexType* getFeatureSampleBuf() DAAL_C11_OVERRIDE { return _featureSample.get(); }
    virtual void releaseFeatureSampleBuf(IndexType* buf) DAAL_C11_OVERRIDE{}

    virtual typename super::IndexTypeVector* getIndexedFeatureCountsBuf(size_t size) DAAL_C11_OVERRIDE{ return &_aIndexedFeatureCounts; }
    virtual void releaseIndexedFeatureCountsBuf(typename super::IndexTypeVector*) DAAL_C11_OVERRIDE{}

    virtual typename super::ghTypeVector* getGradBuf(size_t size) DAAL_C11_OVERRIDE{ return &_aGH; }
    virtual void releaseGradBuf(typename super::ghTypeVector*) DAAL_C11_OVERRIDE{}

    //get buffer for the feature values to be used for the split at the current level
    virtual typename super::algorithmFPTypeVector* getFeatureValueBuf(size_t size) DAAL_C11_OVERRIDE
    { DAAL_ASSERT(_aFeatureValue.size() >= size);  return &_aFeatureValue; }
    //release the buffer
    virtual void releaseFeatureValueBuf(typename super::algorithmFPTypeVector* buf) DAAL_C11_OVERRIDE{}

    virtual typename super::IndexTypeVector* getSortedFeatureIdxBuf(size_t size) DAAL_C11_OVERRIDE
    { DAAL_ASSERT(false);  return nullptr; }//should never be called

    virtual void releaseSortedFeatureIdxBuf(typename super::IndexTypeVector* p) DAAL_C11_OVERRIDE{}

protected:
    typename super::IndexTypeVector _featureSample;
    typename super::IndexTypeVector _aIndexedFeatureCounts;
    typename super::ghTypeVector _aGH;
    typename super::algorithmFPTypeVector _aFeatureValue;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Service class, keeps an array in ls and resizes it in local()
//////////////////////////////////////////////////////////////////////////////////////////
template<typename VectorType>
class lsVector : public ls<VectorType*>
{
public:
    typedef ls<VectorType*> super;
    explicit lsVector() : super([=]()->VectorType*{ return new VectorType(); }){}
    ~lsVector() { this->reduce([](VectorType* ptr) { if(ptr) delete ptr; }); }
    VectorType* local(size_t size)
    {
        auto ptr = super::local();
        if(ptr && (ptr->size() < size))
        {
            ptr->reset(size);
            if(!ptr->get())
            {
                this->release(ptr);
                ptr = nullptr;
            }
        }
        return ptr;
    }
};

//////////////////////////////////////////////////////////////////////////////////////////
// Implementation of memory helper for threaded version
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class MemHelperThr : public MemHelperBase<algorithmFPType, cpu>
{
public:
    typedef MemHelperBase<algorithmFPType, cpu> super;
    MemHelperThr(size_t nFeaturesIdx) : super(nFeaturesIdx),
        _lsFeatureSample([=]()->IndexType*{ return services::internal::service_scalable_malloc<IndexType, cpu>(this->_nFeaturesIdx); })
    {
    }
    ~MemHelperThr()
    {
        _lsFeatureSample.reduce([](IndexType* ptr){ if(ptr) services::internal::service_scalable_free<IndexType, cpu>(ptr); });
    }
public:
    virtual bool init() DAAL_C11_OVERRIDE { return true; }
    virtual IndexType* getFeatureSampleBuf() DAAL_C11_OVERRIDE
    {
        return _lsFeatureSample.local();
    }

    virtual void releaseFeatureSampleBuf(IndexType* p) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(p);
        _lsFeatureSample.release(p);
    }

    virtual typename super::IndexTypeVector* getIndexedFeatureCountsBuf(size_t size) DAAL_C11_OVERRIDE
    {
        return _lsIndexedFeatureCountsBuf.local(size);
    }
    virtual void releaseIndexedFeatureCountsBuf(typename super::IndexTypeVector* p) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(p);
        _lsIndexedFeatureCountsBuf.release(p);
    }

    virtual typename super::ghTypeVector* getGradBuf(size_t size) DAAL_C11_OVERRIDE
    {
        return _lsGHBuf.local(size);
    }
    virtual void releaseGradBuf(typename super::ghTypeVector* p) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(p);
        _lsGHBuf.release(p);
    }

    //get buffer for the feature values to be used for the split at the current level
    virtual typename super::algorithmFPTypeVector* getFeatureValueBuf(size_t size) DAAL_C11_OVERRIDE
    {
        return _lsFeatureValueBuf.local(size);
    }

    //release the buffer
    virtual void releaseFeatureValueBuf(typename super::algorithmFPTypeVector* p) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(p);
        _lsFeatureValueBuf.release(p);
    }

    virtual typename super::IndexTypeVector* getSortedFeatureIdxBuf(size_t size) DAAL_C11_OVERRIDE
    {
        return _lsSortedFeatureIdxBuf.local(size);
    }

    virtual void releaseSortedFeatureIdxBuf(typename super::IndexTypeVector* p) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(p);
        _lsSortedFeatureIdxBuf.release(p);
    }

protected:
    ls<IndexType*> _lsFeatureSample;
    lsVector<typename super::IndexTypeVector> _lsIndexedFeatureCountsBuf;
    lsVector<typename super::ghTypeVector> _lsGHBuf;
    lsVector<typename super::algorithmFPTypeVector> _lsFeatureValueBuf;
    lsVector<typename super::IndexTypeVector> _lsSortedFeatureIdxBuf;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Job to be performed in one node
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
struct SplitJob
{
public:
    typedef dtrees::internal::TreeImpRegression<> TreeType;
    typedef typename TreeType::NodeType NodeType;
    typedef ImpurityData<algorithmFPType, cpu> ImpurityType;

    SplitJob(const SplitJob& o): iStart(o.iStart), n(o.n), level(o.level), imp(o.imp), res(o.res){}
    SplitJob(size_t _iStart, size_t _n, size_t _level, const ImpurityType& _imp, NodeType::Base*& _res) :
        iStart(_iStart), n(_n), level(_level), imp(_imp), res(_res){}
public:
    const size_t iStart;
    const size_t n;
    const size_t level;
    const ImpurityType imp;
    NodeType::Base*& res;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Base task class. Implements general pipeline of tree building
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, typename DataHelper, typename AlgoType, CpuType cpu>
class TrainBatchTaskBase
{
public:
    typedef dtrees::internal::TreeImpRegression<> TreeType;
    typedef typename TreeType::NodeType NodeType;
    typedef ImpurityData<algorithmFPType, cpu> ImpurityType;
    typedef SplitData<algorithmFPType, ImpurityType> SplitDataType;
    typedef LossFunction<algorithmFPType, cpu> LossFunctionType;
    typedef SplitJob<algorithmFPType, cpu> SplitJobType;

    struct SplitTask : public SplitJobType
    {
        typedef TrainBatchTaskBase<algorithmFPType, DataHelper, AlgoType, cpu> Task;
        typedef SplitJobType super;

        SplitTask(const SplitTask& o) : super(o), _task(o._task){}
        SplitTask(Task& task, size_t _iStart, size_t _n, size_t _level, const ImpurityType& _imp, NodeType::Base*& _res) :
            super(_iStart, _n, _level, _imp, _res), _task(task){}
        Task& _task;
        void operator()()
        {
            _task._nParallelNodes.inc();
            _task.buildSplit(*this);
            _task._nParallelNodes.dec();
        }
    };

    class BestSplit
    {
    public:
        BestSplit(SplitDataType& split, Mutex* mt) :
            _split(split), _mt(mt), _iIndexedFeatureSplitValue(-1), _iFeatureInSample(-1){}
        algorithmFPType impurityDecrease() const { return _split.impurityDecrease; }

        void update(const SplitDataType& split, int iIndexedFeatureSplitValue, int iFeatureInSample)
        {
            if(_mt)
            {
                _mt->lock();
                updateImpl(split, iIndexedFeatureSplitValue, iFeatureInSample);
                _mt->unlock();
            }
            else
                updateImpl(split, iIndexedFeatureSplitValue, iFeatureInSample);
        }

        void update(const SplitDataType& split, int iFeatureInSample, IndexType* bestSplitIdx, const IndexType* aIdx, size_t n)
        {
            if(_mt)
            {
                _mt->lock();
                if(updateImpl(split, -1, iFeatureInSample))
                    tmemcpy<IndexType, cpu>(bestSplitIdx, aIdx, n);
                _mt->unlock();
            }
            else
            {
                if(updateImpl(split, -1, iFeatureInSample))
                    tmemcpy<IndexType, cpu>(bestSplitIdx, aIdx, n);
            }
        }

        int iIndexedFeatureSplitValue() const { return _iIndexedFeatureSplitValue; }
        int iFeatureInSample() const { return _iFeatureInSample; }
        bool isThreadedMode() const { return _mt != nullptr;  }

    private:
        bool updateImpl(const SplitDataType& split, int iIndexedFeatureSplitValue, int iFeatureInSample)
        {
            if(split.impurityDecrease > impurityDecrease())
            {
                _iFeatureInSample = (int)iFeatureInSample;
                split.copyTo(_split);
                _iIndexedFeatureSplitValue = iIndexedFeatureSplitValue;
                return true;
            }
            return false;
        }
    private:
        SplitDataType& _split;
        Mutex* _mt;
        volatile int _iIndexedFeatureSplitValue;
        volatile int _iFeatureInSample;
    };

    const LossFunctionType* lossFunc() const { DAAL_ASSERT(_loss); return _loss; }
    LossFunctionType* lossFunc() { DAAL_ASSERT(_loss); return _loss; }

    services::Status run(dtrees::internal::DecisionTreeTable** aTbl, size_t nTrees, size_t iIteration);
    services::Status init()
    {
        delete _loss;
        _loss = nullptr;
        initLossFunc();
        _aSample.reset(_nSamples);
        const auto nRows = _data->getNumberOfRows();
        if(_nSamples < nRows)
        {
            _aSampleToF.reset(nRows);
            DAAL_CHECK_MALLOC(_aSampleToF.get());
        }
        const auto nF = nRows*_algo.nTrees();
        _aF.reset(nF);
        _aBestSplitIdxBuf.reset(_nSamples);
        DAAL_CHECK_MALLOC(_aSample.get() && _aF.get() && _dataHelper.reset(_nSamples) && _aBestSplitIdxBuf.get());
        if(isParallelNodes() && !_taskGroup)
            DAAL_CHECK_MALLOC((_taskGroup = new daal::task_group()));
        DAAL_CHECK_MALLOC(initMemHelper());
        return _algo.init(_dataHelper);
    }
    //number of trees per iteration
    size_t nTrees() const { return _algo.nTrees(); }

protected:
    typedef dtrees::internal::TVector<algorithmFPType, cpu, DefaultAllocator<cpu>> algorithmFPTypeArray;
    typedef dtrees::internal::TVector<IndexType, cpu, DefaultAllocator<cpu>> IndexTypeArray;

    TrainBatchTaskBase(const NumericTable *x, const NumericTable *y, const Parameter& par,
        const dtrees::internal::FeatureTypeHelper<cpu>& featHelper,
        const dtrees::internal::SortedFeaturesHelper* sortedFeatHelper,
        engines::internal::BatchBaseImpl& engine,
        size_t nClasses) :
        _data(x), _resp(y), _par(par), _engine(engine), _nClasses(nClasses),
        _nSamples(par.observationsPerTreeFraction*x->getNumberOfRows()),
        _nFeaturesPerNode(par.featuresPerNode ? par.featuresPerNode : x->getNumberOfColumns()),
        _dataHelper(sortedFeatHelper, nClasses),
        _featHelper(featHelper),
        _accuracy(daal::data_feature_utils::internal::EpsilonVal<algorithmFPType, cpu>::get()),
        _initialF(0.),
        _algo(nClasses > 2 ? nClasses : 1, par.minObservationsInLeafNode, par.lambda),
        _loss(nullptr),
        _memHelper(nullptr),
        _taskGroup(nullptr),
        _nThreadsMax(threader_get_max_threads_number()),
        _nParallelNodes(0)
    {
        _bThreaded = ((_nThreadsMax > 1) && ((par.internalOptions & parallelAll) != 0));
        _bParallelFeatures = _bThreaded && ((par.internalOptions & parallelFeatures) != 0);
        _bParallelNodes = _bThreaded && ((par.internalOptions & parallelNodes) != 0);
    }
    ~TrainBatchTaskBase()
    {
        delete _loss;
        delete _memHelper;
        delete _taskGroup;
    }
    void buildSplit(SplitJobType& job);
    int numAvailableThreads() const { auto n = _nParallelNodes.get(); return _nThreadsMax > n ? _nThreadsMax - n : 0; }
    void initializeF(algorithmFPType initValue)
    {
        const auto nRows = _data->getNumberOfRows();
        const auto nF = nRows*_algo.nTrees();
        //initialize f. TODO: input argument
        algorithmFPType* pf = f();
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < nF; ++i)
            pf[i] = initValue;
    }

protected:
    virtual void initLossFunc() = 0;
    virtual bool getInitialF(const algorithmFPType* py, size_t n, algorithmFPType& val) { return false; }
    bool initMemHelper();
    size_t nFeatures() const { return _data->getNumberOfColumns(); }
    bool isThreaded() const { return _bThreaded; }
    bool isParallelFeatures() const { return _bParallelFeatures; }
    bool isParallelNodes() const { return _bParallelNodes; }
    //loss function arguments (current estimation of y)
    algorithmFPType* f() { return _aF.get(); }
    const algorithmFPType* f() const { return _aF.get(); }

    NodeType::Base* buildRoot(size_t iTree)
    {
        ImpurityType imp;
        _algo.setCurrentTree(iTree);
        _algo.getInitialImpurity(imp, _nSamples);
        typename NodeType::Base* res = buildLeaf(0, _nSamples, 0, imp);
        if(res)
            return res;
        _nParallelNodes.inc();
        SplitJobType job(0, _nSamples, 0, imp, res);
        buildSplit(job);
        if(_taskGroup)
            _taskGroup->wait();
        return res;
    }

    void buildNode(size_t iStart, size_t n, size_t level, const ImpurityType& imp, NodeType::Base*& res);
    NodeType::Base* buildLeaf(size_t iStart, size_t n, size_t level, const ImpurityType& imp)
    {
        return terminateCriteria(n, level, imp) ? makeLeaf(_aSample.get() + iStart, n, imp) : nullptr;
    }

    IndexType* bestSplitIdxBuf() const { return _aBestSplitIdxBuf.get(); }
    bool terminateCriteria(size_t nSamples, size_t level, const ImpurityType& imp) const
    {
        return ((nSamples < 2 * _par.minObservationsInLeafNode) ||
            ((_par.maxTreeDepth > 0) && (level >= _par.maxTreeDepth)));
    }
    typename NodeType::Split* makeSplit(size_t iFeature, algorithmFPType featureValue, bool bUnordered);
    typename NodeType::Leaf* makeLeaf(const IndexType* idx, size_t n, const ImpurityType& imp)
    {
        typename NodeType::Leaf* pNode = nullptr;
        if(isThreaded())
        {
            _mtAlloc.lock();
            pNode = _tree.allocator().allocLeaf();
            _mtAlloc.unlock();
        }
        else
            pNode = _tree.allocator().allocLeaf();
        pNode->response = _initialF + _algo.computeLeafWeightUpdateF(idx, n, f(), _aSampleToF.get(), imp, _par);
        return pNode;
    }
    bool findBestSplit(SplitJobType& job, SplitDataType& split, IndexType& iFeature);
    int findBestSplitImpl(SplitJobType& job, SplitDataType& split,
        IndexType& iFeature, int& idxFeatureValueBestSplit);
    void findSplitOneFeature(const IndexType* featureSample, size_t iFeatureInSample, SplitJobType& job, BestSplit& bestSplit);

    bool simpleSplit(SplitJobType& job, SplitDataType& split, IndexType& iFeature);
    void finalizeBestSplitFeatIndexed(const IndexType* aIdx, size_t n,
        SplitDataType& bestSplit, IndexType iFeature, size_t idxFeatureValueBestSplit, IndexType* bestSplitIdx) const;

    bool findBestSplitFeatSorted(const algorithmFPType* featureVal, const IndexType* aIdx,
        size_t n, const ImpurityType& curImpurity, SplitDataType& split) const
    {
        return split.featureUnordered ? _algo.findBestSplitFeatSortedCategorical(_dataHelper, featureVal, aIdx, n, _accuracy, curImpurity, split) :
            _algo.findBestSplitFeatSortedOrdered(_dataHelper, featureVal, aIdx, n, _accuracy, curImpurity, split);
    }

    void featureValuesToBuf(size_t iFeature, algorithmFPType* featureVal, IndexType* aIdx, size_t n)
    {
        _dataHelper.getColumnValues(iFeature, aIdx, n, featureVal);
        daal::algorithms::internal::qSort<algorithmFPType, int, cpu>(n, featureVal, aIdx);
    }

    //find features to check in the current split node
    const IndexType* chooseFeatures()
    {
        const size_t n = nFeatures();
        if(n == _nFeaturesPerNode)
            return nullptr;
        IndexType* featureSample = _memHelper->getFeatureSampleBuf();
        AUTOLOCK(_mtEngine);
        RNGs<IndexType, cpu>().uniformWithoutReplacement(_nFeaturesPerNode, featureSample, featureSample + _nFeaturesPerNode,
            _engine.getState(), 0, n);
        return featureSample;
    }
    void updateOOB(size_t iTree);

public:
    daal::services::AtomicInt _nParallelNodes;

protected:
    engines::internal::BatchBaseImpl& _engine;
    daal::Mutex _mtEngine;

    const NumericTable *_data;
    const NumericTable *_resp;
    const Parameter& _par;
    const size_t _nSamples;
    const size_t _nFeaturesPerNode;
    mutable TVector<IndexType, cpu, DefaultAllocator<cpu>> _aSample;
    mutable IndexTypeArray _aBestSplitIdxBuf;
    TVector<algorithmFPType, cpu, DefaultAllocator<cpu>> _aF; //loss function arguments (f)

    //bagging, first _nSamples indices are the mapping of sample to row indices, the rest is OOB indices
    TVector<IndexType, cpu, DefaultAllocator<cpu>> _aSampleToF;

    using MemHelperType = MemHelperBase<algorithmFPType, cpu>;
    MemHelperType* _memHelper;
    DataHelper _dataHelper;
    TreeType _tree;
    const FeatureTypeHelper<cpu>& _featHelper;
    algorithmFPType _accuracy;
    algorithmFPType _initialF;
    size_t _nClasses;
    AlgoType _algo;
    LossFunctionType* _loss;

    bool _bThreaded;
    bool _bParallelFeatures;
    bool _bParallelNodes;

    daal::task_group* _taskGroup;
    daal::Mutex _mtAlloc;
    const int _nThreadsMax;
};

template <typename algorithmFPType, typename DataHelper, typename AlgoType, CpuType cpu>
bool TrainBatchTaskBase<algorithmFPType, DataHelper, AlgoType, cpu>::initMemHelper()
{
    const auto nFeaturesSample = (nFeatures() == _nFeaturesPerNode ? 0 : //do not allocate
        2 * _nFeaturesPerNode); //_nFeaturesPerNode elements are used by algorithm, the rest are used internally by rng generator

    if(isThreaded())
        _memHelper = new MemHelperThr<algorithmFPType, cpu>(nFeaturesSample);
    else
        _memHelper = new MemHelperSeq<algorithmFPType, cpu>(nFeaturesSample,
            _par.memorySavingMode ? 0 : _dataHelper.sortedFeatures().getMaxNumberOfDiffValues(),
            _nSamples); //TODO
    return _memHelper && _memHelper->init();
}

template <typename algorithmFPType, typename DataHelper, typename AlgoType, CpuType cpu>
services::Status TrainBatchTaskBase<algorithmFPType, DataHelper, AlgoType, cpu>::run(dtrees::internal::DecisionTreeTable** aTbl,
    size_t nTrees, size_t iIteration) //TODO
{
    for(size_t i = 0; i < nTrees; ++i)
        aTbl[i] = nullptr;

    const size_t nRows = _data->getNumberOfRows();
    if(_nSamples < nRows)
    {
        auto aSampleToF = _aSampleToF.get();
        for(size_t i = 0; i < nRows; ++i)
            aSampleToF[i] = i;
        //no need to lock mutex here
        dtrees::training::internal::shuffle<cpu>(_engine.getState(), nRows, aSampleToF);
        auto aSample = _aSample.get();
        daal::algorithms::internal::qSort<IndexType, cpu>(_nSamples, aSampleToF);
        daal::algorithms::internal::qSort<IndexType, cpu>(nRows - _nSamples, aSampleToF + _nSamples);
        for(size_t i = 0; i < _nSamples; ++i)
            aSample[i] = aSampleToF[i];
    }
    else
    {
        auto aSample = _aSample.get();
        for(size_t i = 0; i < _nSamples; ++i)
            aSample[i] = i;
    }
    if(!iIteration || _nSamples < nRows)
    {
        //init responses buffer, keep _aSample values in it
        //if no bootstrap then it is done once, at the first iteration only
        DAAL_CHECK_MALLOC(_dataHelper.init(_data, _resp, _aSample.get()));
    }
    {
        //use _aSample as an array of response indices stored by helper from now on
        auto aSample = _aSample.get();
        TVector<algorithmFPType, cpu, DefaultAllocator<cpu>> y(_nSamples);
        auto py = y.get();
        DAAL_CHECK_MALLOC(py);
        const auto response = _dataHelper.responses();
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS;
        for(size_t i = 0; i < _nSamples; ++i)
        {
            py[i] = response[i].val;
            aSample[i] = i;
        }
        if(iIteration)
        {
            _initialF = 0;
        }
        else
        {
            if(!getInitialF(py, _nSamples, _initialF))
                _initialF = algorithmFPType(0);
            initializeF(_initialF);
        }
        _algo.step(_nSamples, py, f(), _aSampleToF.get(), *lossFunc());
    }
    //boost trees
    for(size_t i = 0; i < nTrees; ++i)
    {
        _tree.destroy();
        _nParallelNodes.set(0);
        typename NodeType::Base* nd = buildRoot(i);
        DAAL_CHECK_MALLOC(nd);
        _tree.reset(nd, false); //bUnorderedFeaturesUsed - TODO?
        aTbl[i] = gbt::internal::ModelImpl::treeToTable(_tree);
        if(_aSampleToF.get() && _tree.top()) //bagging
            updateOOB(i);
    }
    return Status();
}

template <typename algorithmFPType, typename DataHelper, typename AlgoType, CpuType cpu>
void TrainBatchTaskBase<algorithmFPType, DataHelper, AlgoType, cpu>::updateOOB(size_t iTree)
{
    const auto aSampleToF = _aSampleToF.get();
    auto pf = f();
    const size_t n = _aSampleToF.size();
    for(size_t i = _nSamples; i < n; ++i) //todo: parallel_for
    {
        IndexType iRow = aSampleToF[i];
        ReadRows<algorithmFPType, cpu> x(const_cast<NumericTable*>(_dataHelper.data()), iRow, 1);
        auto pNode = dtrees::prediction::internal::findNode<algorithmFPType, TreeType, cpu>(_tree, x.get());
        DAAL_ASSERT(pNode);
        algorithmFPType inc = TreeType::NodeType::castLeaf(pNode)->response;
        pf[iRow*_nClasses + iTree] += inc;
    }
}

template <typename algorithmFPType, typename DataHelper, typename AlgoType, CpuType cpu>
void TrainBatchTaskBase<algorithmFPType, DataHelper, AlgoType, cpu>::buildNode(
    size_t iStart, size_t n, size_t level, const ImpurityType& imp, NodeType::Base*&res)
{
    if(_taskGroup)
    {
        SplitTask job(*this, iStart, n, level, imp, res);
        _taskGroup->run(job);
    }
    else
    {
        SplitJobType job(iStart, n, level, imp, res);
        buildSplit(job);
    }
}

template <typename algorithmFPType, typename DataHelper, typename AlgoType, CpuType cpu>
void TrainBatchTaskBase<algorithmFPType, DataHelper, AlgoType, cpu>::buildSplit(SplitJobType& job)
{
    SplitDataType split;
    IndexType iFeature;
    if(findBestSplit(job, split, iFeature))
    {
        typename NodeType::Split* res = makeSplit(iFeature, split.featureValue, split.featureUnordered);
        if(res)
        {
            job.res = res;
            res->kid[0] = buildLeaf(job.iStart, split.nLeft, job.level + 1, split.left);
            res->kid[1] = buildLeaf(job.iStart + split.nLeft, job.n - split.nLeft, job.level + 1, job.imp - split.left);
            if(res->kid[0])
            {
                if(res->kid[1])
                    return; //all done
                SplitJobType right(job.iStart + split.nLeft, job.n - split.nLeft, job.level + 1, job.imp - split.left, res->kid[1]);
                buildSplit(right); //by this thread, no new job
            }
            else if(res->kid[1])
            {
                SplitJobType left(job.iStart, split.nLeft, job.level + 1, split.left, res->kid[0]);
                buildSplit(left); //by this thread, no new job
            }
            else
            {
                //one kid can be a new job, the left one, if there are available threads
                if(numAvailableThreads())
                    buildNode(job.iStart, split.nLeft, job.level + 1, split.left, res->kid[0]);
                else
                {
                    SplitJobType left(job.iStart, split.nLeft, job.level + 1, split.left, res->kid[0]);
                    buildSplit(left); //by this thread, no new job
                }
                //and another kid is processed in the same thread
                SplitJobType right(job.iStart + split.nLeft, job.n - split.nLeft, job.level + 1, job.imp - split.left, res->kid[1]);
                buildSplit(right); //by this thread, no new job
            }
            return;
        }
    }
    job.res = makeLeaf(_aSample.get() + job.iStart, job.n, job.imp);
}

template <typename algorithmFPType, typename DataHelper, typename AlgoType, CpuType cpu>
typename TrainBatchTaskBase<algorithmFPType, DataHelper, AlgoType, cpu>::NodeType::Split*
    TrainBatchTaskBase<algorithmFPType, DataHelper, AlgoType, cpu>::makeSplit(size_t iFeature,
    algorithmFPType featureValue, bool bUnordered)
{
    typename NodeType::Split* pNode = nullptr;
    if(isThreaded())
    {
        _mtAlloc.lock();
        pNode = _tree.allocator().allocSplit();
        _mtAlloc.unlock();
    }
    else
        pNode = _tree.allocator().allocSplit();
    pNode->set(iFeature, featureValue, bUnordered);
    return pNode;
}

template <typename algorithmFPType, typename DataHelper, typename AlgoType, CpuType cpu>
bool TrainBatchTaskBase<algorithmFPType, DataHelper, AlgoType, cpu>::findBestSplit(SplitJobType& job,
    SplitDataType& bestSplit, IndexType& iFeature)
{
    if(job.n == 2)
    {
        DAAL_ASSERT(_par.minObservationsInLeafNode == 1);
        return simpleSplit(job, bestSplit, iFeature);
    }

    int idxFeatureValueBestSplit = -1; //when sorted feature is used
    const int iBestSplit = findBestSplitImpl(job, bestSplit, iFeature, idxFeatureValueBestSplit);
    if(iBestSplit < 0)
        return false;
    IndexType* bestSplitIdx = bestSplitIdxBuf() + job.iStart;
    bool bCopyToIdx = true;
    IndexType* aIdx = _aSample.get() + job.iStart;
    if(idxFeatureValueBestSplit >= 0)
    {
        //indexed feature was used
        //calculate impurity (??) and get split to bestSplitIdx
        finalizeBestSplitFeatIndexed(aIdx, job.n, bestSplit, iFeature, idxFeatureValueBestSplit, bestSplitIdx);
    }
    else if(bestSplit.featureUnordered)
    {
        if(bestSplit.iStart)
        {
            DAAL_ASSERT(bestSplit.iStart + bestSplit.nLeft <= job.n);
            tmemcpy<IndexType, cpu>(aIdx, bestSplitIdx + bestSplit.iStart, bestSplit.nLeft);
            aIdx += bestSplit.nLeft;
            tmemcpy<IndexType, cpu>(aIdx, bestSplitIdx, bestSplit.iStart);
            aIdx += bestSplit.iStart;
            bestSplitIdx += bestSplit.iStart + bestSplit.nLeft;
            if(job.n > (bestSplit.iStart + bestSplit.nLeft))
                tmemcpy<IndexType, cpu>(aIdx, bestSplitIdx, job.n - bestSplit.iStart - bestSplit.nLeft);
            bCopyToIdx = false;//done
        }
    }
    else
        bCopyToIdx = (isParallelFeatures() || (iBestSplit + 1 < _nFeaturesPerNode));
        //in sequential mode, if iBestSplit is the last considered feature then aIdx already contains the best split, no need to copy

    if(bCopyToIdx)
        tmemcpy<IndexType, cpu>(aIdx, bestSplitIdx, job.n);
    return true;
}

template <typename algorithmFPType, typename DataHelper, typename AlgoType, CpuType cpu>
bool TrainBatchTaskBase<algorithmFPType, DataHelper, AlgoType, cpu>::simpleSplit(SplitJobType& job,
    SplitDataType& split, IndexType& iFeature)
{
    algorithmFPType featBuf[2];
    IndexType* aIdx = _aSample.get() + job.iStart;
    for(size_t i = 0; i < _nFeaturesPerNode; ++i)
    {
        {
            AUTOLOCK(_mtEngine);
            RNGs<IndexType, cpu>().uniform(1, &iFeature, _engine.getState(), 0, _data->getNumberOfColumns());
        }
        featureValuesToBuf(iFeature, featBuf, aIdx, 2);
        if(featBuf[1] - featBuf[0] <= _accuracy) //all values of the feature are the same
            continue;
        split.featureValue = featBuf[0];
        split.nLeft = 1;
        split.iStart = 0;
        _algo.simpleSplit(featBuf, aIdx, split.left);
        split.impurityDecrease = job.imp.value(_par.lambda);//TODO
        return true;
    }
    return false;
}

template <typename algorithmFPType, typename DataHelper, typename AlgoType, CpuType cpu>
void TrainBatchTaskBase<algorithmFPType, DataHelper, AlgoType, cpu>::finalizeBestSplitFeatIndexed(const IndexType* aIdx, size_t n,
    SplitDataType& split, IndexType iFeature, size_t idxFeatureValueBestSplit, IndexType* bestSplitIdx) const
{
    DAAL_ASSERT(split.nLeft > 0);
    IndexType* bestSplitIdxRight = bestSplitIdx + split.nLeft;
    const int iRowSplitVal = doPartition<typename DataHelper::super::Response, IndexType, typename SortedFeaturesHelper::IndexType, size_t, cpu>(
        n, aIdx, _dataHelper.responses(),
        _dataHelper.sortedFeatures().data(iFeature), split.featureUnordered,
        idxFeatureValueBestSplit,
        bestSplitIdxRight, bestSplitIdx,
        split.nLeft);

    DAAL_ASSERT(iRowSplitVal >= 0);
    split.iStart = 0;
    split.featureValue = _dataHelper.getValue(iFeature, iRowSplitVal);
}

template <typename algorithmFPType, typename DataHelper, typename AlgoType, CpuType cpu>
void TrainBatchTaskBase<algorithmFPType, DataHelper, AlgoType, cpu>::findSplitOneFeature(
    const IndexType* featureSample, size_t iFeatureInSample, SplitJobType& job, BestSplit& bestSplit)
{
    const float qMax = 0.02; //min fracture of observations to be handled as indexed feature values
    const IndexType iFeature = featureSample ? featureSample[iFeatureInSample] : (IndexType)iFeatureInSample;
    const bool bUseSortedFeatures = (!_par.memorySavingMode) &&
        (float(job.n) > qMax*float(_dataHelper.sortedFeatures().getMaxNumberOfDiffValues(iFeature)));
    IndexType* aIdx = _aSample.get() + job.iStart;

    if(bUseSortedFeatures)
    {
        if(!_dataHelper.hasDiffFeatureValues(iFeature, aIdx, job.n))
            return;//all values of the feature are the same
        //use best split estimation when searching on iFeature
        SplitDataType split(bestSplit.impurityDecrease(), _featHelper.isUnordered(iFeature));
        //index of best feature value in the array of sorted feature values
        const int idxFeatureValue = _algo.findBestSplitFeatIndexed(_dataHelper, *_memHelper, iFeature, aIdx, job.n, job.imp, split);
        if(idxFeatureValue < 0)
            return;
        bestSplit.update(split, idxFeatureValue, iFeatureInSample);
    }
    else
    {
        const bool bThreaded = bestSplit.isThreadedMode();
        IndexType* bestSplitIdx = bestSplitIdxBuf() + job.iStart;
        auto aFeatBuf = _memHelper->getFeatureValueBuf(job.n); //TODO?
        typename MemHelperType::IndexTypeVector* aFeatIdxBuf = nullptr;
        if(bThreaded)
        {
            //get a local index, since it is used by parallel threads
            aFeatIdxBuf = _memHelper->getSortedFeatureIdxBuf(job.n);
            tmemcpy<IndexType, cpu>(aFeatIdxBuf->get(), aIdx, job.n);
            aIdx = aFeatIdxBuf->get();
        }
        algorithmFPType* featBuf = aFeatBuf->get();
        featureValuesToBuf(iFeature, featBuf, aIdx, job.n);
        if(featBuf[job.n - 1] - featBuf[0] <= _accuracy) //all values of the feature are the same
        {
            _memHelper->releaseFeatureValueBuf(aFeatBuf);
            if(aFeatIdxBuf)
                _memHelper->releaseSortedFeatureIdxBuf(aFeatIdxBuf);
            return;
        }
        //use best split estimation when searching on iFeature
        SplitDataType split(bestSplit.impurityDecrease(), _featHelper.isUnordered(iFeature));
        bool bFound = findBestSplitFeatSorted(featBuf, aIdx, job.n, job.imp, split);
        _memHelper->releaseFeatureValueBuf(aFeatBuf);
        if(bFound)
        {
            DAAL_ASSERT(split.iStart < job.n);
            DAAL_ASSERT(split.iStart + split.nLeft <= job.n);
            if(split.featureUnordered || bThreaded || (iFeatureInSample + 1 < _nFeaturesPerNode))
                bestSplit.update(split, iFeatureInSample, bestSplitIdx, aIdx, job.n);
            else
                bestSplit.update(split, -1, iFeatureInSample);
        }
        if(aFeatIdxBuf)
            _memHelper->releaseSortedFeatureIdxBuf(aFeatIdxBuf);
    }
}

template <typename algorithmFPType, typename DataHelper, typename AlgoType, CpuType cpu>
int TrainBatchTaskBase<algorithmFPType, DataHelper, AlgoType, cpu>::findBestSplitImpl(SplitJobType& job,
    SplitDataType& split, IndexType& iFeature, int& idxFeatureValueBestSplit)
{
    const IndexType* featureSample = chooseFeatures();
    int iFeatureInSample = -1;
    if(isParallelFeatures())//TODO: use numAvailableThreads()
    {
        daal::Mutex mtBestSplit;
        BestSplit bestSplit(split, &mtBestSplit);
        daal::threader_for(_nFeaturesPerNode, _nFeaturesPerNode, [&](size_t i)
        {
            findSplitOneFeature(featureSample, i, job, bestSplit);
        });
        idxFeatureValueBestSplit = bestSplit.iIndexedFeatureSplitValue();
        iFeatureInSample = bestSplit.iFeatureInSample();
    }
    else
    {
        BestSplit bestSplit(split, nullptr);
        for(size_t i = 0; i < _nFeaturesPerNode; ++i)
        {
            findSplitOneFeature(featureSample, i, job, bestSplit);
        }
        idxFeatureValueBestSplit = bestSplit.iIndexedFeatureSplitValue();
        iFeatureInSample = bestSplit.iFeatureInSample();
    }
    if(iFeatureInSample >= 0)
        iFeature = featureSample ? featureSample[iFeatureInSample] : iFeatureInSample;
    if(featureSample)
        _memHelper->releaseFeatureSampleBuf(const_cast<IndexType*>(featureSample));

    if(iFeatureInSample < 0)
        return -1; //not found
    //now calculate full impurity decrease
    split.impurityDecrease -= job.imp.value(_par.lambda);
    if(split.impurityDecrease < _par.minSplitLoss)
        return -1; //not found
    return iFeatureInSample;
}

template<typename algorithmFPType, CpuType cpu>
class AlgoXBoost : public AlgoBase<algorithmFPType, cpu>
{
public:
    typedef AlgoBase<algorithmFPType, cpu> super;
    typedef gh<algorithmFPType, cpu> ghType;
    using typename super::ImpurityType;
    typedef SplitData<algorithmFPType, ImpurityType> SplitDataType;
    typedef MemHelperBase<algorithmFPType, cpu> MemHelperType;
    using typename super::DataHelperType;

    AlgoXBoost(size_t numTrees, size_t nMinSplitPart, double lambda) : super(numTrees), _dataSize(0), _nMinSplitPart(nMinSplitPart), _lambda(lambda){}
    virtual void getInitialImpurity(ImpurityType& val, size_t nSamples) DAAL_C11_OVERRIDE
    {
        ghType* pgh = grad(this->_curTree);
        auto& G = val.g;
        auto& H = val.h;
        G = H = 0;
        PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < nSamples; ++i)
        {
            G += pgh[i].g;
            H += pgh[i].h;
        }
    }
    bool reset(size_t n) { _dataSize = n; n *= this->_nTrees; _aGH.reset(n); return _aGH.get(); }

    //loss function gradient and hessian values calculated in f() points
    ghType* grad(size_t iTree) { return _aGH.get() + iTree*dataSize(); }
    const ghType* grad(size_t iTree) const { return _aGH.get() + iTree*dataSize(); }
    //data size, number of observations
    size_t dataSize() const { return _dataSize; }

    //called once at the very beginning
    virtual Status init(const DataHelperType& inputData) DAAL_C11_OVERRIDE
    {
        DAAL_CHECK_MALLOC(reset(inputData.size()));
        return Status();
    }

    virtual void step(size_t n, const algorithmFPType* y, const algorithmFPType* f,
        const IndexType* aSample, LossFunction<algorithmFPType, cpu>& func) DAAL_C11_OVERRIDE
    {
        func.getGradients(n, y, f, aSample, (algorithmFPType*)_aGH.get());
    }

    virtual double computeLeafWeightUpdateF(const IndexType* idx, size_t n, algorithmFPType* f,
        const IndexType* sampleInd, const ImpurityType& imp, const Parameter& par) DAAL_C11_OVERRIDE
    {
        algorithmFPType val = imp.h + par.lambda;
        if(isZero<algorithmFPType, cpu>(val))
            return 0;

        val = -imp.g / val;
        const algorithmFPType inc = val*par.shrinkage;
        if(sampleInd)
        {
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for(size_t i = 0; i < n; ++i)
                    f[sampleInd[idx[i]] * this->_nTrees + this->_curTree] += inc;
        }
        else
        {
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for(size_t i = 0; i < n; ++i)
                f[idx[i] * this->_nTrees + this->_curTree] += inc;
        }
        return inc;
    }

    void simpleSplit(const algorithmFPType* featureVal, const IndexType* aIdx, ImpurityType& imp)
    {
        const auto gh = grad(this->_curTree);
        imp.reset(gh[*aIdx].g, gh[*aIdx].h);
    }

    int findBestSplitFeatIndexed(const DataHelperType& inputData, MemHelperType& memHelper,
        IndexType iFeature, const IndexType* aIdx, size_t n,
        const ImpurityType& curImpurity, SplitDataType& split) const;

    bool findBestSplitFeatSortedOrdered(const DataHelperType& inputData, const algorithmFPType* featureVal, const IndexType* aIdx,
        size_t n, algorithmFPType accuracy, const ImpurityType& curImpurity, SplitDataType& split) const;

    bool findBestSplitFeatSortedCategorical(const DataHelperType& inputData, const algorithmFPType* featureVal, const IndexType* aIdx,
        size_t n, algorithmFPType accuracy, const ImpurityType& curImpurity, SplitDataType& split) const;

private:
    void calcImpurity(const IndexType* aIdx, size_t n, ImpurityType& imp) const //todo: tree?
    {
        DAAL_ASSERT(n);
        const ghType* pgh = grad(this->_curTree);
        imp = pgh[aIdx[0]];
        PRAGMA_VECTOR_ALWAYS
        for(size_t i = 1; i < n; ++i)
        {
            imp.g += pgh[aIdx[i]].g;
            imp.h += pgh[aIdx[i]].h;
        }
    }

private:
    ImpurityType _initialImpurity;
    size_t _dataSize;
    size_t _nMinSplitPart;
    double _lambda;
    TVector<ghType, cpu, DefaultAllocator<cpu>> _aGH; //loss function first and second order derivatives
};

template<typename algorithmFPType, CpuType cpu>
int AlgoXBoost<algorithmFPType, cpu>::findBestSplitFeatIndexed(const DataHelperType& inputData,
    MemHelperType& memHelper,
    IndexType iFeature, const IndexType* aIdx, size_t n,
    const ImpurityType& curImpurity, SplitDataType& split) const
{
    const size_t nDiffFeatMax = inputData.sortedFeatures().getMaxNumberOfDiffValues(iFeature);
    auto nFeatIdxBuf = memHelper.getIndexedFeatureCountsBuf(nDiffFeatMax); //counts of indexed feature values
    DAAL_ASSERT(nFeatIdxBuf); //TODO: return status
    if(!nFeatIdxBuf)
        return -1;
    auto nFeatIdx = nFeatIdxBuf->get();
    DAAL_ASSERT(nFeatIdx);

    auto aGHBuf = memHelper.getGradBuf(nDiffFeatMax);
    DAAL_ASSERT(aGHBuf); //TODO: return status
    if(!aGHBuf)
    {
        memHelper.releaseIndexedFeatureCountsBuf(nFeatIdxBuf);
        return -1;
    }
    auto pBuf = aGHBuf->get();
    DAAL_ASSERT(pBuf);

    for(size_t i = 0; i < nDiffFeatMax; ++i)
        nFeatIdx[i] = 0;

    //make a copy since it can be corrected below. TODO: propagate this corrected value to the argument?
    ImpurityType imp(curImpurity);
    //the buffer keeps sums of g and h for each of unique feature values
    for(size_t i = 0; i < nDiffFeatMax; ++i)
        pBuf[i].g = pBuf[i].h = algorithmFPType(0);

    //below we calculate only part of the impurity decrease dependent on split itself
    algorithmFPType bestImpDecrease = split.impurityDecrease;
    algorithmFPType gTotal = 0; //total sum of g in the set being split
    algorithmFPType hTotal = 0; //total sum of h in the set being split
    {
        const SortedFeaturesHelper::IndexType* sortedFeaturesIdx = inputData.sortedFeatures().data(iFeature);
        const auto aResponse = inputData.responses();
        const ghType* pgh = grad(this->_curTree);
        PRAGMA_VECTOR_ALWAYS
        for(size_t i = 0; i < n; ++i)
        {
            const IndexType iSample = aIdx[i];
            const typename DataHelperType::super::Response& r = aResponse[aIdx[i]];
            const typename SortedFeaturesHelper::IndexType idx = sortedFeaturesIdx[r.idx];
            ++nFeatIdx[idx];
            pBuf[idx].g += pgh[iSample].g;
            pBuf[idx].h += pgh[iSample].h;
            gTotal += pgh[iSample].g;
            hTotal += pgh[iSample].h;
        }
    }
    if(!isZero<algorithmFPType, cpu>(gTotal - imp.g))
        imp.g = gTotal;
    if(!isZero<algorithmFPType, cpu>(hTotal - imp.h))
        imp.h = hTotal;
    size_t nLeft = 0;
    ImpurityType left;
    int idxFeatureBestSplit = -1; //index of best feature value in the array of sorted feature values
    for(size_t i = 0; i < nDiffFeatMax; ++i)
    {
        if(!nFeatIdx[i])
            continue;
        nLeft = (split.featureUnordered ? nFeatIdx[i] : nLeft + nFeatIdx[i]);
        if((nLeft == n) //last split
            || ((n - nLeft) < _nMinSplitPart))
            break;
        if(split.featureUnordered)
        {
            if(nLeft < _nMinSplitPart)
                continue;
            left.reset(pBuf[i].g, pBuf[i].h);
        }
        else
        {
            left.add(pBuf[i]);
            if(nLeft < _nMinSplitPart)
                continue;
        }
        ImpurityType right(imp, left);
        //the part of the impurity decrease dependent on split itself
        const algorithmFPType impDecrease = left.value(_lambda) + right.value(_lambda);
        if(impDecrease > bestImpDecrease)
        {
            split.left = left;
            split.nLeft = nLeft;
            idxFeatureBestSplit = i;
            bestImpDecrease = impDecrease;
        }
    }
    if(idxFeatureBestSplit >= 0)
        split.impurityDecrease = bestImpDecrease;
    memHelper.releaseIndexedFeatureCountsBuf(nFeatIdxBuf);
    memHelper.releaseGradBuf(aGHBuf);
    return idxFeatureBestSplit;
}

template <typename algorithmFPType, CpuType cpu>
bool AlgoXBoost<algorithmFPType, cpu>::findBestSplitFeatSortedOrdered(const DataHelperType& inputData, const algorithmFPType* featureVal,
    const IndexType* aIdx, size_t n, algorithmFPType accuracy, const ImpurityType& curImpurity, SplitDataType& split) const
{
    ImpurityType left(grad(this->_curTree)[*aIdx]);
    algorithmFPType bestImpurityDecrease = split.impurityDecrease;
    IndexType iBest = -1;
    const algorithmFPType last = featureVal[n - _nMinSplitPart];
    for(size_t i = 1; i < (n - _nMinSplitPart + 1); ++i)
    {
        const bool bSameFeaturePrev(featureVal[i] <= featureVal[i - 1] + accuracy);
        if(!(bSameFeaturePrev || i < _nMinSplitPart))
        {
            //can make a split
            //nLeft == i, nRight == n - i
            ImpurityType right(curImpurity, left);
            const algorithmFPType v = left.value(_lambda) + right.value(_lambda);
            if(v > bestImpurityDecrease)
            {
                bestImpurityDecrease = v;
                split.left = left;
                iBest = i;
            }
        }

        //update impurity and continue
        left.add(grad(this->_curTree)[aIdx[i]]);
    }
    if(iBest < 0)
        return false;

    split.impurityDecrease = bestImpurityDecrease;
    split.nLeft = iBest;
    split.iStart = 0;
    split.featureValue = featureVal[iBest - 1];
    return true;
}

template <typename algorithmFPType, CpuType cpu>
bool AlgoXBoost<algorithmFPType, cpu>::findBestSplitFeatSortedCategorical(const DataHelperType& inputData, const algorithmFPType* featureVal,
    const IndexType* aIdx, size_t n, algorithmFPType accuracy, const ImpurityType& curImpurity, SplitDataType& split) const
{
    DAAL_ASSERT(n >= 2 * _nMinSplitPart);
    algorithmFPType bestImpurityDecrease = split.impurityDecrease;
    ImpurityType left;
    bool bFound = false;
    size_t nDiffFeatureValues = 0;
    for(size_t i = 0; i < n - _nMinSplitPart;)
    {
        ++nDiffFeatureValues;
        size_t count = 1;
        const algorithmFPType first = featureVal[i];
        const size_t iStart = i;
        for(++i; (i < n) && (featureVal[i] == first); ++count, ++i);
        if((count < _nMinSplitPart) || ((n - count) < _nMinSplitPart))
            continue;

        if((i == n) && (nDiffFeatureValues == 2) && bFound)
            break; //only 2 feature values, one possible split, already found

        calcImpurity(aIdx + iStart, count, left);
        ImpurityType right(curImpurity, left);
        const algorithmFPType v = left.value(_lambda) + right.value(_lambda);
        if(v > bestImpurityDecrease)
        {
            bestImpurityDecrease = v;
            split.left = left;
            split.nLeft = count;
            split.iStart = iStart;
            split.featureValue = first;
            bFound = true;
        }
    }
    if(bFound)
        split.impurityDecrease = bestImpurityDecrease;
    return bFound;
}

//////////////////////////////////////////////////////////////////////////////////////////
// compute() implementation
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu, typename TaskType>
services::Status computeImpl(const NumericTable *x, const NumericTable *y, gbt::internal::ModelImpl& md,
    const gbt::training::Parameter& par, engines::internal::BatchBaseImpl& engine, size_t nClasses)
{
    const size_t nTrees = nClasses > 2 ? nClasses : 1;
    DAAL_CHECK_MALLOC(md.reserve(par.maxIterations*nClasses));
    dtrees::internal::FeatureTypeHelper<cpu> featHelper;
    DAAL_CHECK_MALLOC(featHelper.init(x));

    dtrees::internal::SortedFeaturesHelper sortedFeatHelper;
    services::Status s;
    if(!par.memorySavingMode)
        DAAL_CHECK_STATUS(s, (sortedFeatHelper.init<algorithmFPType, cpu>(*x)));
    TaskType task(x, y, par, featHelper, par.memorySavingMode ? nullptr : &sortedFeatHelper, engine, nClasses);
    DAAL_CHECK_STATUS(s, task.init());
    TVector<dtrees::internal::DecisionTreeTable*, cpu, DefaultAllocator<cpu> > aTables;
    typename dtrees::internal::DecisionTreeTable* pTbl = nullptr;
    dtrees::internal::DecisionTreeTable** aTbl = &pTbl;
    if(nTrees > 1)
    {
        aTables.reset(nTrees);
        DAAL_CHECK_MALLOC(aTables.get());
        aTbl = aTables.get();
    }
    for(size_t i = 0; i < par.maxIterations; ++i)
    {
        s = task.run(aTbl, nTrees, i);
        if(!s)
        {
            deleteTables<cpu>(aTbl, nTrees);
            break;
        }
        size_t iTree = 0;
        for(; (iTree < nTrees) && (aTbl[iTree]->getNumberOfRows() < 2); ++iTree);
        if(iTree == nTrees) //all are one level (constant response) trees
        {
            deleteTables<cpu>(aTbl, nTrees);
            break;
        }
        for(iTree = 0; iTree < nTrees; ++iTree)
            md.add(aTbl[iTree]);
        if((i + 1 < par.maxIterations) && task.done())
            break;
    }
    return s;
}

} /* namespace internal */
} /* namespace training */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif
