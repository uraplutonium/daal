/* file: algorithm_base_batch.h */
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
//  Implementation of base classes defining algorithm interface for batch processing mode.
//--
*/

#ifndef __ALGORITHM_BASE_BATCH_H__
#define __ALGORITHM_BASE_BATCH_H__

#include "services/daal_memory.h"

namespace daal
{
namespace algorithms
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMCONTAINERIFACE"></a>
 * \brief Abstract interface class that provides virtual methods to access and run implementations
 *        of the algorithms in %batch mode. It is associated with the Algorithm<batch> class
 *        and supports the methods for computation of the algorithm results.
 *        The methods of the container are defined in derivative containers defined for each algorithm.
 */
template<> class AlgorithmContainerIface<batch>
{
public:
    /** Default constructor */
    AlgorithmContainerIface(daal::services::Environment::env *daalEnv = 0): _par(0), _in(0), _res(0), _env(daalEnv),
        _errors(new services::ErrorCollection()), _kernel(NULL) {};

    virtual ~AlgorithmContainerIface() {}

    /**
     * Sets the information about the environment
     * \param[in] daalEnv   Pointer to the structure that contains information about the environment
     */
    void setEnvironment(daal::services::Environment::env *daalEnv)
    {
        _env = daalEnv;
    }

    /**
     * Sets arguments of the algorithm
     * \param[in] in    Pointer to the input arguments of the algorithm
     * \param[in] res   Pointer to the final results of the algorithm
     * \param[in] par   Pointer to the parameters of the algorithm
     */
    void setArguments(Input *in, Result *res, Parameter *par)
    {
        _in  = in;
        _par = par;
        _res = res;
    }

    /**
     * Sets the collection of errors
     * \param[in] errors    Pointer to the collection of errors
     */
    void setErrorCollection(services::SharedPtr<services::ErrorCollection> errors)
    {
        _errors = errors;
        setKernelErrorCollection();
    }

    /**
     * Sets the collection of errors to kernels
     */
    void setKernelErrorCollection()
    {
        services::SharedPtr<services::KernelErrorCollection> e = _errors->getErrors();
        if(_kernel)
        {
            _kernel->setErrorCollection(e);
        }
    }

    /**
     * Retrieves final results of the algorithm
     * \return   Pointer to the final results of the algorithm
     */
    Result *getResult()
    {
        return _res;
    }

    /**
     * Computes final results of the algorithm.
     * This method behaves similarly to compute method of the Algorithm<batch> class.
     */
    virtual void compute() = 0;

    /**
     *  Allocates memory of the given size for container object
     *  \param[in] sz number of bytes to be allocated
     *  \return pointer to the allocated memory
     */
    static void *operator new(std::size_t sz)
    {
        return services::daal_malloc(sz);
    }

    /**
     *  Allocates memory of the given size for array of container objects
     *  \param[in] sz number of bytes to be allocated
     *  \return pointer to the allocated memory
     */
    static void *operator new[](std::size_t sz)
    {
        return services::daal_malloc(sz);
    }

    /**
     *  Frees memory allocated for container object
     *  \param[in] ptr pointer to the allocated memory
     *  \param[in] sz  number of bytes to be freed
     */
    static void operator delete(void *ptr, std::size_t sz)
    {
        services::daal_free(ptr);
    }

    /**
     *  Frees memory allocated for array of container objects
     *  \param[in] ptr pointer to the allocated memory
     *  \param[in] sz  number of bytes to be freed
     */
    static void operator delete[](void *ptr, std::size_t sz)
    {
        services::daal_free(ptr);
    }

    /**
     *  Placement new for container object
     *  \param[in] sz number of bytes to be allocated
     *  \param[in] where pointer to memory
     *  \return pointer to the allocated memory
     */
    static void *operator new(std::size_t sz, void *where) { return where; }

    /**
     *  Placement new for array of container objects
     *  \param[in] sz number of bytes to be allocated
     *  \param[in] where pointer to memory
     *  \return pointer to the allocated memory
     */
    static void *operator new[](std::size_t sz, void *where) { return where; }


protected:
    Parameter                            *_par;
    Input                                *_in;
    Result                               *_res;
    daal::services::Environment::env     *_env;
    services::SharedPtr<services::ErrorCollection> _errors;

    Kernel *_kernel;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMDISPATCHCONTAINER"></a>
 * \brief Implements a container to dispatch batch processing algorithms to CPU-specific implementations.
 *
 *
 * \ref opt_notice
 *
 *
 * \tparam mode                 Computation mode of the algorithm, \ref ComputeMode
 * \tparam sse2Container        Implementation for Intel(R) Streaming SIMD Extensions 2 (Intel(R) SSE2)
 * \tparam ssse3Container       Implementation for Intel(R) Supplemental Streaming SIMD Extensions 3 (Intel(R) SSSE3)
 * \tparam sse42Container       Implementation for Intel(R) Streaming SIMD Extensions 42 (Intel(R) SSE42)
 * \tparam avxContainer         Implementation for Intel(R) Advanced Vector Extensions (Intel(R) AVX)
 * \tparam avx2Container        Implementation for Intel(R) Advanced Vector Extensions 2 (Intel(R) AVX2)
 * \tparam avx512_micContainer  Implementation for Intel(R) Xeon Phi(TM) processors/coprocessors based on Intel(R) Advanced Vector
 *                              Extensions 512 (Intel(R) AVX512)
 * \tparam avx512Container      Implementation for Intel(R) Xeon(R) processors based on Intel AVX-512
 */
template<typename sse2Container, typename ssse3Container, typename sse42Container,
         typename avxContainer, typename avx2Container,
         typename avx512_micContainer, typename avx512Container>
class DAAL_EXPORT AlgorithmDispatchContainer<batch, sse2Container, ssse3Container, sse42Container,
          avxContainer, avx2Container, avx512_micContainer, avx512Container> : public AlgorithmContainerIface<batch>
{
public:
    /** Default constructor. Constructs empty container */
    AlgorithmDispatchContainer(daal::services::Environment::env *daalEnv);
    virtual ~AlgorithmDispatchContainer() { delete _cntr; }

    virtual void compute()
    {
        _cntr->setArguments(this->_in, this->_res, this->_par);
        _cntr->setErrorCollection(this->_errors);
        _cntr->compute();
    }

protected:
    AlgorithmContainerIface<batch> *_cntr;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHM"></a>
 * \brief Implements the abstract interface AlgorithmIface. Algorithm<batch> is, in turn, the base class
 *        for the classes interfacing the major stages of data processing in %batch mode:
 *        Analysis<batch>, Training<batch> and Prediction.
 */
template<> class Algorithm<batch> : public AlgorithmIface
{
public:
    /** Default constructor */
    Algorithm(): _ac(0), _par(0), _in(0), _res(0), _errors(new services::ErrorCollection())
    {
        getEnvironment();
    };

    virtual ~Algorithm()
    {
        if(_ac)
        {
            delete _ac;
        }
    }

    virtual void clean() {}

    /**
     * Validates parameters of the compute method
     */
    virtual void checkComputeParams() = 0;

    services::SharedPtr<services::ErrorCollection> getErrors()
    {
        return _errors;
    }

protected:
    void allocateInputMemory()
    {
        allocateInput();
    }

    void allocateResultMemory()
    {
        if(_res == 0)
        {
            allocateResult();
        }
    }

    virtual void setParameter() {}

    virtual void allocateResult() = 0;
    virtual void allocateInput() {};

    virtual Algorithm<batch> *cloneImpl() const = 0;

    void getEnvironment()
    {
        int cpuid = (int)daal::services::Environment::getInstance()->getCpuId();
        if(cpuid < 0)
        {
            _errors->add(services::ErrorCpuNotSupported);
        }
        _env.cpuid = cpuid;
    }

    virtual void throwIfPossible()
    {
#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
        throw services::Exception::getException(this->_errors->getDescription());
#endif
    }

    daal::algorithms::AlgorithmContainerIface<batch> *_ac;
    daal::services::Environment::env    _env;

    Parameter *_par;
    Input     *_in;
    Result    *_res;
    services::SharedPtr<services::ErrorCollection> _errors;
};
} // namespace interface1
using interface1::AlgorithmContainerIface;
using interface1::Algorithm;

}
}

#endif
