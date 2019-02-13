/* file: logitboost_train_friedman_aux.i */
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
//  Implementation of auxiliary functions for Logit Boost
//  LBFriedman method.
//--
*/
/*
//
//  REFERENCES
//
//  1. J. Friedman, T. Hastie, R. Tibshirani.
//     Additive logistic regression: a statistical view of boosting,
//     The annals of Statistics, 2000, v28 N2, pp. 337-407
//  2. J. Friedman, T. Hastie, R. Tibshirani.
//     The Elements of Statistical Learning:
//     Data Mining, Inference, and Prediction,
//     Springer, 2001.
//
*/

#ifndef __LOGITBOOST_TRAIN_FRIEDMAN_AUX_I__
#define __LOGITBOOST_TRAIN_FRIEDMAN_AUX_I__

#include "stump_train_impl.i"

using namespace daal::algorithms::logitboost::internal;

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace training
{
namespace internal
{

/**
 *  \brief Update working responses and weights for current class
 *
 *  \param n[in]        Number of observations in training data set
 *  \param nc[in]       Number of classes in training data set
 *  \param curClass[in] Index of the current class
 *  \param label[in]    Array of class labels for each observation
 *  \param P[in]        Matrix of probabilities of size nc x n
 *  \param thrW[in]     Threshold for weight calculations
 *  \param w[out]       Array of weights of size n
 *  \param thrZ[in]     Threshold for responses calculations
 *  \param w[out]       Array of responses of size n
 */
template<typename algorithmFPType, CpuType cpu>
void initWZ(size_t n, size_t nc, size_t curClass, const int *label, const algorithmFPType *P,
            algorithmFPType thrW, algorithmFPType *w, algorithmFPType thrZ, algorithmFPType *z)
{
    const algorithmFPType one = (algorithmFPType)1.0;
    const algorithmFPType three = (algorithmFPType)3.0;
    const algorithmFPType negThree = (algorithmFPType)(-3.0);
    int iCurClass = (int)curClass;
    const algorithmFPType *Pptr = P + curClass * n;

    algorithmFPType sumW = 0.0;
    for ( size_t i = 0; i < n; i++ )
    {
        algorithmFPType p = Pptr[i];
        w[i] = p * (one - p);
        if (thrW > w[i]) { w[i] = thrW; }
        sumW += w[i];

        if (label[i] == iCurClass)
        {
            if (p > thrZ) { z[i] = one / p; }
            else { z[i] = three; }
        }
        else
        {
            if (one - p > thrZ) { z[i] = -one / (one - p); }
            else { z[i] = negThree; }
        }
    }

    algorithmFPType invSumW = one / sumW;
    for ( size_t i = 0; i < n; i++ )
    {
        w[i] *= invSumW;
    }
}

/**
 *  \brief Calculate training accuracy
 *
 *  \param n[in]        Number of observations
 *  \param nc[in]       Number of classes
 *  \param y_label[in]  Classes labels
 *  \param P[in]        Array of probabilities of size nc x n
 *  \param lCurPtr[out] Log-likelihood of the model
 *  \param accPtr[out]  Training accuracy
 */
template<typename algorithmFPType, CpuType cpu>
void calculateAccuracy( size_t n, size_t nc, const int *y_label, algorithmFPType *P,
                        algorithmFPType& lCur, algorithmFPType& acc)
{
    algorithmFPType lPrev = lCur;
    lCur = 0.0;
    for ( size_t i = 0; i < n; i++)
    {
        lCur -= daal::internal::Math<algorithmFPType,cpu>::sLog(P[y_label[i] * n + i]);
    }
    const algorithmFPType diff = daal::internal::Math<algorithmFPType, cpu>::sFabs(lPrev - lCur);
    acc  = daal::internal::Math<algorithmFPType,cpu>::sMin( diff, diff / (lPrev + (algorithmFPType)1e-6) );
}

} // namepsace internal
} // namespace prediction
} // namespace logitboost
} // namespace algorithms
} // namespace daal

#endif
