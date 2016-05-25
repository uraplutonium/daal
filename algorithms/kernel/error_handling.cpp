/** file error_handling.cpp */
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

#include <cstring>
#include "error_handling.h"
#include "data_management/data/data_utils.h"
#include "daal_string.h"
#include "mkl_daal.h"

namespace daal
{
namespace services
{
/*
    Example:

    SharedPtr<Error> e(new Error(ErrorIncorrectNumberOfFeatures));
    e->addIntDetail(Row, 10);
    e->addIntDetail(Column, 40);
    e->addStringDetail(services::Method, services::String("CorrelationDense"));
    this->_errors->add(e);
    this->_errors->add(services::ErrorIncorrectNumberOfObservations);
    this->_errors->add(services::ErrorIncorrectNumberOfElementsInResultCollection);
*/
DAAL_EXPORT const int daal::services::interface1::String::__DAAL_STR_MAX_SIZE = 4096;

void String::initialize(const char *str, const size_t length)
{
    if(length)
    {
        _c_str = (char *)daal::services::daal_malloc(sizeof(char) * (length + 1));
        fpk_serv_strncpy_s(_c_str, length + 1, str, length + 1);
    }
}

String::String(const char *str, size_t capacity) : _c_str(0)
{
    size_t strLength = 0;
    if(str)
    {
        strLength = strnlen(str, String::__DAAL_STR_MAX_SIZE);
    }
    initialize(str, strLength);
};

String::String(const String &str) : _c_str(0)
{
    initialize(str.c_str(), str.length());
};

String::~String()
{
    if(_c_str) { daal_free(_c_str); }
}

size_t String::length() const
{
    if(_c_str)
    {
        return strnlen(_c_str, String::__DAAL_STR_MAX_SIZE);
    }
    return 0;
}

void String::add(const String &str)
{
    size_t prevLength = length();
    char *prevStr = (char *)daal::services::daal_malloc(sizeof(char) * (prevLength + 1));
    fpk_serv_strncpy_s(prevStr, prevLength + 1, _c_str, prevLength + 1);

    size_t newLength = prevLength + str.length() + 1;
    if(_c_str) { daal_free(_c_str); }
    _c_str = (char *)daal::services::daal_malloc(sizeof(char) * (newLength + 1));

    fpk_serv_strncpy_s(_c_str, prevLength + 1, prevStr, prevLength + 1);
    fpk_serv_strncat_s(_c_str, newLength, str.c_str(), newLength - prevLength);

    if(prevStr) { daal_free(prevStr); }
}

String &String::operator+ (const String &str)
{
    add(str);
    return *this;
}

char String::operator[] (size_t index) const
{
    return _c_str[index];
}

char String::get(size_t index) const
{
    return _c_str[index];

}

const char *String::c_str() const
{
    return _c_str;
}

namespace
{
template<class T>
void toString(T number, char *buffer)
{}

template<> void toString<int>(int value, char *buffer)
{
#if defined(_WIN32) || defined(_WIN64)
    sprintf_s(buffer, String::__DAAL_STR_MAX_SIZE, "%d", value);
#else
    snprintf(buffer, String::__DAAL_STR_MAX_SIZE, "%d", value);
#endif
}

template<> void toString<double>(double value, char *buffer)
{
#if defined(_WIN32) || defined(_WIN64)
    sprintf_s(buffer, String::__DAAL_STR_MAX_SIZE, "%f", value);
#else
    snprintf(buffer, String::__DAAL_STR_MAX_SIZE, "%f", value);
#endif
}

template<> void toString<String>(String value, char *buffer)
{
    fpk_serv_strncpy_s(buffer, String::__DAAL_STR_MAX_SIZE, value.c_str(), String::__DAAL_STR_MAX_SIZE - value.length() );
}

/**
* <a name="DAAL-ENUM-SERVICES__MESSAGE"></a>
* \brief Class that represents Message
* \tparam IDType Type of message
*/
template<class IDType>
class Message
{
public:
    /**
    * Constructs Message from identifier and description
    * \param[in] id Error identifier
    * \param[in] description Description for message
    */
    Message(const IDType &id, const char *description) : _id(id), _description(description) {};

    /**
    * Destructor of Message class
    */
    virtual ~Message() {};

    /**
    * Returns identifier of a message
    * \return identifier of a message
    */
    const IDType id() const { return _id; }

    /**
    * Returns description of a message
    * \return description of a message
    */
    const char *description() const { return _description.c_str(); }

private:
    IDType _id;
    const String _description;
};

/**
* <a name="DAAL-ENUM-SERVICES__MESSAGECOLLECTION"></a>
* \brief Class that represents an Message collection
* \tparam IDType Type of message in message collection
*/
template<class IDType>
class MessageCollection : public Collection<SharedPtr<Message<IDType> > >
{
public:
    /**
    * Constructs a message collection
    * \param[in] noMessageFound Index retuned when no corresponding element is found in the collection
    */
    MessageCollection(IDType noMessageFound) : _noMessageFound(noMessageFound) {};

    /**
    * Finds message for error by error ID
    * \param[in] id Error identifier
    * \return Pointer to message
    */
    services::SharedPtr<Message<IDType> > find(IDType id) const
    {
        bool found = false;
        size_t index = 0;

        if(this->size() == 0) { return find(_noMessageFound); }

        for(size_t i = 0; i < this->size() && found == false; i++)
        {
            if((*this)[i]->id() == id)
            {
                found = true;
                index = i;
            }
        }

        if(found) { return (*this)[index]; }
        else { return find(_noMessageFound); }
    }

    /**
    * Destructor of a message collection
    */
    virtual ~MessageCollection() {};

protected:
    const IDType _noMessageFound;
};

/**
* <a name="DAAL-ENUM-SERVICES__ERRORMESSAGECOLLECTION"></a>
* \brief Class that represents an error message collection
*/
class ErrorMessageCollection : public MessageCollection<ErrorID>
{
public:
    /**
    * Constructs an error message collection
    */
    ErrorMessageCollection() : MessageCollection<ErrorID>(NoErrorMessageFound)
    {
        parseResourceFile();
    }

    /**
    * Destructor of an error message collection
    */
    ~ErrorMessageCollection() {}

protected:
    void parseResourceFile();
};

/**
* <a name="DAAL-ENUM-SERVICES__ERRORDETAILCOLLECTION"></a>
* \brief Class that represents an error detail collection
*/
class ErrorDetailCollection : public MessageCollection<ErrorDetailID>
{
public:
    /**
    * Construct an error detail collection
    */
    ErrorDetailCollection(): MessageCollection<ErrorDetailID>(NoErrorMessageDetailFound)
    {
        parseResourceFile();
    }

    /**
    * Destructor of an error detail collection
    */
    ~ErrorDetailCollection() {}

protected:
    void parseResourceFile();
};

const ErrorMessageCollection& errorMessageCollection()
{
    static const ErrorMessageCollection inst;
    return inst;
}

const ErrorDetailCollection& errorDetailCollection()
{
    static const ErrorDetailCollection inst;
    return inst;
}

int cat(const char *source, char *destination)
{
    return fpk_serv_strncat_s(destination,
                              String::__DAAL_STR_MAX_SIZE, source,
                              String::__DAAL_STR_MAX_SIZE - strnlen(destination, String::__DAAL_STR_MAX_SIZE));
}
}

template<typename T>
void ErrorDetail<T>::describe(char* str) const
{
    fpk_serv_strncat_s(str, String::__DAAL_STR_MAX_SIZE, errorDetailCollection().find(id())->description(),
        String::__DAAL_STR_MAX_SIZE - strnlen(str, String::__DAAL_STR_MAX_SIZE));
    fpk_serv_strncat_s(str, String::__DAAL_STR_MAX_SIZE, ": ",
        String::__DAAL_STR_MAX_SIZE - strnlen(str, String::__DAAL_STR_MAX_SIZE));

    char buffer[String::__DAAL_STR_MAX_SIZE] = { 0 };
    toString<T>(value(), buffer);
    fpk_serv_strncat_s(str, String::__DAAL_STR_MAX_SIZE, buffer,
        String::__DAAL_STR_MAX_SIZE - strnlen(str, String::__DAAL_STR_MAX_SIZE));

    fpk_serv_strncat_s(str, String::__DAAL_STR_MAX_SIZE, "\n",
        String::__DAAL_STR_MAX_SIZE - strnlen(str, String::__DAAL_STR_MAX_SIZE));
}

Error::Error(const ErrorID &id) : _id(id), _details(nullptr){}

Error::Error(const Error &e) : _id(e._id), _details(nullptr)
{
    ErrorDetailBase* pCur = nullptr;
    for(auto ptr = e.details(); ptr; ptr = ptr->next())
    {
        auto pClone = ptr->clone();
        if(pCur)
        {
            pCur->addNext(pClone);
            pCur = pClone;
        }
        else
        {
            _details = pCur = pClone;
        }
    }
}

Error::~Error()
{
    for(auto ptr = _details; ptr; ptr = ptr->next())
        delete ptr;
}

const char *Error::description() const { return errorMessageCollection().find(_id)->description(); }

Error& Error::addDetail(ErrorDetailBase* detail)
{
    if(detail)
    {
        auto ptr = _details;
        if(ptr)
        {
            for(; ptr->next(); ptr = ptr->next());
            ptr->addNext(detail);
        }
        else
            _details = detail;
    }
    return *this;
}

Error& Error::addIntDetail(const ErrorDetailID &id, const int &value)
{
    return addDetail(new ErrorDetail<int>(id, value));
}

Error& Error::addDoubleDetail(const ErrorDetailID &id, const double &value)
{
    return addDetail(new ErrorDetail<double>(id, value));
}

Error& Error::addStringDetail(const ErrorDetailID &id, const String &value)
{
    return addDetail(new ErrorDetail<String>(id, value));
}

KernelErrorCollection::KernelErrorCollection(const KernelErrorCollection &other) : super(other), _description(0)
{
}

Error& KernelErrorCollection::add(const ErrorID &id)
{
    ErrorPtr p(new Error(id));
    push_back(p);
    return *p.get();
}

void KernelErrorCollection::add(const ErrorPtr &e)
{
    push_back(e);
}

void KernelErrorCollection::add(const services::SharedPtr<KernelErrorCollection> &e)
{
    const super& p = *e;
    for(size_t i = 0; i < e->size(); i++)
        push_back(p[i]);
}

KernelErrorCollection::~KernelErrorCollection()
{
    if(_description)
        daal_free(_description);
}

const char *KernelErrorCollection::getDescription() const
{
    if(size() == 0)
    {
        if(_description) { daal_free(_description); }
        _description = (char *)daal::services::daal_malloc(sizeof(char) * 1);
        _description[0] = '\0';
        return _description;
    }

    size_t descriptionSize = 0;
    char **errorDescription = (char **)daal::services::daal_malloc(sizeof(char *) * size());

    for(size_t i = 0; i < size(); i++)
    {
        errorDescription[i] = (char *)daal::services::daal_malloc(sizeof(char) * (String::__DAAL_STR_MAX_SIZE));
        errorDescription[i][0] = '\0';

        services::SharedPtr<Error> e = _array[i];

        const char *currentDescription = errorMessageCollection().find(e->id())->description();
        cat(currentDescription, errorDescription[i]);

        const char *newLine = "\n";
        cat(newLine, errorDescription[i]);

        if(e->details())
        {
            const char *details = "Details:\n";
            cat(details, errorDescription[i]);
            for(const auto* ptr = e->details(); ptr; ptr = ptr->next())
                ptr->describe(errorDescription[i]);
        }

        descriptionSize += strnlen(errorDescription[i], String::__DAAL_STR_MAX_SIZE);
    }

    if(_description) { daal_free(_description); }
    _description = (char *)daal::services::daal_malloc(sizeof(char) * (descriptionSize + 1));
    _description[0] = '\0';

    for(size_t i = 0; i < size(); i++)
    {
        cat(errorDescription[i], _description);
        daal_free(errorDescription[i]);
    }

    daal_free(errorDescription);
    return _description;
}

size_t KernelErrorCollection::size() const { return super::size(); }

Error* KernelErrorCollection::at(size_t index)
{
    return super::operator[](index).get();
}

const Error* KernelErrorCollection::at(size_t index) const
{
    return super::operator[](index).get();
}

Error* KernelErrorCollection::operator[](size_t index)
{
    return super::operator[](index).get();
}

const Error* KernelErrorCollection::operator[](size_t index) const
{
    return super::operator[](index).get();
}

void ErrorMessageCollection::parseResourceFile()
{
    // Input errors: -1..-1999
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorNullNumericTable,
                                                                          "Null numeric table is not supported")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectNumberOfColumns,
                                                                          "Number of columns in numeric table is incorrect")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectNumberOfRows,
                                                                          "Number of rows in numeric table is incorrect")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectTypeOfNumericTable,
                                                                          "Incorrect type of Numeric Table")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorUnsupportedCSRIndexing,
                                                                          "CSR Numeric Table has unsupported indexing type")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorSignificanceLevel,
        "Incorrect significance level value")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorAccuracyThreshold,
        "Incorrect accuracy threshold")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectNumberOfBetas,
        "Incorrect number of betas in linear regression model")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectNumberOfBetasInReducedModel,
        "Incorrect number of betas in reduced linear regression model")));

    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorMethodNotSupported,
              "Method not supported by the algorithm")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectNumberOfFeatures,
              "Number of columns in numeric table is incorrect")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectNumberOfObservations,
              "Number of rows in numeric table is incorrect")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectSizeOfArray, "Incorrect size of array")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorNullParameterNotSupported,
              "Null parameter is not supported by the algorithm")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectNumberOfArguments,
              "Number of arguments is incorrect")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectInputNumericTable,
              "Input numeric table is incorrect")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorEmptyInputNumericTable,
              "Input numeric table is empty")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectDataRange, "Data range is incorrect")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorPrecomputedStatisticsIndexOutOfRange,
              "Precomputed statistics index is out of range")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectNumberOfInputNumericTables,
              "Incorrect number of input numeric tables")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectNumberOfOutputNumericTables,
              "Incorrect number of output numeric tables")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorNullInputNumericTable,
              "Null input numeric table is not supported")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorNullOutputNumericTable,
              "Null output numeric table is not supported")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorNullModel, "Null model is not supported")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorInconsistentNumberOfRows,
              "Number of rows in provided numeric tables is inconsistent")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorInconsistentNumberOfColumns,
              "Number of columns in provided numeric tables is inconsistent")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectSizeOfInputNumericTable,
              "Number of columns or rows in input numeric table is incorrect")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectSizeOfOutputNumericTable,
              "Number of columns or rows in output numeric table is incorrect")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectNumberOfRowsInInputNumericTable,
              "Number of rows in input numeric table is incorrect")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectNumberOfColumnsInInputNumericTable,
              "Number of columns in input numeric table is incorrect")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectNumberOfRowsInOutputNumericTable,
              "Number of rows in output numeric table is incorrect")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(
                  ErrorIncorrectNumberOfColumnsInOutputNumericTable,
                  "Number of columns in output numeric table is incorrect")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectTypeOfInputNumericTable,
              "Incorrect type of input NumericTable")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectTypeOfOutputNumericTable,
              "Incorrect type of output NumericTable")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectNumberOfElementsInInputCollection,
              "Incorrect number of elements in input collection")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectNumberOfElementsInResultCollection,
              "Incorrect number of elements in result collection")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorNullInput, "Input not set")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorNullResult, "Result not set")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectParameter, "Incorrect parameter")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorModelNotFullInitialized,
              "Model is not full initialized")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectIndex, "Index in collection is out of range")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorDataArchiveInternal, "Incorrect size of data block")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorNullPartialModel, "Null partial model is not supported")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorNullInputDataCollection,
              "Null input data collection is not supported")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorNullOutputDataCollection,
              "Null output data collection is not supported")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorNullPartialResult,
              "Partial result not set")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectNumberOfInputNumericTensors,
                "Incorrect number of elements in input collection")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectNumberOfOutputNumericTensors,
                "Incorrect number of elements in output collection")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorNullTensor,
                                                                          "Null input or result tensor is not supported")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectNumberOfDimensionsInTensor,
                                                                          "Number of dimensions in the tensor is incorrect")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectSizeOfDimensionInTensor,
                                                                          "Size of the dimension in input tensor is incorrect")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorNullLayerData,
                                                                          "Null layer data is not supported")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectSizeOfLayerData,
                                                                          "Incorrect number of elements in layer data collection")));

    // Environment errors: -2000..-2999
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorCpuNotSupported, "CPU not supported")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorMemoryAllocationFailed, "Memory allocation failed")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorEmptyDataBlock, "Empty data block")));

    // Workflow errors: -3000..-3999
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectCombinationOfComputationModeAndStep,
              "Incorrect combination of computation mode and computation step")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorDictionaryAlreadyAvailable,
              "Data Dictionary is already available")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorDictionaryNotAvailable,
              "Data Dictionary is not available")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorNumericTableNotAvailable,
              "Numeric Table is not available")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorNumericTableAlreadyAllocated,
              "Numeric Table was already allocated")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorNumericTableNotAllocated,
              "Numeric Table is not allocated")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorPrecomputedSumNotAvailable,
              "Precomputed sums are not available")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorPrecomputedMinNotAvailable,
              "Precomputed minimum values are not available")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorPrecomputedMaxNotAvailable,
              "Precomputed maximum values are not available")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorServiceMicroTableInternal,
              "Numeric Table internal error")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorEmptyCSRNumericTable, "CSR Numeric Table is empty")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorEmptyHomogenNumericTable,
              "Homogeneous Numeric Table is empty")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorSourceDataNotAvailable,
              "Source data is not available")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorEmptyDataSource, "Data source is empty")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectClassLabels,
              "Class labels provided to classification algorithm are incorrect")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectSizeOfModel, "Incorrect size of model")));

    // Common computation errors: -4000...
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorInputSigmaMatrixHasNonPositiveMinor,
              "Input sigma matrix has non positive minor")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorInputSigmaMatrixHasIllegalValue,
              "Input sigma matrix has illegal value")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectInternalFunctionParameter,
              "Incorrect parameter in internal function call")));

    /* Apriori algorithm errors -5000..-5199 */
    push_back(services::SharedPtr<Message<ErrorID> >(
                  new Message<ErrorID>(ErrorAprioriIncorrectItemsetTableSize,
                                       "Number of rows in the output table containing 'large' item sets is too small")));
    push_back(services::SharedPtr<Message<ErrorID> >(
                  new Message<ErrorID>(ErrorAprioriIncorrectSupportTableSize,
                                       "Number of rows in the output table containing 'large' item sets support values is too small")));
    push_back(services::SharedPtr<Message<ErrorID> >(
                  new Message<ErrorID>(ErrorAprioriIncorrectLeftRuleTableSize,
                                       "Number of rows in the output table containing left parts of the association rules is too small")));
    push_back(services::SharedPtr<Message<ErrorID> >(
                  new Message<ErrorID>(ErrorAprioriIncorrectRightRuleTableSize,
                                       "Number of rows in the output table containing right parts of the association rules is too small")));
    push_back(services::SharedPtr<Message<ErrorID> >(
                  new Message<ErrorID>(ErrorAprioriIncorrectConfidenceTableSize,
                                       "Number of rows in the output table containing association rules confidence is too small")));

    // BrownBoost errors: -5200..-5399

    // Cholesky errors: -5400..-5599
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorCholeskyInternal, "Cholesky internal error")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorInputMatrixHasNonPositiveMinor,
              "Input matrix has non positive minor")));

    // Covariance errors: -5600..-5799
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorCovarianceInternal, "Covariance internal error")));

    // Distance errors: -5800..-5999

    // EM errors: -6000..-6099
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorEMMatrixInverse,
              "Sigma matrix on M-step cannot be inverted")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorEMIncorrectToleranceToConverge,
              "Incorrect value of tolerance to converge in EM parameter")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorEMIllConditionedCovarianceMatrix,
              "Ill-conditioned covariance matrix")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorEMIncorrectMaxNumberOfIterations,
              "Incorrect maximum number of iterations value in EM parameter")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorEMNegativeDefinedCovarianceMartix,
              "Negative-defined covariance matrix")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorEMEmptyComponent,
              "Empty component during computation")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorEMCovariance,
              "Error during covariance computation for component on M step")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorEMIncorrectNumberOfComponents,
              "Incorrect number of components value in EM parameter")));

    // EM initialization errors: -6100..-6199
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorEMInitNoTrialConverges,
              "No trial of internal EM start converges")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorEMInitIncorrectToleranceToConverge,
              "Incorrect tolerance to converge value in EM initialization parameter")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorEMInitIncorrectDepthNumberIterations,
              "Incorrect depth number of iterations value in EM init parameter")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorEMInitIncorrectNumberOfTrials,
              "Incorrect number of trials value in EM initialization parameter")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorEMInitIncorrectNumberOfComponents,
              "Incorrect number of components value in EM initialization parameter")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorEMInitInconsistentNumberOfComponents,
              "Inconsistent number of component: number of observations should be greater than number of components")));

    // KernelFunction errors: -6200..-6399

    // KMeans errors: -6400..-6599

    // Linear Rergession errors: -6600..-6799
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorLinearRegressionInternal,
              "Linear Regression internal error")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorNormEqSystemSolutionFailed,
              "Failed to solve the system of normal equations")));

    // LogitBoots errors: -6800..-6999

    // LowOrderMoments errors: -7000..-7199
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorLowOrderMomentsInternal, "Low Order Moments internal error")));

    // MultiClassClassifier errors: -7200..-7399
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectNumberOfClasses,
              "Number of classes provided to multi-class classifier is too small")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorMultiClassNullTwoClassTraining,
              "Null two-class classifier training algorithm is not supported")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorMultiClassFailedToTrainTwoClassClassifier,
              "Failed to train a model of two-class classifier")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorMultiClassFailedToComputeTwoClassPrediction,
              "Failed to compute prediction based on two-class classifier model")));

    // NaiveBayes errors: -7400..-7599

    // OutlierDetection errors: -7600..-7799
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorOutlierDetectionInternal,
              "Outlier Detection internal error")));

    /* PCA errors: -7800..-7999 */
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorPCAFailedToComputeCorrelationEigenvalues,
              "Failed to compute eigenvalues of the correlation matrix")));
    push_back(services::SharedPtr<Message<ErrorID> >(
                  new Message<ErrorID>(ErrorPCACorrelationInputDataTypeSupportsOfflineModeOnly,
                                       "This type of the input data supports only offline mode of the computations")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorIncorrectCrossProductTableSize,
              "Number of columns or rows in cross-product numeric table is incorrect")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorCrossProductTableIsNotSquare,
              "Number of columns or rows in cross-product numeric table is not equal")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(
                  ErrorInputCorrelationNotSupportedInOnlineAndDistributed,
                  "Input correlation matrix is not supported in online and distributed computation modes")));

    // QR errors: -8000..-8199

    // Stump errors: -8200..-8399

    // SVD errors: -8400..-8599

    // SVM errors: -8600..-8799
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorSVMinnerKernel, "Error in kernel function")));

    // WeakLearner errors: -8800..-8999

    // Compression errors: -9000..-9199
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorCompressionNullInputStream,
              "Null input stream is not supported")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorCompressionNullOutputStream,
              "Null output stream is not supported")));

    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorCompressionEmptyInputStream,
              "Input stream of size 0 is not supported")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorCompressionEmptyOutputStream,
              "Output stream of size 0 is not supported")));

    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorZlibInternal, "Zlib internal error")));
    push_back(services::SharedPtr<Message<ErrorID> >(
                  new Message<ErrorID>(ErrorZlibDataFormat,
                                       "Input compressed stream is in wrong format, corrupted or contains not a whole number of compressed blocks")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorZlibParameters, "Unsupported Zlib parameters")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorZlibMemoryAllocationFailed,
              "Internal Zlib memory allocation failed")));
    push_back(services::SharedPtr<Message<ErrorID> >(
                  new Message<ErrorID>(ErrorZlibNeedDictionary,
                                       "Specific dictionary is needed for decompression, currently unsupported Zlib feature")));

    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorBzip2Internal, "Bzip2 internal error")));
    push_back(services::SharedPtr<Message<ErrorID> >(
                  new Message<ErrorID>(ErrorBzip2DataFormat,
                                       "Input compressed stream is in wrong format, corrupted or contains not a whole number of compressed blocks")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorBzip2Parameters, "Unsupported Bzip2 parameters")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorBzip2MemoryAllocationFailed,
              "Internal Bzip2 memory allocation failed")));

    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorLzoInternal, "LZO internal error")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorLzoOutputStreamSizeIsNotEnough,
              "Size of output stream is not enough to start compression")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorLzoDataFormat,
              "Input compressed stream is in wrong format or corrupted")));
    push_back(services::SharedPtr<Message<ErrorID> >(
                  new Message<ErrorID>(ErrorLzoDataFormatLessThenHeader,
                                       "Size of input compressed stream is less then compressed block header size")));
    push_back(services::SharedPtr<Message<ErrorID> >(
                  new Message<ErrorID>(ErrorLzoDataFormatNotFullBlock,
                                       "Input compressed stream contains not a whole number of compressed blocks")));

    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorRleInternal, "RLE internal error")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorRleOutputStreamSizeIsNotEnough,
              "Size of output stream is not enough to start compression")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorRleDataFormat,
              "Input compressed stream is in wrong format or corrupted")));
    push_back(services::SharedPtr<Message<ErrorID> >(
                  new Message<ErrorID>(ErrorRleDataFormatLessThenHeader,
                                       "Size of input compressed stream is less then compressed block header size")));
    push_back(services::SharedPtr<Message<ErrorID> >(
                  new Message<ErrorID>(ErrorRleDataFormatNotFullBlock,
                                       "Input compressed stream contains not a whole number of compressed blocks")));

    // Quantile error: -10000..-11000
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorQuantileOrderValueIsInvalid,
              "Quantile order value is invalid")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorQuantilesInternal,
              "Quantile internal error")));

    // ALS errors: -11000..-12000
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorALSInternal,
              "ALS algorithm internal error")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(
                  ErrorALSInconsistentSparseDataBlocks,
                  "Failed to find a non-zero value with needed indices in a sparse data block")));

    // Sorting error: -12000..-13000
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorSortingInternal,
              "Sorting internal error")));

    // SGD error: -13000..-14000
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorNegativeLearningRate,
              "Negative learning rate")));

    // Normalization errors: -14000..-15000
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorMeanAndStandardDeviationComputing,
              "Computation of mean and standard deviation failed")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorNullVariance,
              "Failed to normalize data in column: it has null variance")));

    // Sum of functions error: -14000..-15000
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorZeroNumberOfTerms,
              "Number of terms can not be zero")));

    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorConvolutionInternal,
              "Convolution layer internal error")));

    //Math errors: -90000..-100000
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorDataSourseNotAvailable,
              "ErrorDataSourseNotAvailable")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorHandlesSQL, "ErrorHandlesSQL")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorODBC, "ErrorODBC")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorSQLstmtHandle, "ErrorSQLstmtHandle")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorOnFileOpen, "Error on file open")));

    // Other errors: -100000..
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorObjectDoesNotSupportSerialization,
              "SerializationIface is not implemented or implemented incorrectly")));

    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorCouldntAttachCurrentThreadToJavaVM,
              "Couldn't attach current thread to Java VM")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorCouldntCreateGlobalReferenceToJavaObject,
              "Couldn't create global reference to Java object")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorCouldntFindJavaMethod, "Couldn't find Java method")));
    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(ErrorCouldntFindClassForJavaObject,
              "Couldn't find class for Java object")));

    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(UnknownError, "UnknownError")));

    push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(NoErrorMessageFound, "NoErrorMessageFound")));
}

void ErrorDetailCollection::parseResourceFile()
{
    push_back(services::SharedPtr<Message<ErrorDetailID> >(new Message<ErrorDetailID>(NoErrorMessageDetailFound, "NoErrorMessageDetailFound")));
    push_back(services::SharedPtr<Message<ErrorDetailID> >(new Message<ErrorDetailID>(Row, "Row")));
    push_back(services::SharedPtr<Message<ErrorDetailID> >(new Message<ErrorDetailID>(Column, "Column")));
    push_back(services::SharedPtr<Message<ErrorDetailID> >(new Message<ErrorDetailID>(Rank, "Rank")));
    push_back(services::SharedPtr<Message<ErrorDetailID> >(new Message<ErrorDetailID>(StatisticsName, "StatisticsName")));
    push_back(services::SharedPtr<Message<ErrorDetailID> >(new Message<ErrorDetailID>(Method, "Method")));
    push_back(services::SharedPtr<Message<ErrorDetailID> >(new Message<ErrorDetailID>(Iteration, "Iteration")));
    push_back(services::SharedPtr<Message<ErrorDetailID> >(new Message<ErrorDetailID>(Component, "Component")));
    push_back(services::SharedPtr<Message<ErrorDetailID> >(new Message<ErrorDetailID>(Minor, "Matrix minor")));
    push_back(services::SharedPtr<Message<ErrorDetailID> >(new Message<ErrorDetailID>(ArgumentName, "Argument name")));
    push_back(services::SharedPtr<Message<ErrorDetailID> >(new Message<ErrorDetailID>(ElementInCollection, "ElementInCollection")));
    push_back(services::SharedPtr<Message<ErrorDetailID> >(new Message<ErrorDetailID>(Dimension, "Tensor dimension")));
}


}
}
