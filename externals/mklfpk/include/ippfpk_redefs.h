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

#if !defined( __IPPFPK_REDEFS_H__ )
#define __IPPFPK_REDEFS_H__

#if !defined( __NO_FPK_REDEF__ )

#define ippInit ippfpkInit
#define ippsAdler32_8u ippfpksAdler32_8u
#define ippsBWTFwdGetBufSize_SelectSort_8u ippfpksBWTFwdGetBufSize_SelectSort_8u
#define ippsBWTFwd_SelectSort_8u ippfpksBWTFwd_SelectSort_8u
#define ippsCRC32_8u ippfpksCRC32_8u
#define ippsCRC32_BZ2_8u ippfpksCRC32_BZ2_8u
#define ippsCopyLE_1u ippfpksCopyLE_1u
#define ippsCopy_8u ippfpksCopy_8u
#define ippsDecodeBlockGetSize_BZ2_8u ippfpksDecodeBlockGetSize_BZ2_8u
#define ippsDecodeBlock_BZ2_16u8u ippfpksDecodeBlock_BZ2_16u8u
#define ippsDecodeHuffGetSize_BZ2_8u16u ippfpksDecodeHuffGetSize_BZ2_8u16u
#define ippsDecodeHuffInit_BZ2_8u16u ippfpksDecodeHuffInit_BZ2_8u16u
#define ippsDecodeHuff_BZ2_8u16u ippfpksDecodeHuff_BZ2_8u16u
#define ippsDecodeLZOSafe_8u ippfpksDecodeLZOSafe_8u
#define ippsDecodeLZO_8u ippfpksDecodeLZO_8u
#define ippsDecodeRLEStateFlush_BZ2_8u ippfpksDecodeRLEStateFlush_BZ2_8u
#define ippsDecodeRLEStateInit_BZ2_8u ippfpksDecodeRLEStateInit_BZ2_8u
#define ippsDecodeRLEState_BZ2_8u ippfpksDecodeRLEState_BZ2_8u
#define ippsDecodeRLE_8u ippfpksDecodeRLE_8u
#define ippsDecodeZ1Z2_BZ2_16u8u ippfpksDecodeZ1Z2_BZ2_16u8u
#define ippsDeflateHuff_8u ippfpksDeflateHuff_8u
#define ippsDeflateLZ77_8u ippfpksDeflateLZ77_8u
#define ippsEncodeHuffGetSize_BZ2_16u8u ippfpksEncodeHuffGetSize_BZ2_16u8u
#define ippsEncodeHuffInit_BZ2_16u8u ippfpksEncodeHuffInit_BZ2_16u8u
#define ippsEncodeHuff_BZ2_16u8u ippfpksEncodeHuff_BZ2_16u8u
#define ippsEncodeLZOGetSize ippfpksEncodeLZOGetSize
#define ippsEncodeLZOInit_8u ippfpksEncodeLZOInit_8u
#define ippsEncodeLZO_8u ippfpksEncodeLZO_8u
#define ippsEncodeRLEFlush_BZ2_8u ippfpksEncodeRLEFlush_BZ2_8u
#define ippsEncodeRLEInit_BZ2_8u ippfpksEncodeRLEInit_BZ2_8u
#define ippsEncodeRLE_8u ippfpksEncodeRLE_8u
#define ippsEncodeRLE_BZ2_8u ippfpksEncodeRLE_BZ2_8u
#define ippsEncodeZ1Z2_BZ2_8u16u ippfpksEncodeZ1Z2_BZ2_8u16u
#define ippsExpandDictionary_8u_I ippfpksExpandDictionary_8u_I
#define ippsInflateBuildHuffTable ippfpksInflateBuildHuffTable
#define ippsInflate_8u ippfpksInflate_8u
#define ippsMTFFwd_8u ippfpksMTFFwd_8u
#define ippsMTFGetSize_8u ippfpksMTFGetSize_8u
#define ippsMTFInit_8u ippfpksMTFInit_8u
#define ippsMove_8u ippfpksMove_8u
#define ippsPackHuffContext_BZ2_16u8u ippfpksPackHuffContext_BZ2_16u8u
#define ippsRLEGetInUseTable_8u ippfpksRLEGetInUseTable_8u
#define ippsRLEGetSize_BZ2_8u ippfpksRLEGetSize_BZ2_8u
#define ippsReduceDictionary_8u_I ippfpksReduceDictionary_8u_I
#define ippsSet_32s ippfpksSet_32s
#define ippsUnpackHuffContext_BZ2_8u16u ippfpksUnpackHuffContext_BZ2_8u16u
#define ippsZero_8u ippfpksZero_8u

#endif /* __NO_FPK_REDEF__    */
#endif /* __IPPFPK_REDEFS_H__ */
