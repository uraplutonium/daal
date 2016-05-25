/* file: PackedSymmetricMatrixUtils.java */
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

package com.intel.daal.data_management.data;

import java.io.Serializable;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

import com.intel.daal.services.DaalContext;

class PackedSymmetricMatrixUtils {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    interface SymmetricAccessIface {
        int getPosition(int row, int column, int nDim);
    }

    static class SymmetricAccess {
        static public SymmetricAccessIface getAccess(NumericTable.StorageLayout packedLayout) {
            if (packedLayout == NumericTable.StorageLayout.upperPackedSymmetricMatrix) {
                return new UpperSymmetricAccess();
            } else
            if (packedLayout == NumericTable.StorageLayout.lowerPackedSymmetricMatrix) {
                return new LowerSymmetricAccess();
            }
            return null;
        }
    }

    static class UpperSymmetricAccess implements SymmetricAccessIface, Serializable {
        @Override
        public int getPosition(int row, int column, int nDim) {
            if (row > column) {
                int tmp = row;
                row = column;
                column = tmp;
            }
            int rowStartOffset = ((2 * nDim - 1 * (row - 1)) * row) / 2;
            int colStartOffset = column - row;
            return rowStartOffset + colStartOffset;
       }
    }

    static class LowerSymmetricAccess implements SymmetricAccessIface, Serializable {
        @Override
        public int getPosition(int row, int column, int nDim) {
            if (row < column) {
                int tmp = row;
                row = column;
                column = tmp;
            }
            int rowStartOffset = ((2 + 1 * (row - 1)) * row) / 2;
            int colStartOffset = column;
            return rowStartOffset + colStartOffset;
        }
    }

    interface SymmetricUpCastIface {
        void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess);
        void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess);
    }

    interface SymmetricDownCastIface {
        void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess);
        void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess);
    }

    /** @private */
    static class SymmetricUpCast {
        static public SymmetricUpCastIface getCast(Class<?> fromCls, Class<?> toCls) {
            if (fromCls == Double.class || fromCls == double.class) {
                if (toCls == Double.class || toCls == double.class) {
                    return new SymmetricUpCastDouble2Double();
                } else if (toCls == Float.class || toCls == float.class) {
                    return new SymmetricUpCastDouble2Float();
                } else if (toCls == Integer.class || toCls == int.class) {
                    return new SymmetricUpCastDouble2Int();
                }
            } else if (fromCls == Float.class || fromCls == float.class) {
                if (toCls == Double.class || toCls == double.class) {
                    return new SymmetricUpCastFloat2Double();
                } else if (toCls == Float.class || toCls == float.class) {
                    return new SymmetricUpCastFloat2Float();
                } else if (toCls == Integer.class || toCls == int.class) {
                    return new SymmetricUpCastFloat2Int();
                }
            } else if (fromCls == Integer.class || fromCls == int.class) {
                if (toCls == Double.class || toCls == double.class) {
                    return new SymmetricUpCastInt2Double();
                } else if (toCls == Float.class || toCls == float.class) {
                    return new SymmetricUpCastInt2Float();
                } else if (toCls == Integer.class || toCls == int.class) {
                    return new SymmetricUpCastInt2Int();
                }
            } else if (fromCls == Long.class || fromCls == long.class) {
                if (toCls == Double.class || toCls == double.class) {
                    return new SymmetricUpCastLong2Double();
                } else if (toCls == Float.class || toCls == float.class) {
                    return new SymmetricUpCastLong2Float();
                } else if (toCls == Integer.class || toCls == int.class) {
                    return new SymmetricUpCastLong2Int();
                }
            }
            return null;
        }
    }

    /** @private */
    static class SymmetricDownCast {
        static public SymmetricDownCastIface getCast(Class<?> fromCls, Class<?> toCls) {
            if (fromCls == Double.class || fromCls == double.class) {
                if (toCls == Double.class || toCls == double.class) {
                    return new SymmetricDownCastDouble2Double();
                } else if (toCls == Float.class || toCls == float.class) {
                    return new SymmetricDownCastDouble2Float();
                } else if (toCls == Integer.class || toCls == int.class) {
                    return new SymmetricDownCastDouble2Int();
                } else if (toCls == Long.class || toCls == long.class) {
                    return new SymmetricDownCastDouble2Long();
                }
            } else if (fromCls == Float.class || fromCls == float.class) {
                if (toCls == Double.class || toCls == double.class) {
                    return new SymmetricDownCastFloat2Double();
                } else if (toCls == Float.class || toCls == float.class) {
                    return new SymmetricDownCastFloat2Float();
                } else if (toCls == Integer.class || toCls == int.class) {
                    return new SymmetricDownCastFloat2Int();
                } else if (toCls == Long.class || toCls == long.class) {
                    return new SymmetricDownCastFloat2Long();
                }
            } else if (fromCls == Integer.class || fromCls == int.class) {
                if (toCls == Double.class || toCls == double.class) {
                    return new SymmetricDownCastInt2Double();
                } else if (toCls == Float.class || toCls == float.class) {
                    return new SymmetricDownCastInt2Float();
                } else if (toCls == Integer.class || toCls == int.class) {
                    return new SymmetricDownCastInt2Int();
                } else if (toCls == Long.class || toCls == long.class) {
                    return new SymmetricDownCastInt2Long();
                }
            }
            return null;
        }
    }

    /** @private */
    static class SymmetricUpCastDouble2Double implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            double[] data = (double[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            double[] data = (double[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricDownCastDouble2Double implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            DoubleBuffer buf = (DoubleBuffer)src;
            double[] data = (double[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = buf.get (row * nDim + column);
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            DoubleBuffer buf = (DoubleBuffer)src;
            double[] data = (double[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = buf.get (row);
            }
        }
    }

    /** @private */
    static class SymmetricUpCastDouble2Float implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            double[] data = (double[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, (float)(data[symmetricAccess.getPosition(firstRow + row, column, nDim)]));
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            double[] data = (double[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, (float)(data[symmetricAccess.getPosition(firstRow + row, column, nDim)]));
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricDownCastDouble2Float implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            DoubleBuffer buf = (DoubleBuffer)src;
            float[] data = (float[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (float)(buf.get (row * nDim + column));
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            DoubleBuffer buf = (DoubleBuffer)src;
            float[] data = (float[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (float)(buf.get (row));
            }
        }
    }

    /** @private */
    static class SymmetricUpCastDouble2Int implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            double[] data = (double[])src;
            IntBuffer buf = (IntBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, (int)(data[symmetricAccess.getPosition(firstRow + row, column, nDim)]));
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            double[] data = (double[])src;
            IntBuffer buf = (IntBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, (int)(data[symmetricAccess.getPosition(firstRow + row, column, nDim)]));
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricDownCastDouble2Int implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            DoubleBuffer buf = (DoubleBuffer)src;
            int[] data = (int[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (int)(buf.get (row * nDim + column));
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            DoubleBuffer buf = (DoubleBuffer)src;
            int[] data = (int[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (int)(buf.get (row));
            }
        }
    }

    /** @private */
    static class SymmetricUpCastFloat2Double implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            float[] data = (float[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, (double)(data[symmetricAccess.getPosition(firstRow + row, column, nDim)]));
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            float[] data = (float[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, (double)(data[symmetricAccess.getPosition(firstRow + row, column, nDim)]));
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricDownCastFloat2Double implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            FloatBuffer buf = (FloatBuffer)src;
            double[] data = (double[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (double)(buf.get (row * nDim + column));
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            FloatBuffer buf = (FloatBuffer)src;
            double[] data = (double[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (double)(buf.get (row));
            }
        }
    }

    /** @private */
    static class SymmetricUpCastFloat2Float implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            float[] data = (float[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            float[] data = (float[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricDownCastFloat2Float implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            FloatBuffer buf = (FloatBuffer)src;
            float[] data = (float[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = buf.get (row * nDim + column);
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            FloatBuffer buf = (FloatBuffer)src;
            float[] data = (float[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = buf.get (row);
            }
        }
    }

    /** @private */
    static class SymmetricUpCastFloat2Int implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            float[] data = (float[])src;
            IntBuffer buf = (IntBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, (int)(data[symmetricAccess.getPosition(firstRow + row, column, nDim)]));
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            float[] data = (float[])src;
            IntBuffer buf = (IntBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, (int)(data[symmetricAccess.getPosition(firstRow + row, column, nDim)]));
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricDownCastFloat2Int implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            FloatBuffer buf = (FloatBuffer)src;
            int[] data = (int[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (int)(buf.get (row * nDim + column));
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            FloatBuffer buf = (FloatBuffer)src;
            int[] data = (int[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (int)(buf.get (row));
            }
        }
    }

    /** @private */
    static class SymmetricUpCastInt2Double implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            int[] data = (int[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, (double)data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            int[] data = (int[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, (double)data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricDownCastInt2Double implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            IntBuffer buf = (IntBuffer)src;
            double[] data = (double[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (double)buf.get (row * nDim + column);
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            IntBuffer buf = (IntBuffer)src;
            double[] data = (double[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (double)buf.get (row);
            }
        }
    }

    /** @private */
    static class SymmetricUpCastInt2Float implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            int[] data = (int[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, (float)(data[symmetricAccess.getPosition(firstRow + row, column, nDim)]));
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            int[] data = (int[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, (float)(data[symmetricAccess.getPosition(firstRow + row, column, nDim)]));
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricDownCastInt2Float implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            IntBuffer buf = (IntBuffer)src;
            float[] data = (float[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (float)(buf.get (row * nDim + column));
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            IntBuffer buf = (IntBuffer)src;
            float[] data = (float[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (float)(buf.get (row));
            }
        }
    }

    /** @private */
    static class SymmetricUpCastInt2Int implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            int[] data = (int[])src;
            IntBuffer buf = (IntBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            int[] data = (int[])src;
            IntBuffer buf = (IntBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricDownCastInt2Int implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            IntBuffer buf = (IntBuffer)src;
            int[] data = (int[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = buf.get (row * nDim + column);
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            IntBuffer buf = (IntBuffer)src;
            int[] data = (int[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = buf.get (row);
            }
        }
    }

    /** @private */
    static class SymmetricUpCastLong2Double implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            long[] data = (long[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, (double)data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            long[] data = (long[])src;
            DoubleBuffer buf = (DoubleBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, (double)data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricUpCastLong2Float implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            long[] data = (long[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, (float)data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            long[] data = (long[])src;
            FloatBuffer buf = (FloatBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, (float)data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricUpCastLong2Int implements SymmetricUpCastIface, Serializable {
        @Override
        public void upCast(int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            long[] data = (long[])src;
            IntBuffer buf = (IntBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    buf.put (row * nDim + column, (int)data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
                }
            }
            dst.position(0);
        }

        @Override
        public void upCastFeature(int column, int firstRow, int nRows, int nDim, Object src, Buffer dst, SymmetricAccessIface symmetricAccess) {
            dst.position(0);
            long[] data = (long[])src;
            IntBuffer buf = (IntBuffer)dst;
            for (int row = 0; row < nRows; row++) {
                buf.put (row, (int)data[symmetricAccess.getPosition(firstRow + row, column, nDim)]);
            }
            dst.position(0);
        }
    }

    /** @private */
    static class SymmetricDownCastDouble2Long implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            DoubleBuffer buf = (DoubleBuffer)src;
            long[] data = (long[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (long)(buf.get (row * nDim + column));
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            DoubleBuffer buf = (DoubleBuffer)src;
            long[] data = (long[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (long)(buf.get (row));
            }
        }
    }

    /** @private */
    static class SymmetricDownCastFloat2Long implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            FloatBuffer buf = (FloatBuffer)src;
            long[] data = (long[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (long)(buf.get (row * nDim + column));
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            FloatBuffer buf = (FloatBuffer)src;
            long[] data = (long[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (long)(buf.get (row));
            }
        }
    }

    /** @private */
    static class SymmetricDownCastInt2Long implements SymmetricDownCastIface, Serializable {
        @Override
        public void downCast(int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            IntBuffer buf = (IntBuffer)src;
            long[] data = (long[])dst;
            for (int row = 0; row < nRows; row++) {
                for (int column = 0; column < nDim; column++) {
                    data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (long)(buf.get (row * nDim + column));
                }
            }
        }

        @Override
        public void downCastFeature(int column, int firstRow, int nRows, int nDim, Buffer src, Object dst, SymmetricAccessIface symmetricAccess) {
            IntBuffer buf = (IntBuffer)src;
            long[] data = (long[])dst;
            for (int row = 0; row < nRows; row++) {
                data[symmetricAccess.getPosition(firstRow + row, column, nDim)] = (long)(buf.get (row));
            }
        }
    }
}
