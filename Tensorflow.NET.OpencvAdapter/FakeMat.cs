using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.NumPy;

namespace Tensorflow.OpencvAdapter
{
    /// <summary>
    /// A class to enable implicit conversion from NDArray to Mat.
    /// </summary>
    public class FakeMat: Mat
    {
        internal FakeMat(int rows, int cols, MatType type, IntPtr data, long step = 0) : 
            base(rows, cols, type, data, step)
        {

        }

        internal FakeMat(int rows, int cols, MatType type) : base(rows, cols, type)
        {

        }

        internal static FakeMat FromNDArray(NDArray array)
        {
            if (CvNDArray.AdapterMode == OpencvAdapterMode.StrictNoCopy || CvNDArray.AdapterMode == OpencvAdapterMode.AllowCopy)
            {
                var dataPointer = array.TensorDataPointer;
                var (matType, rows, cols) = AdapterUtils.DeduceMatInfoFromNDArray(array.shape, array.dtype);
                return new FakeMat(rows, cols, matType, dataPointer);
            }
            else // AdapterMode == OpencvAdapterMode.AlwaysCopy
            {
                var (matType, rows, cols) = AdapterUtils.DeduceMatInfoFromNDArray(array.shape, array.dtype);
                FakeMat m = new FakeMat(rows, cols, matType);
                AdapterUtils.SetMatFromNDArrayData(array, m);
                return m;
            }
        }

        public static implicit operator FakeMat(NDArray array)
        {
            return FromNDArray(array);
        }
    }
}
