using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.NumPy;

namespace Tensorflow.OpencvAdapter.Extensions
{
    public static class NDArrayExtensions
    {
        /// <summary>
        /// If the array already has a shared Mat, then return it. 
        /// Otherwise return the result of `NDArray.ToMat()`.
        /// </summary>
        /// <param name="array"></param>
        /// <returns></returns>
        public static Mat AsMat(this NDArray array)
        {
            if(array is CvNDArray cvarray)
            {
                return cvarray.AsMat();
            }
            else
            {
                return array.ToMat(false);
            }
        }

        internal static InputArray? ToInputArray(this NDArray? array)
        {
            if(array is null)
            {
                return null;
            }
            return (InputArray)(array.AsMat());
        }
    }
}
