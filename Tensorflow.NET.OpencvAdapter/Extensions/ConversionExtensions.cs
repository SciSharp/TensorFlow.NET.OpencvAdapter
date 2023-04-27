using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using Tensorflow.NumPy;

namespace Tensorflow.OpencvAdapter.Extensions
{
    public static class ConversionExtensions
    {
        /// <summary>
        /// Convert the NDArray to Opencv Mat object. If the parameter clone is set to true, 
        /// then memory copying will happen. If it's set to false, the returned Mat object will
        /// share the memory with the source NDArray. So be careful if the NDArray will still be 
        /// modified after calling this method.
        /// </summary>
        /// <param name="array">The source NDArray.</param>
        /// <param name="clone">Whether to perform memory copying.</param>
        /// <returns></returns>
        public static Mat ToMat(this NDArray array, bool clone)
        {
            if (clone)
            {
                using(new OpencvAdapterContext(OpencvAdapterMode.AlwaysCopy))
                {
                    return AdapterUtils.ConvertNDArrayToMat(array);
                }
            }
            else
            {
                return AdapterUtils.ConvertNDArrayToMat(array);
            }
        }

        /// <summary>
        /// Convert the opencv Mat to NDArray. If the parameter clone is set to true, 
        /// then memory copying will happen. If it's set to false, the returned NDArray will
        /// share the memory with the source Mat. So be careful if the Mat will still be 
        /// modified after calling this method.
        /// </summary>
        /// <param name="mat">The source Opencv Mat object.</param>
        /// <param name="clone">Whether to perform memory copying.</param>
        /// <returns></returns>
        public static NDArray ToNDArray(this Mat mat, bool clone)
        {
            if (clone)
            {
                using (new OpencvAdapterContext(OpencvAdapterMode.AlwaysCopy))
                {
                    return new CvNDArray(mat);
                }
            }
            else
            {
                return new CvNDArray(mat);
            }
        }

        /// <summary>
        /// Convert the opencv Mat to NDArray. If the parameter clone is set to true, 
        /// then memory copying will happen. If it's set to false, the returned NDArray will
        /// share the memory with the source Mat. By default `clone` is set to false. 
        /// So be careful if the Mat will still be modified after calling this method.
        /// </summary>
        /// <param name="mat">The source Opencv Mat object.</param>
        /// <param name="clone">Whether to perform memory copying.</param>
        /// <returns></returns>
        public static NDArray numpy(this Mat mat, bool clone = false)
        {
            if (clone)
            {
                using (new OpencvAdapterContext(OpencvAdapterMode.AlwaysCopy))
                {
                    return new CvNDArray(mat);
                }
            }
            else
            {
                return new CvNDArray(mat);
            }
        }
    }
}
