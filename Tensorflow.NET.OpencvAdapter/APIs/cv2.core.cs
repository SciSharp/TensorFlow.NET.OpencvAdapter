using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.Checkpoint;
using Tensorflow.NumPy;
using Tensorflow.OpencvAdapter.Extensions;

namespace Tensorflow.OpencvAdapter.APIs
{
    public partial class Cv2API
    {
        /// <summary>
        /// Computes the source location of an extrapolated pixel.
        /// </summary>
        /// <param name="p">0-based coordinate of the extrapolated pixel along one of the axes, likely &lt;0 or &gt;= len</param>
        /// <param name="len">Length of the array along the corresponding axis.</param>
        /// <param name="borderType">Border type, one of the #BorderTypes, except for #BORDER_TRANSPARENT and BORDER_ISOLATED. 
        /// When borderType==BORDER_CONSTANT, the function always returns -1, regardless</param>
        /// <returns></returns>
        public int borderInterpolate(int p, int len, BorderTypes borderType)
        {
            return Cv2.BorderInterpolate(p, len, borderType);
        }

        /// <summary>
        /// Forms a border around the image
        /// </summary>
        /// <param name="src">The source image</param>
        /// <param name="top">Specify how much pixels in each direction from the source image rectangle one needs to extrapolate</param>
        /// <param name="bottom">Specify how much pixels in each direction from the source image rectangle one needs to extrapolate</param>
        /// <param name="left">Specify how much pixels in each direction from the source image rectangle one needs to extrapolate</param>
        /// <param name="right">Specify how much pixels in each direction from the source image rectangle one needs to extrapolate</param>
        /// <param name="borderType">The border type</param>
        /// <param name="value">The border value if borderType == Constant</param>
        public NDArray copyMakeBorder(NDArray src, int top, int bottom, int left, int right,
            BorderTypes borderType, Scalar? value = null)
        {
            Mat dst = new Mat();
            Cv2.CopyMakeBorder(src.AsMat(), dst, top, bottom, left, right, borderType, value);
            return new CvNDArray(dst);
        }

        /// <summary>
        /// Computes the per-element sum of two arrays or an array and a scalar.
        /// </summary>
        /// <param name="src1">The first source array</param>
        /// <param name="src2">The second source array. It must have the same size and same type as src1</param>
        /// <param name="dst">The destination array; it will have the same size and same type as src1</param>
        /// <param name="mask">The optional operation mask, 8-bit single channel array; specifies elements of the destination array to be changed. [By default this is null]</param>
        /// <param name="dtype"></param>
        public NDArray add(NDArray src1, NDArray src2, NDArray? dst = null, NDArray? mask = null,
            TF_DataType dtype = TF_DataType.DtInvalid)
        {
            Mat dstMat = dst is null ? new Mat() : dst.AsMat();
            var src1Mat = src1.AsMat();
            var src2Mat = src2.AsMat();
            Cv2.Add(src1Mat, src2Mat, dstMat, mask.ToInputArray(),
                dtype.ToMatTypeNumber(Math.Max(src1Mat.Channels(), src2Mat.Channels())));
            return dst ?? new CvNDArray(dstMat);
        }

        /// <summary>
        /// Calculates per-element difference between two arrays or array and a scalar
        /// </summary>
        /// <param name="src1">The first source array</param>
        /// <param name="src2">The second source array. It must have the same size and same type as src1</param>
        /// <param name="dst">The destination array; it will have the same size and same type as src1</param>
        /// <param name="mask">The optional operation mask, 8-bit single channel array; specifies elements of the destination array to be changed. [By default this is null]</param>
        /// <param name="dtype"></param>
        public NDArray subtract(NDArray src1, NDArray src2, NDArray? dst = null, NDArray? mask = null,
            TF_DataType dtype = TF_DataType.DtInvalid)
        {
            Mat dstMat = dst is null ? new Mat() : dst.AsMat();
            var src1Mat = src1.AsMat();
            var src2Mat = src2.AsMat();
            Cv2.Subtract(src1Mat, src2Mat, dstMat, mask.ToInputArray(),
                dtype.ToMatTypeNumber(Math.Max(src1Mat.Channels(), src2Mat.Channels())));
            return dst ?? new CvNDArray(dstMat);
        }

        /// <summary>
        /// Calculates per-element difference between two arrays or array and a scalar
        /// </summary>
        /// <param name="src1">The first source array</param>
        /// <param name="src2">The second source array. It must have the same size and same type as src1</param>
        /// <param name="dst">The destination array; it will have the same size and same type as src1</param>
        /// <param name="mask">The optional operation mask, 8-bit single channel array; specifies elements of the destination array to be changed. [By default this is null]</param>
        /// <param name="dtype"></param>
        public NDArray subtract(NDArray src1, Scalar src2, NDArray? dst = null, NDArray? mask = null,
            TF_DataType dtype = TF_DataType.DtInvalid)
        {
            Mat dstMat = dst is null ? new Mat() : dst.AsMat();
            var src1Mat = src1.AsMat();
            Cv2.Subtract(src1Mat, src2, dstMat, mask.ToInputArray(), dtype.ToMatTypeNumber(src1Mat.Channels()));
            return dst ?? new CvNDArray(dstMat);
        }

        /// <summary>
        /// Calculates per-element difference between two arrays or array and a scalar
        /// </summary>
        /// <param name="src1">The first source array</param>
        /// <param name="src2">The second source array. It must have the same size and same type as src1</param>
        /// <param name="dst">The destination array; it will have the same size and same type as src1</param>
        /// <param name="mask">The optional operation mask, 8-bit single channel array; specifies elements of the destination array to be changed. [By default this is null]</param>
        /// <param name="dtype"></param>
        public NDArray subtract(Scalar src1, NDArray src2, NDArray? dst = null, NDArray? mask = null,
            TF_DataType dtype = TF_DataType.DtInvalid)
        {
            Mat dstMat = dst is null ? new Mat() : dst.AsMat();
            var src2Mat = src2.AsMat();
            Cv2.Subtract(src1, src2Mat, dstMat, mask.ToInputArray(), dtype.ToMatTypeNumber(src2Mat.Channels()));
            return dst ?? new CvNDArray(dstMat);
        }

        /// <summary>
        /// Calculates the per-element scaled product of two arrays
        /// </summary>
        /// <param name="src1">The first source array</param>
        /// <param name="src2">The second source array of the same size and the same type as src1</param>
        /// <param name="dst">The destination array; will have the same size and the same type as src1</param>
        /// <param name="scale">The optional scale factor. [By default this is 1]</param>
        /// <param name="dtype"></param>
        public NDArray multiply(NDArray src1, NDArray src2, NDArray? dst = null, double scale = 1,
            TF_DataType dtype = TF_DataType.DtInvalid)
        {
            Mat dstMat = dst is null ? new Mat() : dst.AsMat();
            var src1Mat = src1.AsMat();
            var src2Mat = src2.AsMat();
            Cv2.Multiply(src1Mat, src2Mat, dstMat, scale,
                dtype.ToMatTypeNumber(Math.Max(src1Mat.Channels(), src2Mat.Channels())));
            return dst ?? new CvNDArray(dstMat);
        }

        /// <summary>
        /// Performs per-element division of two arrays or a scalar by an array.
        /// </summary>
        /// <param name="src1">The first source array</param>
        /// <param name="src2">The second source array; should have the same size and same type as src1</param>
        /// <param name="dst">The destination array; will have the same size and same type as src2</param>
        /// <param name="scale">Scale factor [By default this is 1]</param>
        /// <param name="dtype"></param>
        public NDArray divide(NDArray src1, NDArray src2, NDArray? dst = null, double scale = 1,
            TF_DataType dtype = TF_DataType.DtInvalid)
        {
            Mat dstMat = dst is null ? new Mat() : dst.AsMat();
            var src1Mat = src1.AsMat();
            var src2Mat = src2.AsMat();
            Cv2.Divide(src1Mat, src2Mat, dstMat, scale,
                dtype.ToMatTypeNumber(Math.Max(src1Mat.Channels(), src2Mat.Channels())));
            return dst ?? new CvNDArray(dstMat);
        }

        /// <summary>
        /// Performs per-element division of two arrays or a scalar by an array.
        /// </summary>
        /// <param name="scale">Scale factor</param>
        /// <param name="src2">The first source array</param>
        /// <param name="dst">The destination array; will have the same size and same type as src2</param>
        /// <param name="dtype"></param>
        public NDArray divide(double scale, NDArray src2, NDArray? dst = null,
            TF_DataType dtype = TF_DataType.DtInvalid)
        {
            Mat dstMat = dst is null ? new Mat() : dst.AsMat();
            var src2Mat = src2.AsMat();
            Cv2.Divide(scale, src2Mat, dstMat, dtype.ToMatTypeNumber(src2Mat.Channels()));
            return dst ?? new CvNDArray(dstMat);
        }

        /// <summary>
        /// adds scaled array to another one (dst = alpha*src1 + src2)
        /// </summary>
        /// <param name="src1"></param>
        /// <param name="alpha"></param>
        /// <param name="src2"></param>
        /// <param name="dst"></param>
        public NDArray scaleAdd(NDArray src1, double alpha, NDArray src2, NDArray? dst = null)
        {
            Mat dstMat = dst is null ? new Mat() : dst.AsMat();
            var src1Mat = src1.AsMat();
            var src2Mat = src2.AsMat();
            Cv2.ScaleAdd(src1Mat, alpha, src2Mat, dstMat);
            return dst ?? new CvNDArray(dstMat);
        }

        /// <summary>
        /// computes weighted sum of two arrays (dst = alpha*src1 + beta*src2 + gamma)
        /// </summary>
        /// <param name="src1"></param>
        /// <param name="alpha"></param>
        /// <param name="src2"></param>
        /// <param name="beta"></param>
        /// <param name="gamma"></param>
        /// <param name="dst"></param>
        /// <param name="dtype"></param>
        public NDArray addWeighted(NDArray src1, double alpha, NDArray src2, double beta, 
            double gamma, NDArray? dst = null, TF_DataType dtype = TF_DataType.DtInvalid)
        {
            Mat dstMat = dst is null ? new Mat() : dst.AsMat();
            var src1Mat = src1.AsMat();
            var src2Mat = src2.AsMat();
            Cv2.AddWeighted(src1Mat, alpha, src2Mat, beta, gamma, dstMat,
                dtype.ToMatTypeNumber(Math.Max(src1Mat.Channels(), src2Mat.Channels())));
            return dst ?? new CvNDArray(dstMat);
        }

        /// <summary>
        /// Scales, computes absolute values and converts the result to 8-bit.
        /// </summary>
        /// <param name="src">The source array</param>
        /// <param name="dst">The destination array</param>
        /// <param name="alpha">The optional scale factor. [By default this is 1]</param>
        /// <param name="beta">The optional delta added to the scaled values. [By default this is 0]</param>
        public NDArray convertScaleAbs(NDArray src, NDArray? dst = null, double alpha = 1, 
            double beta = 0)
        {
            Mat dstMat = dst is null ? new Mat() : dst.AsMat();
            var srcMat = src.AsMat();
            Cv2.ConvertScaleAbs(srcMat, dstMat, alpha, beta);
            return dst ?? new CvNDArray(dstMat);
        }

        /// <summary>
        /// Converts an array to half precision floating number.
        ///
        /// This function converts FP32(single precision floating point) from/to FP16(half precision floating point). CV_16S format is used to represent FP16 data.
        /// There are two use modes(src -&gt; dst) : CV_32F -&gt; CV_16S and CV_16S -&gt; CV_32F.The input array has to have type of CV_32F or
        /// CV_16S to represent the bit depth.If the input array is neither of them, the function will raise an error.
        /// The format of half precision floating point is defined in IEEE 754-2008.
        /// </summary>
        /// <param name="src">input array.</param>
        /// <param name="dst">output array.</param>
        public NDArray convertFp16(NDArray src, NDArray? dst = null)
        {
            Mat dstMat = dst is null ? new Mat() : dst.AsMat();
            var srcMat = src.AsMat();
            Cv2.ConvertFp16(srcMat, dstMat);
            return dst ?? new CvNDArray(dstMat);
        }

        /// <summary>
        /// transforms array of numbers using a lookup table: dst(i)=lut(src(i))
        /// </summary>
        /// <param name="src">Source array of 8-bit elements</param>
        /// <param name="lut">Look-up table of 256 elements. 
        /// In the case of multi-channel source array, the table should either have 
        /// a single channel (in this case the same table is used for all channels)
        ///  or the same number of channels as in the source array</param>
        /// <param name="dst">Destination array; 
        /// will have the same size and the same number of channels as src, 
        /// and the same depth as lut</param>
        public NDArray LUT(NDArray src, NDArray lut, NDArray? dst = null)
        {
            Mat dstMat = dst is null ? new Mat() : dst.AsMat();
            var srcMat = src.AsMat();
            var lutMat = lut.AsMat();
            Cv2.LUT(srcMat, lutMat, dstMat);
            return dst ?? new CvNDArray(dstMat);
        }

        /// <summary>
        /// transforms array of numbers using a lookup table: dst(i)=lut(src(i))
        /// </summary>
        /// <param name="src">Source array of 8-bit elements</param>
        /// <param name="lut">Look-up table of 256 elements. 
        /// In the case of multi-channel source array, the table should either have 
        /// a single channel (in this case the same table is used for all channels) 
        /// or the same number of channels as in the source array</param>
        /// <param name="dst">Destination array; 
        /// will have the same size and the same number of channels as src, 
        /// and the same depth as lut</param>
        public NDArray LUT(NDArray src, byte[] lut, NDArray? dst = null)
        {
            Mat dstMat = dst is null ? new Mat() : dst.AsMat();
            var srcMat = src.AsMat();
            Cv2.LUT(srcMat, lut, dstMat);
            return dst ?? new CvNDArray(dstMat);
        }

        /// <summary>
        /// computes sum of array elements
        /// </summary>
        /// <param name="src">The source array; must have 1 to 4 channels</param>
        /// <returns></returns>
        public Scalar sum(NDArray src)
        {
            return Cv2.Sum(src.AsMat());
        }

        /// <summary>
        /// computes the number of nonzero array elements
        /// </summary>
        /// <param name="mtx">Single-channel array</param>
        /// <returns>number of non-zero elements in mtx</returns>
        public int countNonZero(NDArray mtx)
        {
            return Cv2.CountNonZero(mtx.AsMat());
        }

        /// <summary>
        /// returns the list of locations of non-zero pixels
        /// </summary>
        /// <param name="src"></param>
        /// <param name="idx"></param>
        public NDArray findNonZero(NDArray src, NDArray? idx = null)
        {
            Mat dstMat = idx is null ? new Mat() : idx.AsMat();
            var srcMat = src.AsMat();
            Cv2.FindNonZero(srcMat, dstMat);
            return idx ?? new CvNDArray(dstMat);
        }

        /// <summary>
        /// computes mean value of selected array elements
        /// </summary>
        /// <param name="src">The source array; it should have 1 to 4 channels
        ///  (so that the result can be stored in Scalar)</param>
        /// <param name="mask">The optional operation mask</param>
        /// <returns></returns>
        public Scalar mean(NDArray src, NDArray? mask = null)
        {
            var srcMat = src.AsMat();
            return Cv2.Mean(srcMat, mask.ToInputArray());
        }

        /// <summary>
        /// computes mean value and standard deviation of all or selected array elements
        /// </summary>
        /// <param name="src">The source array; it should have 1 to 4 channels 
        /// (so that the results can be stored in Scalar's)</param>
        /// <param name="mean">The output parameter: computed mean value</param>
        /// <param name="stddev">The output parameter: computed standard deviation</param>
        /// <param name="mask">The optional operation mask</param>
        public (NDArray, NDArray) meanStdDev(NDArray src, NDArray? mean = null, 
            NDArray? stddev = null, NDArray? mask = null)
        {
            var srcMat = src.AsMat();
            var meanMat = mean?.AsMat();
            var stddevMat = stddev?.AsMat();
            Cv2.MeanStdDev(srcMat, meanMat, stddevMat, mask.ToInputArray());
            var meanRes = mean ?? new CvNDArray(meanMat);
            var stddevRes = stddev ?? new CvNDArray(stddevMat);
            return (meanRes, stddevRes);
        }

        /// <summary>
        /// Calculates absolute array norm, absolute difference norm, or relative difference norm.
        /// </summary>
        /// <param name="src1">The first source array</param>
        /// <param name="normType">Type of the norm</param>
        /// <param name="mask">The optional operation mask</param>
        public double norm(NDArray src1, NormTypes normType = NormTypes.L2, NDArray? mask = null)
        {
            var src1Mat = src1.AsMat();
            return Cv2.Norm(src1Mat, normType, mask.ToInputArray());
        }

        /// <summary>
        /// computes norm of selected part of the difference between two arrays
        /// </summary>
        /// <param name="src1">The first source array</param>
        /// <param name="src2">The second source array of the same size and the same type as src1</param>
        /// <param name="normType">Type of the norm</param>
        /// <param name="mask">The optional operation mask</param>
        public double norm(NDArray src1, NDArray src2, NormTypes normType = NormTypes.L2,
            NDArray? mask = null)
        {
            var src1Mat = src1.AsMat();
            var src2Mat = src2.AsMat();
            return Cv2.Norm(src1Mat, src2Mat, normType, mask.ToInputArray());
        }

        /// <summary>
        /// Computes the Peak Signal-to-Noise Ratio (PSNR) image quality metric.
        /// 
        /// This function calculates the Peak Signal-to-Noise Ratio(PSNR) image quality metric in decibels(dB), 
        /// between two input arrays src1 and src2.The arrays must have the same type.
        /// </summary>
        /// <param name="src1">first input array.</param>
        /// <param name="src2">second input array of the same size as src1.</param>
        /// <param name="r">the maximum pixel value (255 by default)</param>
        /// <returns></returns>
        public double PSNR(NDArray src1, NDArray src2, double r = 255.0)
        {
            var src1Mat = src1.AsMat();
            var src2Mat = src2.AsMat();
            return Cv2.PSNR(src1Mat, src2Mat);
        }

        /// <summary>
        /// naive nearest neighbor finder
        /// </summary>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <param name="dtype"></param>
        /// <param name="normType"></param>
        /// <param name="k"></param>
        /// <param name="mask"></param>
        /// <param name="update"></param>
        /// <param name="crosscheck"></param>
        /// <returns>dist and nidx</returns>
        public (NDArray, NDArray) batchDistance(NDArray src1, NDArray src2,
            // ReSharper disable once IdentifierTypo
            int dtype, NormTypes normType = NormTypes.L2,
            int k = 0, NDArray? mask = null,
            int update = 0, bool crosscheck = false)
        {
            var src1Mat = src1.AsMat();
            var src2Mat = src2.AsMat();
            Mat dist = new();
            Mat nidx = new();
            Cv2.BatchDistance(src1Mat, src2Mat, dist, dtype, nidx, normType,
                k, mask.ToInputArray(), update, crosscheck);
            return (dist.numpy(), nidx.numpy());
        }

        /// <summary>
        /// scales and shifts array elements so that either the specified norm (alpha) 
        /// or the minimum (alpha) and maximum (beta) array values get the specified values
        /// </summary>
        /// <param name="src">The source array</param>
        /// <param name="dst">The destination array; will have the same size as src</param>
        /// <param name="alpha">The norm value to normalize to or the lower range boundary 
        /// in the case of range normalization</param>
        /// <param name="beta">The upper range boundary in the case of range normalization; 
        /// not used for norm normalization</param>
        /// <param name="normType">The normalization type</param>
        /// <param name="dtype">When the parameter is negative, 
        /// the destination array will have the same type as src, 
        /// otherwise it will have the same number of channels as src and the depth =CV_MAT_DEPTH(rtype)</param>
        /// <param name="mask">The optional operation mask</param>
        /// <returns>dst</returns>
        public NDArray normalize(NDArray src, NDArray? dst = null, double alpha = 1, double beta = 0,
            NormTypes normType = NormTypes.L2, int dtype = -1, InputArray? mask = null)
        {
            if (!dst.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "normalize needs that. Please consider change the adapter mode.");
            }
            var srcMat = src.AsMat();
            Mat dstMat = dst is null ? new Mat() : dst.AsMat();
            Cv2.Normalize(srcMat, dstMat, alpha, beta, normType, dtype, mask);
            if(dst is null)
            {
                dst = dstMat.numpy();
            }
            return dst;
        }

        /// <summary>
        /// Finds indices of max elements along provided axis
        /// </summary>
        /// <param name="src">Input single-channel array</param>
        /// <param name="axis">Axis to reduce along</param>
        /// <param name="lastIndex">Whether to get the index of first or last occurrence of max</param>
        /// <returns>Output array of type CV_32SC1 with the same dimensionality as src,
        /// except for axis being reduced - it should be set to 1.</returns>
        public NDArray reduceArgMax(NDArray src, int axis, bool lastIndex = false)
        {
            var srcMat = src.AsMat();
            Mat dstMat = new Mat();
            Cv2.ReduceArgMax(srcMat, dstMat, axis, lastIndex);
            return dstMat.numpy();
        }

        /// <summary>
        /// Finds indices of min elements along provided axis
        /// </summary>
        /// <param name="src">Input single-channel array</param>
        /// <param name="axis">Axis to reduce along</param>
        /// <param name="lastIndex">Whether to get the index of first or last occurrence of max</param>
        /// <returns>Output array of type CV_32SC1 with the same dimensionality as src,
        /// except for axis being reduced - it should be set to 1.</returns>
        public NDArray reduceArgMin(NDArray src, int axis, bool lastIndex = false)
        {
            var srcMat = src.AsMat();
            Mat dstMat = new Mat();
            Cv2.ReduceArgMin(srcMat, dstMat, axis, lastIndex);
            return dstMat.numpy();
        }

        /// <summary>
        /// finds global minimum and maximum array elements and returns their values and their locations
        /// </summary>
        /// <param name="src">The source single-channel array</param>
        /// <returns>Pointer to returned minimum value and maximum value.</returns>
        public (double, double) minMaxLoc(NDArray src)
        {
            Cv2.MinMaxLoc(src.AsMat(), out double minVal, out double maxVal);
            return (minVal, maxVal);
        }

        /// <summary>
        /// finds global minimum and maximum array elements and returns their values and their locations
        /// </summary>
        /// <param name="src">The source single-channel array</param>
        /// <param name="mask">The optional mask used to select a sub-array</param>
        /// <returns>Pointer to returned minimum value, maximum value, minimum location, maximum location.</returns>
        public (double, double, Point, Point) minMaxLoc(NDArray src, NDArray? mask = null)
        {
            Cv2.MinMaxLoc(src.AsMat(), out double minVal, out double maxVal, out Point minLoc, out Point maxLoc, mask.ToInputArray());
            return (minVal, maxVal, minLoc, maxLoc);
        }

        /// <summary>
        /// finds global minimum and maximum array elements and returns their values and their locations
        /// </summary>
        /// <param name="src">The source single-channel array</param>
        /// <returns>Pointer to returned minimum value and maximum value.</returns>
        public (double, double) minMaxIdx(NDArray src)
        {
            Cv2.MinMaxIdx(src.AsMat(), out double minVal, out double maxVal);
            return (minVal, maxVal);
        }

        /// <summary>
        /// finds global minimum and maximum array elements and returns their values and their locations
        /// </summary>
        /// <param name="src">The source single-channel array</param>
        /// <param name="mask"></param>
        /// <returns>Pointer to returned minimum value, maximum value, minimum idx, maximum idx.</returns>
        public (double, double, int[], int[]) minMaxIdx(NDArray src, NDArray? mask = null)
        {
            int[] minIdx = new int[src.AsMat().Dims];
            int[] maxIdx = new int[src.AsMat().Dims];
            Cv2.MinMaxIdx(src.AsMat(), out double minVal, out double maxVal,minIdx, maxIdx, mask.ToInputArray());
            return (minVal, maxVal, minIdx, maxIdx);
        }

        /// <summary>
        /// transforms 2D matrix to 1D row or column vector by taking sum, minimum, maximum or mean value over all the rows
        /// </summary>
        /// <param name="src">The source 2D matrix</param>
        /// <param name="dim">The dimension index along which the matrix is reduced. 
        /// 0 means that the matrix is reduced to a single row and 1 means that the matrix is reduced to a single column</param>
        /// <param name="rtype"></param>
        /// <param name="dtype">When it is negative, the destination vector will have 
        /// the same type as the source matrix, otherwise, its type will be CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), mtx.channels())</param>
        /// <returns>The destination vector. 
        public NDArray reduce(NDArray src, ReduceDimension dim, ReduceTypes rtype, TF_DataType dtype)
        {
            var srcMat = src.AsMat();
            Mat dstMat = new();
            Cv2.Reduce(srcMat, dstMat, dim, rtype, dtype.ToMatTypeNumber(srcMat.Channels()));
            return dstMat.numpy();
        }

        /// <summary>
        /// makes multi-channel array out of several single-channel arrays
        /// </summary>
        /// <param name="mv"></param>
        /// <returns></returns>
        public NDArray merge(IEnumerable<NDArray> mv)
        {
            Mat dstMat = new();
            Cv2.Merge(mv.Select(x => x.AsMat()).ToArray(), dstMat);
            return dstMat.numpy();
        }

        /// <summary>
        /// Copies each plane of a multi-channel array to a dedicated array
        /// </summary>
        /// <param name="src">The source multi-channel array</param>
        /// <returns>The number of arrays must match mtx.channels() . 
        /// The arrays themselves will be reallocated if needed</returns>
        public NDArray[] split(NDArray src)
        {
            return Cv2.Split(src.AsMat()).Select(x => x.numpy()).ToArray();
        }

        /// <summary>
        /// extracts a single channel from src (coi is 0-based index)
        /// </summary>
        /// <param name="src"></param>
        /// <param name="coi"></param>
        /// <returns></returns>
        public NDArray extractChannel(NDArray src, int coi)
        {
            Mat dstMat = new();
            Cv2.ExtractChannel(src.AsMat(), dstMat, coi);
            return dstMat.numpy();
        }

        /// <summary>
        /// inserts a single channel to dst (coi is 0-based index)
        /// </summary>
        /// <param name="src"></param>
        /// <param name="coi"></param>
        /// <returns></returns>
        public NDArray insertChannel(NDArray src, int coi)
        {
            Mat dstMat = new();
            Cv2.InsertChannel(src.AsMat(), dstMat, coi);
            return dstMat.numpy();
        }

        /// <summary>
        /// reverses the order of the rows, columns or both in a matrix
        /// </summary>
        /// <param name="src">The source array</param>
        /// <param name="flipCode">Specifies how to flip the array: 
        /// 0 means flipping around the x-axis, positive (e.g., 1) means flipping around y-axis, 
        /// and negative (e.g., -1) means flipping around both axes. See also the discussion below for the formulas.</param>
        /// <returns>The destination array; will have the same size and same type as src</returns>
        public NDArray flip(NDArray src, FlipMode flipCode)
        {
            Mat dstMat = new();
            Cv2.Flip(src.AsMat(), dstMat, flipCode);
            return dstMat.numpy();
        }

        /// <summary>
        /// Rotates a 2D array in multiples of 90 degrees.
        /// </summary>
        /// <param name="src">input array.</param>
        /// <param name="rotateCode">an enum to specify how to rotate the array.</param>
        /// <returns>output array of the same type as src.
        /// The size is the same with ROTATE_180, and the rows and cols are switched for
        /// ROTATE_90_CLOCKWISE and ROTATE_90_COUNTERCLOCKWISE.</returns>
        public NDArray rotate(NDArray src, RotateFlags rotateCode)
        {
            Mat dstMat = new();
            Cv2.Rotate(src.AsMat(), dstMat, rotateCode);
            return dstMat.numpy();
        }

        /// <summary>
        /// replicates the input matrix the specified number of times in the horizontal and/or vertical direction
        /// </summary>
        /// <param name="src">The source array to replicate</param>
        /// <param name="ny">How many times the src is repeated along the vertical axis</param>
        /// <param name="nx">How many times the src is repeated along the horizontal axis</param>
        /// <returns>The destination array; will have the same type as src</returns>
        public NDArray repeat(NDArray src, int ny, int nx)
        {
            Mat dstMat = new();
            Cv2.Repeat(src.AsMat(), ny, nx);
            return dstMat.numpy();
        }

        /// <summary>
        /// Applies horizontal concatenation to given matrices.
        /// </summary>
        /// <param name="src">input array or vector of matrices. all of the matrices must have the same number of rows and the same depth.</param>
        /// <returns>output array. It has the same number of rows and depth as the src, and the sum of cols of the src.</returns>
        public NDArray hconcat(IEnumerable<NDArray> src)
        {
            Mat dstMat = new();
            Cv2.HConcat(src.Select(x => x.AsMat()), dstMat);
            return dstMat.numpy();
        }

        /// <summary>
        /// Applies horizontal concatenation to given matrices.
        /// </summary>
        /// <param name="src1">first input array to be considered for horizontal concatenation.</param>
        /// <param name="src2">second input array to be considered for horizontal concatenation.</param>
        /// <returns>output array. It has the same number of rows and depth as the src1 and src2, and the sum of cols of the src1 and src2.</returns>
        public NDArray hconcat(NDArray src1, NDArray src2)
        {
            Mat dstMat = new();
            Cv2.HConcat(src1.AsMat(), src2.AsMat(), dstMat);
            return dstMat.numpy();
        }

        /// <summary>
        /// Applies vertical concatenation to given matrices.
        /// </summary>
        /// <param name="src">input array or vector of matrices. all of the matrices must have the same number of cols and the same depth.</param>
        /// <returns>output array. It has the same number of cols and depth as the src, and the sum of rows of the src.</returns>
        public NDArray vconcat(IEnumerable<NDArray> src)
        {
            Mat dstMat = new();
            Cv2.VConcat(src.Select(x => x.AsMat()), dstMat);
            return dstMat.numpy();
        }

        /// <summary>
        /// Applies vertical concatenation to given matrices.
        /// </summary>
        /// <param name="src1">first input array to be considered for vertical concatenation.</param>
        /// <param name="src2">second input array to be considered for vertical concatenation.</param>
        /// <returns>output array. It has the same number of cols and depth as the src1 and src2, and the sum of rows of the src1 and src2.</returns>
        public NDArray vconcat(NDArray src1, NDArray src2)
        {
            Mat dstMat = new();
            Cv2.VConcat(src1.AsMat(), src2.AsMat(), dstMat);
            return dstMat.numpy();
        }

        /// <summary>
        /// computes bitwise conjunction of the two arrays (dst = src1 &amp; src2)
        /// </summary>
        /// <param name="src1">first input array or a scalar.</param>
        /// <param name="src2">second input array or a scalar.</param>
        /// <param name="mask">optional operation mask, 8-bit single channel array, that specifies elements of the output array to be changed.</param>
        /// <returns>output array that has the same size and type as the input</returns>
        public NDArray bitwise_and(NDArray src1, NDArray src2, NDArray? mask = null)
        {
            Mat dstMat = new();
            Cv2.BitwiseAnd(src1.AsMat(), src2.AsMat(), dstMat, mask.ToInputArray());
            return dstMat.numpy();
        }

        /// <summary>
        /// computes bitwise conjunction of the two arrays (dst = src1 | src2)
        /// </summary>
        /// <param name="src1">first input array or a scalar.</param>
        /// <param name="src2">second input array or a scalar.</param>
        /// <param name="mask">optional operation mask, 8-bit single channel array, that specifies elements of the output array to be changed.</param>
        /// <returns>output array that has the same size and type as the input</returns>
        public NDArray bitwise_or(NDArray src1, NDArray src2, NDArray? mask = null)
        {
            Mat dstMat = new();
            Cv2.BitwiseOr(src1.AsMat(), src2.AsMat(), dstMat, mask.ToInputArray());
            return dstMat.numpy();
        }

        /// <summary>
        /// computes bitwise conjunction of the two arrays (dst = src1 ^ src2)
        /// </summary>
        /// <param name="src1">first input array or a scalar.</param>
        /// <param name="src2">second input array or a scalar.</param>
        /// <param name="mask">optional operation mask, 8-bit single channel array, that specifies elements of the output array to be changed.</param>
        /// <returns>output array that has the same size and type as the input</returns>
        public NDArray bitwise_xor(NDArray src1, NDArray src2, NDArray? mask = null)
        {
            Mat dstMat = new();
            Cv2.BitwiseXor(src1.AsMat(), src2.AsMat(), dstMat, mask.ToInputArray());
            return dstMat.numpy();
        }

        /// <summary>
        /// inverts each bit of array (dst = ~src)
        /// </summary>
        /// <param name="src">input array.</param>
        /// <param name="mask">optional operation mask, 8-bit single channel array, that specifies elements of the output array to be changed.</param>
        /// <returns>output array that has the same size and type as the input</returns>
        public NDArray bitwise_not(NDArray src, NDArray? mask = null)
        {
            Mat dstMat = new();
            Cv2.BitwiseNot(src.AsMat(), dstMat, mask.ToInputArray());
            return dstMat.numpy();
        }

        /// <summary>
        /// Calculates the per-element absolute difference between two arrays or between an array and a scalar.
        /// </summary>
        /// <param name="src1">first input array or a scalar.</param>
        /// <param name="src2">second input array or a scalar.</param>
        /// <returns>output array that has the same size and type as input arrays.</returns>
        public NDArray absdiff(NDArray src1, NDArray src2)
        {
            Mat dstMat = new();
            Cv2.Absdiff(src1.AsMat(), src2.AsMat(), dstMat);
            return dstMat.numpy();
        }

        /// <summary>
        /// Copies the matrix to another one.
        /// When the operation mask is specified, if the Mat::create call shown above reallocates the matrix, the newly allocated matrix is initialized with all zeros before copying the data.
        /// </summary>
        /// <param name="src">Source matrix.</param>
        /// <param name="dst">Destination matrix. If it does not have a proper size or type before the operation, it is reallocated.</param>
        /// <param name="mask">Operation mask of the same size as \*this. Its non-zero elements indicate which matrix
        /// elements need to be copied.The mask has to be of type CV_8U and can have 1 or multiple channels.</param>
        /// <returns>dst</returns>
        public NDArray copyTo(NDArray src, NDArray? mask = null, NDArray? dst = null)
        {
            Mat dstMat = dst is null ? new Mat() : dst.AsMat();
            Cv2.CopyTo(src.AsMat(), dstMat, mask.ToInputArray());
            if(dst is null)
            {
                dst = dstMat.numpy();
            }
            return dst;
        }

        /// <summary>
        /// Checks if array elements lie between the elements of two other arrays.
        /// </summary>
        /// <param name="src">first input array.</param>
        /// <param name="lowerb">inclusive lower boundary array or a scalar.</param>
        /// <param name="upperb">inclusive upper boundary array or a scalar.</param>
        /// <returns>output array of the same size as src and CV_8U type.</returns>
        public NDArray inRange(NDArray src, NDArray lowerb, NDArray upperb)
        {
            Mat dstMat = new();
            Cv2.InRange(src.AsMat(), lowerb.AsMat(), upperb.AsMat(), dstMat);
            return dstMat.numpy();
        }

        /// <summary>
        /// Checks if array elements lie between the elements of two other arrays.
        /// </summary>
        /// <param name="src">first input array.</param>
        /// <param name="lowerb">inclusive lower boundary array or a scalar.</param>
        /// <param name="upperb">inclusive upper boundary array or a scalar.</param>
        /// <returns>output array of the same size as src and CV_8U type.</returns>
        public NDArray inRange(NDArray src, Scalar lowerb, Scalar upperb)
        {
            Mat dstMat = new();
            Cv2.InRange(src.AsMat(), lowerb, upperb, dstMat);
            return dstMat.numpy();
        }

        /// <summary>
        /// Performs the per-element comparison of two arrays or an array and scalar value.
        /// </summary>
        /// <param name="src1">first input array or a scalar; when it is an array, it must have a single channel.</param>
        /// <param name="src2">second input array or a scalar; when it is an array, it must have a single channel.</param>
        /// <param name="cmpop">a flag, that specifies correspondence between the arrays (cv::CmpTypes)</param>
        /// <returns>output array of type ref CV_8U that has the same size and the same number of channels as the input arrays.</returns>
        public NDArray compare(NDArray src1, NDArray src2, CmpType cmpop)
        {
            Mat dstMat = new();
            Cv2.Compare(src1.AsMat(), src2.AsMat(), dstMat, cmpop);
            return dstMat.numpy();
        }

        public NDArray min(NDArray src1, NDArray src2)
        {
            Mat dstMat = new();
            Cv2.Min(src1.AsMat(), src2.AsMat(), dstMat);
            return dstMat.numpy();
        }

        public NDArray min(NDArray src1, double src2)
        {
            Mat dstMat = new();
            Cv2.Min(src1.AsMat(), src2, dstMat);
            return dstMat.numpy();
        }

        public NDArray max(NDArray src1, NDArray src2)
        {
            Mat dstMat = new();
            Cv2.Max(src1.AsMat(), src2.AsMat(), dstMat);
            return dstMat.numpy();
        }

        public NDArray max(NDArray src1, double src2)
        {
            Mat dstMat = new();
            Cv2.Max(src1.AsMat(), src2, dstMat);
            return dstMat.numpy();
        }

        public NDArray sqrt(NDArray src)
        {
            Mat dstMat = new();
            Cv2.Sqrt(src.AsMat(), dstMat);
            return dstMat.numpy();
        }

        public NDArray pow(NDArray src, double power)
        {
            Mat dstMat = new();
            Cv2.Pow(src.AsMat(), power, dstMat);
            return dstMat.numpy();
        }

        public NDArray exp(NDArray src)
        {
            Mat dstMat = new();
            Cv2.Exp(src.AsMat(), dstMat);
            return dstMat.numpy();
        }

        public NDArray log(NDArray src)
        {
            Mat dstMat = new();
            Cv2.Log(src.AsMat(), dstMat);
            return dstMat.numpy();
        }

        /// <summary>
        /// Calculates x and y coordinates of 2D vectors from their magnitude and angle.
        /// </summary>
        /// <param name="magnitude">input floating-point array of magnitudes of 2D vectors;
        /// it can be an empty matrix(=Mat()), in this case, the function assumes that all the magnitudes are = 1; if it is not empty,
        /// it must have the same size and type as angle.</param>
        /// <param name="angle">input floating-point array of angles of 2D vectors.</param>
        /// <param name="angleInDegrees">when true, the input angles are measured in degrees, otherwise, they are measured in radians.</param>
        /// <returns>output arrays of x-coordinates and y-coordinates of 2D vectors</returns>
        public (NDArray, NDArray) polarToCart(NDArray magnitude, NDArray angle, bool angleInDegrees = false)
        {
            Mat xMat = new();
            Mat yMat = new();
            Cv2.PolarToCart(magnitude.AsMat(), angle.AsMat(), xMat, yMat, angleInDegrees);
            return (xMat.numpy(), yMat.numpy());
        }

        /// <summary>
        /// Calculates the magnitude and angle of 2D vectors.
        /// </summary>
        /// <param name="x">array of x-coordinates; this must be a single-precision or double-precision floating-point array.</param>
        /// <param name="y">array of y-coordinates, that must have the same size and same type as x.</param>
        /// the angles are measured in radians(from 0 to 2\*Pi) or in degrees(0 to 360 degrees).</param>
        /// <param name="angleInDegrees">a flag, indicating whether the angles are measured in radians(which is by default), or in degrees.</param>
        /// <returns>output arrays of magnitudes and angles of the same size and type as x.</returns>
        public (NDArray, NDArray) cartToPolar(NDArray x, NDArray y, bool angleInDegrees = false)
        {
            Mat magnitudeMat = new();
            Mat angleMat = new();
            Cv2.CartToPolar(x.AsMat(), y.AsMat(), magnitudeMat, angleMat, angleInDegrees);
            return (magnitudeMat.numpy(), angleMat.numpy());
        }

        /// <summary>
        /// Calculates the rotation angle of 2D vectors.
        /// </summary>
        /// <param name="x">input floating-point array of x-coordinates of 2D vectors.</param>
        /// <param name="y">input array of y-coordinates of 2D vectors; it must have the same size and the same type as x.</param>
        /// <param name="angleInDegrees">when true, the function calculates the angle in degrees, otherwise, they are measured in radians.</param>
        /// <returns>output array of vector angles; it has the same size and same type as x.</returns>
        public NDArray phase(NDArray x, NDArray y, bool angleInDegrees = false)
        {
            Mat angleMat = new();
            Cv2.Phase(x.AsMat(), y.AsMat(), angleMat, angleInDegrees);
            return angleMat.numpy();
        }

        /// <summary>
        /// Calculates the magnitude of 2D vectors.
        /// </summary>
        /// <param name="x">floating-point array of x-coordinates of the vectors.</param>
        /// <param name="y">floating-point array of y-coordinates of the vectors; it must have the same size as x.</param>
        /// <returns>output array of the same size and type as x.</returns>
        public NDArray magnitude(NDArray x, NDArray y)
        {
            Mat magnitudeMat = new();
            Cv2.Magnitude(x.AsMat(), y.AsMat(), magnitudeMat);
            return magnitudeMat.numpy();
        }

        /// <summary>
        /// checks that each matrix element is within the specified range.
        /// </summary>
        /// <param name="src">The array to check</param>
        /// <param name="quiet">The flag indicating whether the functions quietly 
        /// return false when the array elements are out of range, 
        /// or they throw an exception.</param>
        /// <returns></returns>
        public bool checkRange(NDArray src, bool quiet = true)
        {
            return Cv2.CheckRange(src.AsMat(), quiet);
        }

        /// <summary>
        /// checks that each matrix element is within the specified range.
        /// </summary>
        /// <param name="src">The array to check</param>
        /// <param name="quiet">The flag indicating whether the functions quietly 
        /// return false when the array elements are out of range, 
        /// or they throw an exception.</param>
        /// <param name="minVal">The inclusive lower boundary of valid values range</param>
        /// <param name="maxVal">The exclusive upper boundary of valid values range</param>
        /// <returns>The optional output parameter, where the position of 
        /// the first outlier is stored.</returns>
        /// <returns></returns>
        public (bool, Point) checkRange(NDArray src, bool quiet, 
            double minVal = double.MinValue, double maxVal = double.MaxValue)
        {
            bool retVal = Cv2.CheckRange(src.AsMat(), quiet, out var pos, minVal, maxVal);
            return (retVal, pos);
        }

        /// <summary>
        /// converts NaN's to the given number
        /// </summary>
        /// <param name="a"></param>
        /// <param name="val"></param>
        /// <returns></returns>
        public NDArray patchNaNs(NDArray a, double val = 0)
        {
            if (!a.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "patchNaNs needs that. Please consider change the adapter mode.");
            }
            Cv2.PatchNaNs(a.AsMat(), val);
            return a;
        }

        /// <summary>
        /// implements generalized matrix product algorithm GEMM from BLAS
        /// </summary>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <param name="alpha"></param>
        /// <param name="src3"></param>
        /// <param name="gamma"></param>
        /// <param name="flags"></param>
        /// <returns></returns>
        public NDArray gemm(NDArray src1, NDArray src2, double alpha, NDArray src3, 
            double gamma, GemmFlags flags = GemmFlags.None)
        {
            Mat dstMat = new();
            Cv2.Gemm(src1.AsMat(), src2.AsMat(), alpha, src3.AsMat(), gamma, dstMat, flags);
            return dstMat.numpy();
        }

        /// <summary>
        /// multiplies matrix by its transposition from the left or from the right
        /// </summary>
        /// <param name="src">The source matrix</param>
        /// <param name="aTa">Specifies the multiplication ordering; see the description below</param>
        /// <param name="delta">The optional delta matrix, subtracted from src before the 
        /// multiplication. When the matrix is empty ( delta=Mat() ), it’s assumed to be 
        /// zero, i.e. nothing is subtracted, otherwise if it has the same size as src, 
        /// then it’s simply subtracted, otherwise it is "repeated" to cover the full src 
        /// and then subtracted. Type of the delta matrix, when it's not empty, must be the 
        /// same as the type of created destination matrix, see the rtype description</param>
        /// <param name="scale">The optional scale factor for the matrix product</param>
        /// <param name="dtype">When it’s negative, the destination matrix will have the 
        /// same type as src . Otherwise, it will have type=CV_MAT_DEPTH(rtype), 
        /// which should be either CV_32F or CV_64F</param>
        /// <returns>The destination square matrix</returns>
        public NDArray mulTransposed(NDArray src, bool aTa, NDArray? delta = null, double scale = 1, 
            TF_DataType dtype = TF_DataType.DtInvalid)
        {
            Mat dstMat = new();
            var srcMat = src.AsMat();
            Cv2.MulTransposed(srcMat, dstMat, aTa, delta.ToInputArray(), scale, dtype.ToMatTypeNumber(srcMat.Channels()));
            return dstMat.numpy();
        }

        /// <summary>
        /// transposes the matrix
        /// </summary>
        /// <param name="src">The source array</param>
        /// <returns>The destination array of the same type as src</returns>
        public NDArray transpose(NDArray src)
        {
            Mat dstMat = new();
            Cv2.Transpose(src.AsMat(), dstMat);
            return dstMat.numpy();
        }

        /// <summary>
        /// performs affine transformation of each element of multi-channel input matrix
        /// </summary>
        /// <param name="src">The source array; must have as many channels (1 to 4) as mtx.cols or mtx.cols-1</param>
        /// <param name="m">The transformation matrix</param>
        /// <returns>The destination array; will have the same size and depth as src and as many channels as mtx.rows</returns>
        public NDArray transform(NDArray src, NDArray m)
        {
            Mat dstMat = new();
            Cv2.Transform(src.AsMat(), dstMat, m.AsMat());
            return dstMat.numpy();
        }

        /// <summary>
        /// performs perspective transformation of each element of multi-channel input matrix
        /// </summary>
        /// <param name="src">The source two-channel or three-channel floating-point array; 
        /// each element is 2D/3D vector to be transformed</param>
        /// <param name="m">3x3 or 4x4 transformation matrix</param>
        /// <returns>The destination array; it will have the same size and same type as src</returns>
        public NDArray perspectiveTransform(NDArray src, NDArray m)
        {
            Mat dstMat = new();
            Cv2.PerspectiveTransform(src.AsMat(), dstMat, m.AsMat());
            return dstMat.numpy();
        }

        /// <summary>
        /// performs perspective transformation of each element of multi-channel input matrix
        /// </summary>
        /// <param name="src">The source two-channel or three-channel floating-point array; 
        /// each element is 2D/3D vector to be transformed</param>
        /// <param name="m">3x3 or 4x4 transformation matrix</param>
        /// <returns>The destination array; it will have the same size and same type as src</returns>
        public Point2f[] perspectiveTransform(IEnumerable<Point2f> src, NDArray m)
        {
            return Cv2.PerspectiveTransform(src, m.AsMat());
        }

        /// <summary>
        /// performs perspective transformation of each element of multi-channel input matrix
        /// </summary>
        /// <param name="src">The source two-channel or three-channel floating-point array; 
        /// each element is 2D/3D vector to be transformed</param>
        /// <param name="m">3x3 or 4x4 transformation matrix</param>
        /// <returns>The destination array; it will have the same size and same type as src</returns>
        public Point2d[] perspectiveTransform(IEnumerable<Point2d> src, NDArray m)
        {
            return Cv2.PerspectiveTransform(src, m.AsMat());
        }

        /// <summary>
        /// performs perspective transformation of each element of multi-channel input matrix
        /// </summary>
        /// <param name="src">The source two-channel or three-channel floating-point array; 
        /// each element is 2D/3D vector to be transformed</param>
        /// <param name="m">3x3 or 4x4 transformation matrix</param>
        /// <returns>The destination array; it will have the same size and same type as src</returns>
        public Point3f[] perspectiveTransform(IEnumerable<Point3f> src, NDArray m)
        {
            return Cv2.PerspectiveTransform(src, m.AsMat());
        }

        /// <summary>
        /// performs perspective transformation of each element of multi-channel input matrix
        /// </summary>
        /// <param name="src">The source two-channel or three-channel floating-point array; 
        /// each element is 2D/3D vector to be transformed</param>
        /// <param name="m">3x3 or 4x4 transformation matrix</param>
        /// <returns>The destination array; it will have the same size and same type as src</returns>
        public Point3d[] perspectiveTransform(IEnumerable<Point3d> src, NDArray m)
        {
            return Cv2.PerspectiveTransform(src, m.AsMat());
        }

        /// <summary>
        /// extends the symmetrical matrix from the lower half or from the upper half
        /// </summary>
        /// <param name="mtx"> Input-output floating-point square matrix</param>
        /// <param name="lowerToUpper">If true, the lower half is copied to the upper half, 
        /// otherwise the upper half is copied to the lower half</param>
        /// <returns>mtx</returns>
        public NDArray completeSymm(NDArray mtx, bool lowerToUpper = false)
        {
            if (!mtx.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "completeSymm needs that. Please consider change the adapter mode.");
            }
            Cv2.CompleteSymm(mtx.AsMat(), lowerToUpper);
            return mtx;
        }

        /// <summary>
        /// initializes scaled identity matrix
        /// </summary>
        /// <param name="mtx">The matrix to initialize (not necessarily square)</param>
        /// <param name="s">The value to assign to the diagonal elements</param>
        /// <returns>mtx</returns>
        public NDArray setIdentity(NDArray mtx, Scalar? s = null)
        {
            if (!mtx.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "setIdentity needs that. Please consider change the adapter mode.");
            }
            Cv2.SetIdentity(mtx.AsMat(), s);
            return mtx;
        }

        /// <summary>
        /// computes determinant of a square matrix
        /// </summary>
        /// <param name="mtx">The input matrix; must have CV_32FC1 or CV_64FC1 type and square size</param>
        /// <returns>determinant of the specified matrix.</returns>
        public double determinant(NDArray mtx)
        {
            return Cv2.Determinant(mtx.AsMat());
        }

        /// <summary>
        /// computes trace of a matrix
        /// </summary>
        /// <param name="mtx">The source matrix</param>
        /// <returns></returns>
        public Scalar trace(NDArray mtx)
        {
            return Cv2.Trace(mtx.AsMat());
        }

        /// <summary>
        /// computes inverse or pseudo-inverse matrix
        /// </summary>
        /// <param name="src">The source floating-point MxN matrix</param>
        /// <param name="dst">The destination matrix; will have NxM size and the same type as src</param>
        /// <param name="flags">The inversion method</param>
        /// <returns></returns>
        public (double, NDArray) invert(NDArray src, DecompTypes flags = DecompTypes.LU)
        {
            Mat dstMat = new();
            double retVal = Cv2.Invert(src.AsMat(), dstMat, flags);
            return (retVal, dstMat.numpy());
        }

        /// <summary>
        /// solves linear system or a least-square problem
        /// </summary>
        /// <param name="src1"></param>
        /// <param name="src2"></param>
        /// <param name="flags"></param>
        /// <returns>retVal and dst</returns>
        public (bool, NDArray) solve(NDArray src1, NDArray src2, DecompTypes flags = DecompTypes.LU)
        {
            Mat dstMat = new();
            bool retVal = Cv2.Solve(src1.AsMat(), src2.AsMat(), dstMat, flags);
            return (retVal, dstMat.numpy());
        }

        /// <summary>
        /// Solve given (non-integer) linear programming problem using the Simplex Algorithm (Simplex Method).
        /// </summary>
        /// <param name="func">This row-vector corresponds to \f$c\f$ in the LP problem formulation (see above). 
        /// It should contain 32- or 64-bit floating point numbers.As a convenience, column-vector may be also submitted,
        /// in the latter case it is understood to correspond to \f$c^T\f$.</param>
        /// <param name="constr">`m`-by-`n+1` matrix, whose rightmost column corresponds to \f$b\f$ in formulation above 
        /// and the remaining to \f$A\f$. It should containt 32- or 64-bit floating point numbers.</param>
        /// <returns>solve result and the solution will be returned here as a column-vector - it corresponds to \f$c\f$ in the 
        /// formulation above.It will contain 64-bit floating point numbers.</returns>
        public (SolveLPResult, NDArray) solveLP(NDArray func, NDArray constr)
        {
            Mat dstMat = new();
            var retVal = Cv2.SolveLP(func.AsMat(), constr.AsMat(), dstMat);
            return (retVal, dstMat.numpy());
        }

        public NDArray sort(NDArray src, SortFlags flags)
        {
            Mat dstMat = new();
            Cv2.Sort(src.AsMat(), dstMat, flags);
            return dstMat.numpy();
        }

        public NDArray sortIdx(NDArray src, SortFlags flags)
        {
            Mat dstMat = new();
            Cv2.SortIdx(src.AsMat(), dstMat, flags);
            return dstMat.numpy();
        }

        /// <summary>
        /// finds real roots of a cubic polynomial
        /// </summary>
        /// <param name="coeffs">The equation coefficients, an array of 3 or 4 elements</param>
        /// <returns>solve result and the destination array of real roots which will have 1 or 3 elements</returns>
        public (int, NDArray) solveCubic(NDArray coeffs)
        {
            Mat dstMat = new();
            int retVal = Cv2.SolveCubic(coeffs.AsMat(), dstMat);
            return (retVal, dstMat.numpy());
        }

        public NDArray solvePoly(NDArray coeffs, int maxIters = 300)
        {
            Mat dstMat = new();
            Cv2.SolvePoly(coeffs.AsMat(), dstMat, maxIters);
            return dstMat.numpy();
        }

        /// <summary>
        /// Computes eigenvalues and eigenvectors of a symmetric matrix.
        /// </summary>
        /// <param name="src">The input matrix; must have CV_32FC1 or CV_64FC1 type, 
        /// square size and be symmetric: src^T == src</param>
        /// <param name="eigenvalues">The output vector of eigenvalues of the same type as src; 
        /// The eigenvalues are stored in the descending order.</param>
        /// <param name="eigenvectors">The output matrix of eigenvectors; 
        /// It will have the same size and the same type as src; The eigenvectors are stored 
        /// as subsequent matrix rows, in the same order as the corresponding eigenvalues</param>
        /// <returns>solve result; the output vector of eigenvalues of the same type as src,
        /// which are stored in the descending order; the output matrix of eigenvectors; 
        /// It will have the same size and the same type as src, which are stored 
        /// as subsequent matrix rows, in the same order as the corresponding eigenvalues</returns>
        public (bool, NDArray, NDArray) eigen(NDArray src)
        {
            Mat valuesMat = new();
            Mat vectorsMat = new();
            bool retVal = Cv2.Eigen(src.AsMat(), valuesMat, vectorsMat);
            return (retVal, valuesMat.numpy(), vectorsMat.numpy());
        }

        public (NDArray, NDArray) eigenNonSymmetric(NDArray src)
        {
            Mat valuesMat = new();
            Mat vectorsMat = new();
            Cv2.EigenNonSymmetric(src.AsMat(), valuesMat, vectorsMat);
            return (valuesMat.numpy(), vectorsMat.numpy());
        }

        public (NDArray, NDArray, NDArray) PCACompute(NDArray data, NDArray mean, int maxComponents = 0)
        {
            if (!mean.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "PCACompute needs that. Please consider change the adapter mode.");
            }
            Mat vectorsMat = new();
            Mat valuesMat = new();
            Cv2.PCACompute(data.AsMat(), mean.AsMat(), vectorsMat, valuesMat, maxComponents);
            return (mean, vectorsMat.numpy(), valuesMat.numpy());
        }

        public (NDArray, NDArray, NDArray) PCAComputeVar(NDArray data, NDArray mean, double retainedVariance)
        {
            if (!mean.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "PCAComputeVar needs that. Please consider change the adapter mode.");
            }
            Mat vectorsMat = new();
            Mat valuesMat = new();
            Cv2.PCAComputeVar(data.AsMat(), mean.AsMat(), vectorsMat, valuesMat, retainedVariance);
            return (mean, vectorsMat.numpy(), valuesMat.numpy());
        }

        public NDArray PCAProject(NDArray data, NDArray mean, NDArray eigenvectors)
        {
            Mat dstMat = new();
            Cv2.PCAProject(data.AsMat(), mean.AsMat(), eigenvectors.AsMat(), dstMat);
            return dstMat.numpy();
        }

        public NDArray PCABackProject(NDArray data, NDArray mean, NDArray eigenvectors)
        {
            Mat dstMat = new();
            Cv2.PCABackProject(data.AsMat(), mean.AsMat(), eigenvectors.AsMat(), dstMat);
            return dstMat.numpy();
        }

        public (NDArray, NDArray, NDArray) SVDecomp(NDArray src, SVD.Flags flags = SVD.Flags.None)
        {
            Mat wMat = new();
            Mat uMat = new();
            Mat vtMat = new();
            Cv2.SVDecomp(src.AsMat(), wMat, uMat, vtMat, flags);
            return (wMat.numpy(), uMat.numpy(), vtMat.numpy());
        }

        public NDArray SVBackSubst(NDArray w, NDArray u, NDArray vt, NDArray rhs)
        {
            Mat dstMat = new();
            Cv2.SVBackSubst(w.AsMat(), u.AsMat(), vt.AsMat(), rhs.AsMat(), dstMat);
            return dstMat.numpy();
        }

        public double Mahalanobis(NDArray v1, NDArray v2, NDArray icover)
        {
            return Cv2.Mahalanobis(v1.AsMat(), v2.AsMat(), icover.AsMat());
        }

        public NDArray dft(NDArray src, DftFlags flags = DftFlags.None, int nonzeroRows = 0)
        {
            Mat dstMat = new();
            Cv2.Dft(src.AsMat(), dstMat, flags, nonzeroRows);
            return dstMat.numpy();
        }

        public NDArray idft(NDArray src, DftFlags flags = DftFlags.None, int nonzeroRows = 0)
        {
            Mat dstMat = new();
            Cv2.Idft(src.AsMat(), dstMat, flags, nonzeroRows);
            return dstMat.numpy();
        }

        public NDArray dct(NDArray src, DctFlags flags = DctFlags.None)
        {
            Mat dstMat = new();
            Cv2.Dct(src.AsMat(), dstMat, flags);
            return dstMat.numpy();
        }

        public NDArray idct(NDArray src, DctFlags flags = DctFlags.None)
        {
            Mat dstMat = new();
            Cv2.Idct(src.AsMat(), dstMat, flags);
            return dstMat.numpy();
        }

        public NDArray mulSpectrums(NDArray a, NDArray b, DftFlags flags, bool conjB = false)
        {
            Mat dstMat = new();
            Cv2.MulSpectrums(a.AsMat(), b.AsMat(), dstMat, flags, conjB);
            return dstMat.numpy();
        }

        public int getOptimalDFTSize(int vecSize)
        {
            return Cv2.GetOptimalDFTSize(vecSize);
        }

        public NDArray randu(NDArray dst, NDArray low, NDArray high)
        {
            if (!dst.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "randu needs that. Please consider change the adapter mode.");
            }
            Cv2.Randu(dst.AsMat(), low.AsMat(), high.AsMat());
            return dst;
        }

        public NDArray randu(NDArray dst, Scalar low, Scalar high)
        {
            if (!dst.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "randu needs that. Please consider change the adapter mode.");
            }
            Cv2.Randu(dst.AsMat(), low, high);
            return dst;
        }

        public NDArray randn(NDArray dst, NDArray mean, NDArray stddev)
        {
            if (!dst.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "randu needs that. Please consider change the adapter mode.");
            }
            Cv2.Randn(dst.AsMat(), mean.AsMat(), stddev.AsMat());
            return dst;
        }

        public NDArray randn(NDArray dst, Scalar mean, Scalar stddev)
        {
            if (!dst.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "randu needs that. Please consider change the adapter mode.");
            }
            Cv2.Randn(dst.AsMat(), mean, stddev);
            return dst;
        }

        public (NDArray, NDArray?) kmeans(NDArray data, int k, NDArray bestLabels, TermCriteria criteria,
            int attempts, KMeansFlags flags, NDArray? centers)
        {
            if (!bestLabels.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "kmeans needs that. Please consider change the adapter mode.");
            }
            Mat? centersMat = centers is null ? null : centers.AsMat();
            Cv2.Kmeans(data.AsMat(), k, bestLabels.AsMat(), criteria, attempts, flags, centersMat);
            if(centersMat is null)
            {
                return (bestLabels, null);
            }
            else
            {
                return (bestLabels, centersMat.ToNDArray(true));
            }
        }

        public float fastAten2(float y, float x)
        {
            return Cv2.FastAtan2(y, x);
        }

        public float cubeRoot(float val)
        {
            return Cv2.CubeRoot(val);
        }

        public void setNumThreads(int nThreads)
        {
            Cv2.SetNumThreads(nThreads);
        }

        public int getNumThreads()
        {
            return Cv2.GetNumThreads();
        }

        public int getThreadNum()
        {
            return Cv2.GetThreadNum();
        }

        public string getBuildInformation()
        {
            return Cv2.GetBuildInformation();
        }

        public string? getVersionString()
        {
            return Cv2.GetVersionString();
        }

        public int getVersionMajor()
        {
            return Cv2.GetVersionMajor();
        }

        public int getVersionMinor()
        {
            return Cv2.GetVersionMinor();
        }

        public int getVersionRevision()
        {
            return Cv2.GetVersionRevision();
        }

        public long getTickCount()
        {
            return Cv2.GetTickCount();
        }

        public double getTickFrequency()
        {
            return Cv2.GetTickFrequency();
        }

        public long getCpuTickCount()
        {
            return Cv2.GetCpuTickCount();
        }

        public bool checkHardwareSupport(CpuFeatures feature)
        {
            return Cv2.CheckHardwareSupport(feature);
        }

        public string getHardwareFeatureName(CpuFeatures feature)
        {
            return Cv2.GetHardwareFeatureName(feature);
        }

        public string getCpuFeaturesLine()
        {
            return Cv2.GetCpuFeaturesLine();
        }

        public int getNumberOfCpus()
        {
            return Cv2.GetNumberOfCpus();
        }

        public void setUseOptimized(bool onoff)
        {
            Cv2.SetUseOptimized(onoff);
        }

        public bool useOptimized()
        {
            return Cv2.UseOptimized();
        }

        public int alignSize(int sz, int n)
        {
            return Cv2.AlignSize(sz, n);
        }

        public bool setBreakOnError(bool flag)
        {
            return Cv2.SetBreakOnError(flag);
        }

        public string format(NDArray mtx, FormatType format = FormatType.Default)
        {
            return Cv2.Format(mtx.AsMat(), format);
        }
    }
}
