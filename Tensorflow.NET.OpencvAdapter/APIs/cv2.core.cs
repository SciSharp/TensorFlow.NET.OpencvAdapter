using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
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
    }
}
