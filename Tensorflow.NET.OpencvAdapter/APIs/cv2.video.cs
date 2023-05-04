using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.NumPy;
using Tensorflow.OpencvAdapter.Extensions;

namespace Tensorflow.OpencvAdapter.APIs
{
    public partial class Cv2API
    {
        /// <summary>
        /// Finds an object center, size, and orientation.
        /// </summary>
        /// <param name="probImage">Back projection of the object histogram. </param>
        /// <param name="window">Initial search window.</param>
        /// <param name="criteria">Stop criteria for the underlying MeanShift() .</param>
        /// <returns></returns>
        public RotatedRect camShift(NDArray probImage, ref Rect window, TermCriteria criteria)
        {
            return Cv2.CamShift(probImage.AsMat(), ref window, criteria);
        }

        /// <summary>
        /// Finds an object on a back projection image.
        /// </summary>
        /// <param name="probImage">Back projection of the object histogram.</param>
        /// <param name="window">Initial search window.</param>
        /// <param name="criteria">Stop criteria for the iterative search algorithm.</param>
        /// <returns>Number of iterations CAMSHIFT took to converge.</returns>
        public int meanShift(NDArray probImage, ref Rect window, TermCriteria criteria)
        {
            return Cv2.MeanShift(probImage.AsMat(), ref window, criteria);
        }

        /// <summary>
        /// Constructs a pyramid which can be used as input for calcOpticalFlowPyrLK
        /// </summary>
        /// <param name="img">8-bit input image.</param>
        /// <param name="winSize">window size of optical flow algorithm. 
        /// Must be not less than winSize argument of calcOpticalFlowPyrLK(). 
        /// It is needed to calculate required padding for pyramid levels.</param>
        /// <param name="maxLevel">0-based maximal pyramid level number.</param>
        /// <param name="withDerivatives">set to precompute gradients for the every pyramid level. 
        /// If pyramid is constructed without the gradients then calcOpticalFlowPyrLK() will 
        /// calculate them internally.</param>
        /// <param name="pyrBorder">the border mode for pyramid layers.</param>
        /// <param name="derivBorder">the border mode for gradients.</param>
        /// <param name="tryReuseInputImage">put ROI of input image into the pyramid if possible. 
        /// You can pass false to force data copying.</param>
        /// <returns>1. number of levels in constructed pyramid. Can be less than maxLevel. 
        /// 2. output pyramid.</returns>
        public (int, NDArray) buildOpticalFlowPyramid(NDArray img, Size winSize, int maxLevel,
            bool withDerivatives = true,
            BorderTypes pyrBorder = BorderTypes.Reflect101,
            BorderTypes derivBorder = BorderTypes.Constant,
            bool tryReuseInputImage = true)
        {
            Mat dstMat = new();
            var retVal = Cv2.BuildOpticalFlowPyramid(img.AsMat(), dstMat, winSize, maxLevel, 
                withDerivatives, pyrBorder, derivBorder, tryReuseInputImage);
            return (retVal, dstMat.numpy());
        }

        /// <summary>
        /// computes sparse optical flow using multi-scale Lucas-Kanade algorithm
        /// </summary>
        /// <param name="prevImg"></param>
        /// <param name="nextImg"></param>
        /// <param name="prevPts"></param>
        /// <param name="nextPts"></param>
        /// <param name="winSize"></param>
        /// <param name="maxLevel"></param>
        /// <param name="criteria"></param>
        /// <param name="flags"></param>
        /// <param name="minEigThreshold"></param>
        public (NDArray, NDArray, NDArray) calcOpticalFlowPyrLK(NDArray prevImg, NDArray nextImg, NDArray prevPts, NDArray nextPts,
            Size? winSize = null,
            int maxLevel = 3,
            TermCriteria? criteria = null,
            OpticalFlowFlags flags = OpticalFlowFlags.None,
            double minEigThreshold = 1e-4)
        {
            Mat statusMat = new();
            Mat errMat = new();
            Cv2.CalcOpticalFlowPyrLK(prevImg.AsMat(), nextImg.AsMat(), prevPts.AsMat(), nextPts.AsMat(), 
                statusMat, errMat, winSize, maxLevel, criteria, flags, minEigThreshold);
            return (nextPts, statusMat.numpy(), errMat.numpy());
        }

        /// <summary>
        /// Computes a dense optical flow using the Gunnar Farneback's algorithm.
        /// </summary>
        /// <param name="prev">first 8-bit single-channel input image.</param>
        /// <param name="next">second input image of the same size and the same type as prev.</param>
        /// <param name="flow">computed flow image that has the same size as prev and type CV_32FC2.</param>
        /// <param name="pyrScale">parameter, specifying the image scale (&lt;1) to build pyramids for each image; 
        /// pyrScale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.</param>
        /// <param name="levels">number of pyramid layers including the initial image; 
        /// levels=1 means that no extra layers are created and only the original images are used.</param>
        /// <param name="winsize">averaging window size; larger values increase the algorithm robustness to 
        /// image noise and give more chances for fast motion detection, but yield more blurred motion field.</param>
        /// <param name="iterations">number of iterations the algorithm does at each pyramid level.</param>
        /// <param name="polyN">size of the pixel neighborhood used to find polynomial expansion in each pixel; 
        /// larger values mean that the image will be approximated with smoother surfaces, 
        /// yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.</param>
        /// <param name="polySigma">standard deviation of the Gaussian that is used to smooth derivatives used as 
        /// a basis for the polynomial expansion; for polyN=5, you can set polySigma=1.1, 
        /// for polyN=7, a good value would be polySigma=1.5.</param>
        /// <param name="flags">operation flags that can be a combination of OPTFLOW_USE_INITIAL_FLOW and/or OPTFLOW_FARNEBACK_GAUSSIAN</param>
        public NDArray calcOpticalFlowFarneback(NDArray prev, NDArray next, NDArray flow,
            double pyrScale, int levels, int winsize, int iterations, int polyN, double polySigma, OpticalFlowFlags flags)
        {
            Cv2.CalcOpticalFlowFarneback(prev.AsMat(), next.AsMat(), flow.AsMat(), pyrScale, 
                levels, winsize, iterations, polyN, polySigma, flags);
            return flow;
        }

        /// <summary>
        /// Computes the Enhanced Correlation Coefficient value between two images @cite EP08 .
        /// </summary>
        /// <param name="templateImage">single-channel template image; CV_8U or CV_32F array.</param>
        /// <param name="inputImage">single-channel input image to be warped to provide an image similar to templateImage, same type as templateImage.</param>
        /// <param name="inputMask">An optional mask to indicate valid values of inputImage.</param>
        /// <returns></returns>
        public double computeECC(NDArray templateImage, NDArray inputImage, NDArray? inputMask = null)
        {
            return Cv2.ComputeECC(templateImage.AsMat(), inputImage.AsMat(), inputMask?.AsMat());
        }

        /// <summary>
        /// Finds the geometric transform (warp) between two images in terms of the ECC criterion @cite EP08 .
        /// </summary>
        /// <param name="templateImage">single-channel template image; CV_8U or CV_32F array.</param>
        /// <param name="inputImage">single-channel input image which should be warped with the final warpMatrix in
        /// order to provide an image similar to templateImage, same type as templateImage.</param>
        /// <param name="warpMatrix">floating-point \f$2\times 3\f$ or \f$3\times 3\f$ mapping matrix (warp).</param>
        /// <param name="motionType">parameter, specifying the type of motion</param>
        /// <param name="criteria">parameter, specifying the termination criteria of the ECC algorithm;
        /// criteria.epsilon defines the threshold of the increment in the correlation coefficient between two
        /// iterations(a negative criteria.epsilon makes criteria.maxcount the only termination criterion).
        /// Default values are shown in the declaration above.</param>
        /// <param name="inputMask">An optional mask to indicate valid values of inputImage.</param>
        /// <param name="gaussFiltSize">An optional value indicating size of gaussian blur filter; (DEFAULT: 5)</param>
        /// <returns></returns>
        public (double, NDArray) findTransformECC(NDArray templateImage, NDArray inputImage, NDArray warpMatrix, MotionTypes motionType,
            TermCriteria criteria, NDArray? inputMask = null, int gaussFiltSize = 5)
        {
            var retVal = Cv2.FindTransformECC(templateImage.AsMat(), inputImage.AsMat(), warpMatrix.AsMat(), 
                motionType, criteria, inputMask.ToInputArray(), gaussFiltSize);
            return (retVal, warpMatrix);
        }

        /// <summary>
        /// Finds the geometric transform (warp) between two images in terms of the ECC criterion @cite EP08 .
        /// </summary>
        /// <param name="templateImage">single-channel template image; CV_8U or CV_32F array.</param>
        /// <param name="inputImage">single-channel input image which should be warped with the final warpMatrix in
        /// order to provide an image similar to templateImage, same type as templateImage.</param>
        /// <param name="warpMatrix">floating-point \f$2\times 3\f$ or \f$3\times 3\f$ mapping matrix (warp).</param>
        /// <param name="motionType">parameter, specifying the type of motion</param>
        /// <param name="criteria">parameter, specifying the termination criteria of the ECC algorithm;
        /// criteria.epsilon defines the threshold of the increment in the correlation coefficient between two
        /// iterations(a negative criteria.epsilon makes criteria.maxcount the only termination criterion).
        /// Default values are shown in the declaration above.</param>
        /// <param name="inputMask">An optional mask to indicate valid values of inputImage.</param>
        /// <returns></returns>
        public (double, NDArray) findTransformECC(NDArray templateImage, NDArray inputImage, NDArray warpMatrix, MotionTypes motionType,
            TermCriteria criteria, NDArray? inputMask = null)
        {
            var retVal = Cv2.FindTransformECC(templateImage.AsMat(), inputImage.AsMat(), warpMatrix.AsMat(),
                motionType, criteria, inputMask.ToInputArray());
            return (retVal, warpMatrix);
        }


    }
}
