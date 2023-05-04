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
        /// Returns Gaussian filter coefficients.
        /// </summary>
        /// <param name="ksize">Aperture size. It should be odd and positive.</param>
        /// <param name="sigma">Gaussian standard deviation.
        /// If it is non-positive, it is computed from ksize as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`.</param>
        /// <param name="ktype">Type of filter coefficients. It can be CV_32F or CV_64F.</param>
        /// <returns></returns>
        public NDArray? getGaussianKernel(int ksize, double sigma, MatType? ktype = null)
        {
            return Cv2.GetGaussianKernel(ksize, sigma, ktype)?.numpy();
        }

        /// <summary>
        /// Returns filter coefficients for computing spatial image derivatives.
        /// </summary>
        /// <param name="dx">Derivative order in respect of x.</param>
        /// <param name="dy">Derivative order in respect of y.</param>
        /// <param name="ksize">Aperture size. It can be CV_SCHARR, 1, 3, 5, or 7.</param>
        /// <param name="normalize">Flag indicating whether to normalize (scale down) the filter coefficients or not.
        /// Theoretically, the coefficients should have the denominator \f$=2^{ksize*2-dx-dy-2}\f$. 
        /// If you are going to filter floating-point images, you are likely to use the normalized kernels.
        /// But if you compute derivatives of an 8-bit image, store the results in a 16-bit image, 
        /// and wish to preserve all the fractional bits, you may want to set normalize = false.</param>
        /// <param name="ktype">Type of filter coefficients. It can be CV_32f or CV_64F.</param>
        /// <returns>Output matrix of row filter coefficients and output matrix of column filter coefficients. Thet have the type ktype.</returns>
        public (NDArray, NDArray) getDerivKernels(int dx, int dy, int ksize, bool normalize = false, MatType? ktype = null)
        {
            Mat kxMat = new();
            Mat kyMat = new();
            Cv2.GetDerivKernels(kxMat, kyMat, dx, dy, ksize, normalize, ktype);
            return (kxMat.numpy(), kyMat.numpy());
        }

        /// <summary>
        /// Returns Gabor filter coefficients. 
        /// </summary>
        /// <remarks>
        /// For more details about gabor filter equations and parameters, see: https://en.wikipedia.org/wiki/Gabor_filter
        /// </remarks>
        /// <param name="ksize">Size of the filter returned.</param>
        /// <param name="sigma">Standard deviation of the gaussian envelope.</param>
        /// <param name="theta">Orientation of the normal to the parallel stripes of a Gabor function.</param>
        /// <param name="lambd">Wavelength of the sinusoidal factor.</param>
        /// <param name="gamma">Spatial aspect ratio.</param>
        /// <param name="psi">Phase offset.</param>
        /// <param name="ktype">Type of filter coefficients. It can be CV_32F or CV_64F.</param>
        public NDArray getGaborKernel(Size ksize, double sigma, double theta, double lambd, double gamma, double psi, int ktype)
        {
            return Cv2.GetGaborKernel(ksize, sigma, theta, lambd, gamma, psi, ktype).numpy();
        }

        /// <summary>
        /// Returns a structuring element of the specified size and shape for morphological operations.
        /// The function constructs and returns the structuring element that can be further passed to erode,
        /// dilate or morphologyEx.But you can also construct an arbitrary binary mask yourself and use it as the structuring element.
        /// </summary>
        /// <param name="shape">Element shape that could be one of MorphShapes</param>
        /// <param name="ksize">Size of the structuring element.</param>
        /// <returns></returns>
        public NDArray getStructuringElement(MorphShapes shape, Size ksize)
        {
            return Cv2.GetStructuringElement(shape, ksize).numpy();
        }

        /// <summary>
        /// Returns a structuring element of the specified size and shape for morphological operations.
        /// The function constructs and returns the structuring element that can be further passed to erode,
        /// dilate or morphologyEx.But you can also construct an arbitrary binary mask yourself and use it as the structuring element.
        /// </summary>
        /// <param name="shape">Element shape that could be one of MorphShapes</param>
        /// <param name="ksize">Size of the structuring element.</param>
        /// <param name="anchor">Anchor position within the element. The default value (−1,−1) means that the anchor is at the center.
        /// Note that only the shape of a cross-shaped element depends on the anchor position.
        /// In other cases the anchor just regulates how much the result of the morphological operation is shifted.</param>
        /// <returns></returns>
        public NDArray getStructuringElement(MorphShapes shape, Size ksize, Point anchor)
        {
            return Cv2.GetStructuringElement(shape, ksize, anchor).numpy();
        }

        /// <summary>
        /// Smoothes image using median filter
        /// </summary>
        /// <param name="src">The source 1-, 3- or 4-channel image. 
        /// When ksize is 3 or 5, the image depth should be CV_8U , CV_16U or CV_32F. 
        /// For larger aperture sizes it can only be CV_8U</param>
        /// <param name="ksize">The aperture linear size. It must be odd and more than 1, i.e. 3, 5, 7 ...</param>
        /// <return>The destination array; will have the same size and the same type as src</return>
        public NDArray medianBlur(NDArray src, int ksize)
        {
            Mat dstMat = new();
            Cv2.MedianBlur(src.AsMat(), dstMat, ksize);
            return dstMat.numpy();
        }

        /// <summary>
        /// Blurs an image using a Gaussian filter.
        /// </summary>
        /// <param name="src">input image; the image can have any number of channels, which are processed independently, 
        /// but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.</param>
        /// <param name="ksize">Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd. 
        /// Or, they can be zero’s and then they are computed from sigma* .</param>
        /// <param name="sigmaX">Gaussian kernel standard deviation in X direction.</param>
        /// <param name="sigmaY">Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal to sigmaX, 
        /// if both sigmas are zeros, they are computed from ksize.width and ksize.height, 
        /// respectively (see getGaussianKernel() for details); to fully control the result 
        /// regardless of possible future modifications of all this semantics, it is recommended to specify all of ksize, sigmaX, and sigmaY.</param>
        /// <param name="borderType">pixel extrapolation method</param>
        /// <returns>output image of the same size and type as src.</returns>
        public NDArray gaussianBlur(NDArray src, Size ksize, double sigmaX, 
            double sigmaY = 0, BorderTypes borderType = BorderTypes.Default)
        {
            Mat dstMat = new();
            Cv2.GaussianBlur(src.AsMat(), dstMat, ksize, sigmaX, sigmaY, borderType);
            return dstMat.numpy();
        }

        /// <summary>
        /// Applies bilateral filter to the image
        /// </summary>
        /// <param name="src">The source 8-bit or floating-point, 1-channel or 3-channel image</param>
        /// <param name="d">The diameter of each pixel neighborhood, that is used during filtering. 
        /// If it is non-positive, it's computed from sigmaSpace</param>
        /// <param name="sigmaColor">Filter sigma in the color space. 
        /// Larger value of the parameter means that farther colors within the pixel neighborhood 
        /// will be mixed together, resulting in larger areas of semi-equal color</param>
        /// <param name="sigmaSpace">Filter sigma in the coordinate space. 
        /// Larger value of the parameter means that farther pixels will influence each other 
        /// (as long as their colors are close enough; see sigmaColor). Then d>0 , it specifies 
        /// the neighborhood size regardless of sigmaSpace, otherwise d is proportional to sigmaSpace</param>
        /// <param name="borderType"></param>
        /// <returns>The destination image; will have the same size and the same type as src</returns>
        public NDArray bilateralFilter(NDArray src, int d, double sigmaColor,
            double sigmaSpace, BorderTypes borderType = BorderTypes.Default)
        {
            Mat dstMat = new();
            Cv2.BilateralFilter(src.AsMat(), dstMat, d, sigmaColor, sigmaSpace, borderType);
            return dstMat.numpy();
        }

        /// <summary>
        /// Smoothes image using box filter
        /// </summary>
        /// <param name="src">The source image</param>
        /// <param name="ddepth"></param>
        /// <param name="ksize">The smoothing kernel size</param>
        /// <param name="anchor">The anchor point. The default value Point(-1,-1) means that the anchor is at the kernel center</param>
        /// <param name="normalize">Indicates, whether the kernel is normalized by its area or not</param>
        /// <param name="borderType">The border mode used to extrapolate pixels outside of the image</param>
        /// <returns>The destination image; will have the same size and the same type as src</returns>
        public NDArray boxFilter(NDArray src, MatType ddepth,
            Size ksize, Point? anchor = null, bool normalize = true,
            BorderTypes borderType = BorderTypes.Default)
        {
            Mat dstMat = new();
            Cv2.BoxFilter(src.AsMat(), dstMat, ddepth, ksize, anchor, normalize, borderType);
            return dstMat.numpy();
        }

        /// <summary>
        /// Calculates the normalized sum of squares of the pixel values overlapping the filter.
        ///
        /// For every pixel f(x, y) in the source image, the function calculates the sum of squares of those neighboring
        /// pixel values which overlap the filter placed over the pixel f(x, y).
        ///
        /// The unnormalized square box filter can be useful in computing local image statistics such as the the local
        /// variance and standard deviation around the neighborhood of a pixel.
        /// </summary>
        /// <param name="src"></param>
        /// <param name="ddepth"></param>
        /// <param name="ksize"></param>
        /// <param name="anchor"></param>
        /// <param name="normalize"></param>
        /// <param name="borderType"></param>
        /// <returns></returns>
        public NDArray sqrBoxFilter(NDArray src, int ddepth,
            Size ksize, Point? anchor = null, bool normalize = true,
            BorderTypes borderType = BorderTypes.Default)
        {
            Mat dstMat = new();
            Cv2.SqrBoxFilter(src.AsMat(), dstMat, ddepth, ksize, anchor, normalize, borderType);
            return dstMat.numpy();
        }

        /// <summary>
        /// Smoothes image using normalized box filter
        /// </summary>
        /// <param name="src">The source image</param>
        /// <param name="ksize">The smoothing kernel size</param>
        /// <param name="anchor">The anchor point. The default value Point(-1,-1) means that the anchor is at the kernel center</param>
        /// <param name="borderType">The border mode used to extrapolate pixels outside of the image</param>
        /// <returns>The destination image; will have the same size and the same type as src</returns>
        public NDArray blur(NDArray src, Size ksize,
            Point? anchor = null, BorderTypes borderType = BorderTypes.Default)
        {
            Mat dstMat = new();
            Cv2.Blur(src.AsMat(), dstMat, ksize, anchor, borderType);
            return dstMat.numpy();
        }

        /// <summary>
        /// Convolves an image with the kernel
        /// </summary>
        /// <param name="src">The source image</param>
        /// <param name="ddepth">The desired depth of the destination image. If it is negative, it will be the same as src.depth()</param>
        /// <param name="kernel">Convolution kernel (or rather a correlation kernel), 
        /// a single-channel floating point matrix. If you want to apply different kernels to 
        /// different channels, split the image into separate color planes using split() and process them individually</param>
        /// <param name="anchor">The anchor of the kernel that indicates the relative position of 
        /// a filtered point within the kernel. The anchor should lie within the kernel. 
        /// The special default value (-1,-1) means that the anchor is at the kernel center</param>
        /// <param name="delta">The optional value added to the filtered pixels before storing them in dst</param>
        /// <param name="borderType">The pixel extrapolation method</param>
        /// <returns>The destination image. It will have the same size and the same number of channels as src</returns>
        public NDArray filter2D(NDArray src, MatType ddepth,
            NDArray kernel, Point? anchor = null, double delta = 0,
            BorderTypes borderType = BorderTypes.Default)
        {
            Mat dstMat = new();
            Cv2.Filter2D(src.AsMat(), dstMat, ddepth, kernel.AsMat(), anchor, delta, borderType);
            return dstMat.numpy();
        }

        /// <summary>
        /// Applies separable linear filter to an image
        /// </summary>
        /// <param name="src">The source image</param>
        /// <param name="ddepth">The destination image depth</param>
        /// <param name="kernelX">The coefficients for filtering each row</param>
        /// <param name="kernelY">The coefficients for filtering each column</param>
        /// <param name="anchor">The anchor position within the kernel; The default value (-1, 1) means that the anchor is at the kernel center</param>
        /// <param name="delta">The value added to the filtered results before storing them</param>
        /// <param name="borderType">The pixel extrapolation method</param>
        /// <returns>The destination image; will have the same size and the same number of channels as src</returns>
        public NDArray sepFilter2D(NDArray src, MatType ddepth,
            NDArray kernelX, NDArray kernelY, Point? anchor = null, double delta = 0,
            BorderTypes borderType = BorderTypes.Default)
        {
            Mat dstMat = new();
            Cv2.SepFilter2D(src.AsMat(), dstMat, ddepth, kernelX.AsMat(), kernelY.AsMat(), anchor, delta, borderType);
            return dstMat.numpy();
        }

        /// <summary>
        /// Calculates the first, second, third or mixed image derivatives using an extended Sobel operator
        /// </summary>
        /// <param name="src">The source image</param>
        /// <param name="ddepth">The destination image depth</param>
        /// <param name="xorder">Order of the derivative x</param>
        /// <param name="yorder">Order of the derivative y</param>
        /// <param name="ksize">Size of the extended Sobel kernel, must be 1, 3, 5 or 7</param>
        /// <param name="scale">The optional scale factor for the computed derivative values (by default, no scaling is applied</param>
        /// <param name="delta">The optional delta value, added to the results prior to storing them in dst</param>
        /// <param name="borderType">The pixel extrapolation method</param>
        /// <returns>The destination image; will have the same size and the same number of channels as src</returns>
        public NDArray sobel(NDArray src, MatType ddepth, int xorder, int yorder,
            int ksize = 3, double scale = 1, double delta = 0,
            BorderTypes borderType = BorderTypes.Default)
        {
            Mat dstMat = new();
            Cv2.Sobel(src.AsMat(), dstMat, ddepth, xorder, yorder, ksize, scale, delta, borderType);
            return dstMat.numpy();
        }

        /// <summary>
        /// Calculates the first order image derivative in both x and y using a Sobel operator
        /// </summary>
        /// <param name="src">input image.</param>
        /// <param name="ksize">size of Sobel kernel. It must be 3.</param>
        /// <param name="borderType">pixel extrapolation method</param>
        /// <returns>dx and dy</returns>
        public (NDArray, NDArray) spatialGradient(NDArray src, int ksize = 3, BorderTypes borderType = BorderTypes.Default)
        {
            Mat dxMat = new();
            Mat dyMat = new();
            Cv2.SpatialGradient(src.AsMat(), dxMat, dyMat, ksize, borderType);
            return (dxMat.numpy(), dyMat.numpy());
        }

        /// <summary>
        /// Calculates the first x- or y- image derivative using Scharr operator
        /// </summary>
        /// <param name="src">The source image</param>
        /// <param name="ddepth">The destination image depth</param>
        /// <param name="xorder">Order of the derivative x</param>
        /// <param name="yorder">Order of the derivative y</param>
        /// <param name="scale">The optional scale factor for the computed derivative values (by default, no scaling is applie</param>
        /// <param name="delta">The optional delta value, added to the results prior to storing them in dst</param>
        /// <param name="borderType">The pixel extrapolation method</param>
        /// <returns>The destination image; will have the same size and the same number of channels as src</returns>
        public NDArray scharr(NDArray src, MatType ddepth, int xorder, int yorder,
            double scale = 1, double delta = 0, BorderTypes borderType = BorderTypes.Default)
        {
            Mat dstMat = new();
            Cv2.Scharr(src.AsMat(), dstMat, ddepth, xorder, yorder, scale, delta, borderType);
            return dstMat.numpy();
        }

        /// <summary>
        /// Calculates the Laplacian of an image
        /// </summary>
        /// <param name="src">Source image</param>
        /// <param name="ddepth">The desired depth of the destination image</param>
        /// <param name="ksize">The aperture size used to compute the second-derivative filters</param>
        /// <param name="scale">The optional scale factor for the computed Laplacian values (by default, no scaling is applied</param>
        /// <param name="delta">The optional delta value, added to the results prior to storing them in dst</param>
        /// <param name="borderType">The pixel extrapolation method</param>
        /// <returns>Destination image; will have the same size and the same number of channels as src</returns>
        public NDArray laplacian(NDArray src, MatType ddepth,
            int ksize = 1, double scale = 1, double delta = 0,
            BorderTypes borderType = BorderTypes.Default)
        {
            Mat dstMat = new();
            Cv2.Laplacian(src.AsMat(), dstMat, ddepth, ksize, scale, delta, borderType);
            return dstMat.numpy();
        }

        /// <summary>
        /// Finds edges in an image using Canny algorithm.
        /// </summary>
        /// <param name="src">Single-channel 8-bit input image</param>
        /// <param name="threshold1">The first threshold for the hysteresis procedure</param>
        /// <param name="threshold2">The second threshold for the hysteresis procedure</param>
        /// <param name="apertureSize">Aperture size for the Sobel operator [By default this is ApertureSize.Size3]</param>
        /// <param name="L2gradient">Indicates, whether the more accurate L2 norm should be used to compute the image gradient magnitude (true), 
        /// or a faster default L1 norm is enough (false). [By default this is false]</param>
        /// <returns>The output edge map. It will have the same size and the same type as image</returns>
        public NDArray canny(NDArray src, double threshold1, double threshold2, 
            int apertureSize = 3, bool L2gradient = false)
        {
            Mat dstMat = new();
            Cv2.Canny(src.AsMat(), dstMat, threshold1, threshold2, apertureSize, L2gradient);
            return dstMat.numpy();
        }

        /// <summary>
        /// Finds edges in an image using the Canny algorithm with custom image gradient.
        /// </summary>
        /// <param name="dx">16-bit x derivative of input image (CV_16SC1 or CV_16SC3).</param>
        /// <param name="dy">16-bit y derivative of input image (same type as dx).</param>
        /// <param name="threshold1">first threshold for the hysteresis procedure.</param>
        /// <param name="threshold2">second threshold for the hysteresis procedure.</param>
        /// <param name="L2gradient">Indicates, whether the more accurate L2 norm should be used to compute the image gradient magnitude (true), 
        /// or a faster default L1 norm is enough (false). [By default this is false]</param>
        /// <returns>output edge map; single channels 8-bit image, which has the same size as image.</returns>
        public NDArray canny(NDArray dx, NDArray dy, double threshold1, double threshold2,
            bool L2gradient = false)
        {
            Mat dstMat = new();
            Cv2.Canny(dx.AsMat(), dy.AsMat(), dstMat, threshold1, threshold2, L2gradient);
            return dstMat.numpy();
        }

        /// <summary>
        /// Calculates the minimal eigenvalue of gradient matrices for corner detection.
        /// </summary>
        /// <param name="src">Input single-channel 8-bit or floating-point image.</param>
        /// <param name="blockSize">Neighborhood size (see the details on #cornerEigenValsAndVecs ).</param>
        /// <param name="ksize">Aperture parameter for the Sobel operator.</param>
        /// <param name="borderType">Pixel extrapolation method. See #BorderTypes. #BORDER_WRAP is not supported.</param>
        /// <returns>Image to store the minimal eigenvalues. It has the type CV_32FC1 and the same size as src .</returns>
        public NDArray cornerMinEigenVal(NDArray src, int blockSize,
            int ksize = 3, BorderTypes borderType = BorderTypes.Default)
        {
            Mat dstMat = new();
            Cv2.CornerMinEigenVal(src.AsMat(), dstMat, blockSize, ksize, borderType);
            return dstMat.numpy();
        }

        /// <summary>
        /// Harris corner detector.
        /// </summary>
        /// <param name="src">Input single-channel 8-bit or floating-point image.</param>
        /// <param name="blockSize">Neighborhood size (see the details on #cornerEigenValsAndVecs ).</param>
        /// <param name="ksize">Aperture parameter for the Sobel operator.</param>
        /// <param name="k">Harris detector free parameter. See the formula above.</param>
        /// <param name="borderType">Pixel extrapolation method. See #BorderTypes. #BORDER_WRAP is not supported.</param>
        /// <returns>Image to store the Harris detector responses.
        /// It has the type CV_32FC1 and the same size as src.</returns>
        public NDArray cornerHarris(NDArray src, int blockSize,
            int ksize, double k, BorderTypes borderType = BorderTypes.Default)
        {
            Mat dstMat = new();
            Cv2.CornerMinEigenVal(src.AsMat(), dstMat, blockSize, ksize, borderType);
            return dstMat.numpy();
        }

        /// <summary>
        /// computes both eigenvalues and the eigenvectors of 2x2 derivative covariation matrix  at each pixel. The output is stored as 6-channel matrix.
        /// </summary>
        /// <param name="src"></param>
        /// <param name="blockSize"></param>
        /// <param name="ksize"></param>
        /// <param name="borderType"></param>
        /// <returns></returns>
        public NDArray cornerEigenValsAndVecs(NDArray src, int blockSize, int ksize,
            BorderTypes borderType = BorderTypes.Default)
        {
            Mat dstMat = new();
            Cv2.CornerEigenValsAndVecs(src.AsMat(), dstMat, blockSize, ksize, borderType);
            return dstMat.numpy();
        }

        /// <summary>
        /// computes another complex cornerness criteria at each pixel
        /// </summary>
        /// <param name="src"></param>
        /// <param name="ksize"></param>
        /// <param name="borderType"></param>
        /// <returns></returns>
        public NDArray preCornerDetect(NDArray src, int ksize, BorderTypes borderType = BorderTypes.Default)
        {
            Mat dstMat = new();
            Cv2.PreCornerDetect(src.AsMat(), dstMat, ksize, borderType);
            return dstMat.numpy();
        }

        /// <summary>
        /// adjusts the corner locations with sub-pixel accuracy to maximize the certain cornerness criteria
        /// </summary>
        /// <param name="image">Input image.</param>
        /// <param name="inputCorners">Initial coordinates of the input corners and refined coordinates provided for output.</param>
        /// <param name="winSize">Half of the side length of the search window.</param>
        /// <param name="zeroZone">Half of the size of the dead region in the middle of the search zone 
        /// over which the summation in the formula below is not done. It is used sometimes to avoid possible singularities 
        /// of the autocorrelation matrix. The value of (-1,-1) indicates that there is no such a size.</param>
        /// <param name="criteria">Criteria for termination of the iterative process of corner refinement. 
        /// That is, the process of corner position refinement stops either after criteria.maxCount iterations 
        /// or when the corner position moves by less than criteria.epsilon on some iteration.</param>
        /// <returns></returns>
        public Point2f[] cornerSubPix(NDArray image, IEnumerable<Point2f> inputCorners,
            Size winSize, Size zeroZone, TermCriteria criteria)
        {
            return Cv2.CornerSubPix(image.AsMat(), inputCorners, winSize, zeroZone, criteria);
        }

        /// <summary>
        /// finds the strong enough corners where the cornerMinEigenVal() or cornerHarris() report the local maxima
        /// </summary>
        /// <param name="src">Input 8-bit or floating-point 32-bit, single-channel image.</param>
        /// <param name="maxCorners">Maximum number of corners to return. If there are more corners than are found, 
        /// the strongest of them is returned.</param>
        /// <param name="qualityLevel">Parameter characterizing the minimal accepted quality of image corners. 
        /// The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue 
        /// or the Harris function response (see cornerHarris() ). The corners with the quality measure less than 
        /// the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01, 
        /// then all the corners with the quality measure less than 15 are rejected.</param>
        /// <param name="minDistance">Minimum possible Euclidean distance between the returned corners.</param>
        /// <param name="mask">Optional region of interest. If the image is not empty
        ///  (it needs to have the type CV_8UC1 and the same size as image ), it specifies the region 
        /// in which the corners are detected.</param>
        /// <param name="blockSize">Size of an average block for computing a derivative covariation matrix over each pixel neighborhood.</param>
        /// <param name="useHarrisDetector">Parameter indicating whether to use a Harris detector</param>
        /// <param name="k">Free parameter of the Harris detector.</param>
        /// <returns>Output vector of detected corners.</returns>
        public Point2f[] goodFeaturesToTrack(NDArray src, int maxCorners, double qualityLevel, double minDistance,
            NDArray mask, int blockSize, bool useHarrisDetector, double k)
        {
            return Cv2.GoodFeaturesToTrack(src.AsMat(), maxCorners, qualityLevel, minDistance, 
                mask.AsMat(), blockSize, useHarrisDetector, k);
        }

        /// <summary>
        /// Finds lines in a binary image using standard Hough transform.
        /// </summary>
        /// <param name="image">The 8-bit, single-channel, binary source image. The image may be modified by the function</param>
        /// <param name="rho">Distance resolution of the accumulator in pixels</param>
        /// <param name="theta">Angle resolution of the accumulator in radians</param>
        /// <param name="threshold">The accumulator threshold parameter. Only those lines are returned that get enough votes ( &gt; threshold )</param>
        /// <param name="srn">For the multi-scale Hough transform it is the divisor for the distance resolution rho. [By default this is 0]</param>
        /// <param name="stn">For the multi-scale Hough transform it is the divisor for the distance resolution theta. [By default this is 0]</param>
        /// <returns>The output vector of lines. Each line is represented by a two-element vector (rho, theta) . 
        /// rho is the distance from the coordinate origin (0,0) (top-left corner of the image) and theta is the line rotation angle in radians</returns>
        public LineSegmentPolar[] houghLines(NDArray image, double rho, double theta, int threshold,
            double srn = 0, double stn = 0)
        {
            return Cv2.HoughLines(image.AsMat(), rho, theta, threshold, srn, stn);
        }

        /// <summary>
        /// Finds lines segments in a binary image using probabilistic Hough transform.
        /// </summary>
        /// <param name="image"></param>
        /// <param name="rho">Distance resolution of the accumulator in pixels</param>
        /// <param name="theta">Angle resolution of the accumulator in radians</param>
        /// <param name="threshold">The accumulator threshold parameter. Only those lines are returned that get enough votes ( &gt; threshold )</param>
        /// <param name="minLineLength">The minimum line length. Line segments shorter than that will be rejected. [By default this is 0]</param>
        /// <param name="maxLineGap">The maximum allowed gap between points on the same line to link them. [By default this is 0]</param>
        /// <returns>The output lines. Each line is represented by a 4-element vector (x1, y1, x2, y2)</returns>
        public LineSegmentPoint[] houghLinesP(NDArray image, double rho, double theta, int threshold,
            double minLineLength = 0, double maxLineGap = 0)
        {
            return Cv2.HoughLinesP(image.AsMat(), rho, theta, threshold, minLineLength, maxLineGap);
        }

        /// <summary>
        /// Finds lines in a set of points using the standard Hough transform.
        /// The function finds lines in a set of points using a modification of the Hough transform.
        /// </summary>
        /// <param name="point">Input vector of points. Each vector must be encoded as a Point vector \f$(x,y)\f$. Type must be CV_32FC2 or CV_32SC2.</param>
        /// <param name="linesMax">Max count of hough lines.</param>
        /// <param name="threshold">Accumulator threshold parameter. Only those lines are returned that get enough votes</param>
        /// <param name="minRho">Minimum Distance value of the accumulator in pixels.</param>
        /// <param name="maxRho">Maximum Distance value of the accumulator in pixels.</param>
        /// <param name="rhoStep">Distance resolution of the accumulator in pixels.</param>
        /// <param name="minTheta">Minimum angle value of the accumulator in radians.</param>
        /// <param name="maxTheta">Maximum angle value of the accumulator in radians.</param>
        /// <param name="thetaStep">Angle resolution of the accumulator in radians.</param>
        /// <returns>Output vector of found lines. Each vector is encoded as a vector&lt;Vec3d&gt;</returns>
        public NDArray houghLinesPointSet(NDArray point, int linesMax, int threshold,
            double minRho, double maxRho, double rhoStep,
            double minTheta, double maxTheta, double thetaStep)
        {
            Mat dstMat = new();
            Cv2.HoughLinesPointSet(point.AsMat(), dstMat, linesMax, threshold, 
                minRho, maxRho, rhoStep, minTheta, maxTheta, thetaStep);
            return dstMat.numpy();
        }

        /// <summary>
        /// Finds circles in a grayscale image using a Hough transform.
        /// </summary>
        /// <param name="image">The 8-bit, single-channel, grayscale input image</param>
        /// <param name="method">The available methods are HoughMethods.Gradient and HoughMethods.GradientAlt</param>
        /// <param name="dp">The inverse ratio of the accumulator resolution to the image resolution. </param>
        /// <param name="minDist">Minimum distance between the centers of the detected circles. </param>
        /// <param name="param1">The first method-specific parameter. [By default this is 100]</param>
        /// <param name="param2">The second method-specific parameter. [By default this is 100]</param>
        /// <param name="minRadius">Minimum circle radius. [By default this is 0]</param>
        /// <param name="maxRadius">Maximum circle radius. [By default this is 0] </param>
        /// <returns>The output vector found circles. Each vector is encoded as 3-element floating-point vector (x, y, radius)</returns>
        public CircleSegment[] houghCircles(NDArray image, HoughModes method, double dp, double minDist,
            double param1 = 100, double param2 = 100, int minRadius = 0, int maxRadius = 0)
        {
            return Cv2.HoughCircles(image.AsMat(), method, dp, minDist, param1, param2, minRadius, maxRadius);
        }

        /// <summary>
        /// Default borderValue for Dilate/Erode
        /// </summary>
        /// <returns></returns>
        public Scalar MorphologyDefaultBorderValue()
        {
            return Scalar.All(double.MaxValue);
        }

        /// <summary>
        /// Dilates an image by using a specific structuring element.
        /// </summary>
        /// <param name="src">The source image</param>
        /// <param name="element">The structuring element used for dilation. If element=new Mat() , a 3x3 rectangular structuring element is used</param>
        /// <param name="anchor">Position of the anchor within the element. The default value (-1, -1) means that the anchor is at the element center</param>
        /// <param name="iterations">The number of times dilation is applied. [By default this is 1]</param>
        /// <param name="borderType">The pixel extrapolation method. [By default this is BorderType.Constant]</param>
        /// <param name="borderValue">The border value in case of a constant border. The default value has a special meaning. 
        /// [By default this is CvCpp.MorphologyDefaultBorderValue()]</param>
        /// <returns>The destination image. It will have the same size and the same type as src</returns>
        public NDArray dilate(NDArray src, NDArray? element, Point? anchor = null, int iterations = 1,
            BorderTypes borderType = BorderTypes.Constant, Scalar? borderValue = null)
        {
            Mat dstMat = new();
            Cv2.Dilate(src.AsMat(), dstMat, element.ToInputArray(), anchor, iterations, 
                borderType, borderValue);
            return dstMat.numpy();
        }

        /// <summary>
        /// Erodes an image by using a specific structuring element.
        /// </summary>
        /// <param name="src">The source image</param>
        /// <param name="element">The structuring element used for dilation. If element=new Mat(), a 3x3 rectangular structuring element is used</param>
        /// <param name="anchor">Position of the anchor within the element. The default value (-1, -1) means that the anchor is at the element center</param>
        /// <param name="iterations">The number of times erosion is applied</param>
        /// <param name="borderType">The pixel extrapolation method</param>
        /// <param name="borderValue">The border value in case of a constant border. The default value has a special meaning. 
        /// [By default this is CvCpp.MorphologyDefaultBorderValue()]</param>
        /// <returns>The destination image. It will have the same size and the same type as src</returns>
        public NDArray erode(NDArray src, NDArray? element, Point? anchor = null, int iterations = 1,
            BorderTypes borderType = BorderTypes.Constant, Scalar? borderValue = null)
        {
            Mat dstMat = new();
            Cv2.Erode(src.AsMat(), dstMat, element.ToInputArray(), anchor, iterations,
                borderType, borderValue);
            return dstMat.numpy();
        }

        /// <summary>
        /// Performs advanced morphological transformations
        /// </summary>
        /// <param name="src">Source image</param>
        /// <param name="op">Type of morphological operation</param>
        /// <param name="element">Structuring element</param>
        /// <param name="anchor">Position of the anchor within the element. The default value (-1, -1) means that the anchor is at the element center</param>
        /// <param name="iterations">Number of times erosion and dilation are applied. [By default this is 1]</param>
        /// <param name="borderType">The pixel extrapolation method. [By default this is BorderType.Constant]</param>
        /// <param name="borderValue">The border value in case of a constant border. The default value has a special meaning. 
        /// [By default this is CvCpp.MorphologyDefaultBorderValue()]</param>
        /// <returns>Destination image. It will have the same size and the same type as src</returns>
        public NDArray morphologyEx(NDArray src, MorphTypes op, NDArray? element, Point? anchor = null, int iterations = 1,
            BorderTypes borderType = BorderTypes.Constant, Scalar? borderValue = null)
        {
            Mat dstMat = new();
            Cv2.MorphologyEx(src.AsMat(), dstMat, op, element.ToInputArray(), anchor, iterations,
                borderType, borderValue);
            return dstMat.numpy();
        }

        /// <summary>
        /// Resizes an image.
        /// </summary>
        /// <param name="src">input image.</param>
        /// <param name="dsize">output image size; if it equals zero, it is computed as: 
        /// dsize = Size(round(fx*src.cols), round(fy*src.rows))
        /// Either dsize or both fx and fy must be non-zero.</param>
        /// <param name="fx">scale factor along the horizontal axis; when it equals 0, 
        /// it is computed as: (double)dsize.width/src.cols</param>
        /// <param name="fy">scale factor along the vertical axis; when it equals 0, 
        /// it is computed as: (double)dsize.height/src.rows</param>
        /// <param name="interpolation">interpolation method</param>
        /// <returns>output image; it has the size dsize (when it is non-zero) or the size computed 
        /// from src.size(), fx, and fy; the type of dst is the same as of src.</returns>
        public NDArray resize(NDArray src, Size dsize,
            double fx = 0, double fy = 0, InterpolationFlags interpolation = InterpolationFlags.Linear)
        {
            Mat dstMat = new();
            Cv2.Resize(src.AsMat(), dstMat, dsize, fx, fy, interpolation);
            return dstMat.numpy();
        }

        /// <summary>
        /// Applies an affine transformation to an image.
        /// </summary>
        /// <param name="src">input image.</param>
        /// <param name="m">2x3 transformation matrix.</param>
        /// <param name="dsize">size of the output image.</param>
        /// <param name="flags">combination of interpolation methods and the optional flag 
        /// WARP_INVERSE_MAP that means that M is the inverse transformation (dst -> src) .</param>
        /// <param name="borderMode">pixel extrapolation method; when borderMode=BORDER_TRANSPARENT, 
        /// it means that the pixels in the destination image corresponding to the "outliers" 
        /// in the source image are not modified by the function.</param>
        /// <param name="borderValue">value used in case of a constant border; by default, it is 0.</param>
        /// <returns>output image that has the size dsize and the same type as src.</returns>
        public NDArray warpAffine(NDArray src, NDArray m, Size dsize,
            InterpolationFlags flags = InterpolationFlags.Linear,
            BorderTypes borderMode = BorderTypes.Constant, Scalar? borderValue = null)
        {
            Mat dstMat = new();
            Cv2.WarpAffine(src.AsMat(), dstMat, m.AsMat(), dsize, flags, borderMode, borderValue);
            return dstMat.numpy();
        }

        /// <summary>
        /// Applies a perspective transformation to an image.
        /// </summary>
        /// <param name="src">input image.</param>
        /// <param name="m">3x3 transformation matrix.</param>
        /// <param name="dsize">size of the output image.</param>
        /// <param name="flags">combination of interpolation methods (INTER_LINEAR or INTER_NEAREST) 
        /// and the optional flag WARP_INVERSE_MAP, that sets M as the inverse transformation (dst -> src).</param>
        /// <param name="borderMode">pixel extrapolation method (BORDER_CONSTANT or BORDER_REPLICATE).</param>
        /// <param name="borderValue">value used in case of a constant border; by default, it equals 0.</param>
        /// <returns>output image that has the size dsize and the same type as src.</returns>
        public NDArray warpPerspective(NDArray src, NDArray m, Size dsize,
            InterpolationFlags flags = InterpolationFlags.Linear,
            BorderTypes borderMode = BorderTypes.Constant, Scalar? borderValue = null)
        {
            Mat dstMat = new();
            Cv2.WarpPerspective(src.AsMat(), dstMat, m.AsMat(), dsize, flags, borderMode, borderValue);
            return dstMat.numpy();
        }

        /// <summary>
        /// Applies a perspective transformation to an image.
        /// </summary>
        /// <param name="src">input image.</param>
        /// <param name="m">3x3 transformation matrix.</param>
        /// <param name="dsize">size of the output image.</param>
        /// <param name="flags">combination of interpolation methods (INTER_LINEAR or INTER_NEAREST) 
        /// and the optional flag WARP_INVERSE_MAP, that sets M as the inverse transformation (dst -> src).</param>
        /// <param name="borderMode">pixel extrapolation method (BORDER_CONSTANT or BORDER_REPLICATE).</param>
        /// <param name="borderValue">value used in case of a constant border; by default, it equals 0.</param>
        /// <returns>output image that has the size dsize and the same type as src.</returns>
        public NDArray warpPerspective(NDArray src, float[,] m, Size dsize,
            InterpolationFlags flags = InterpolationFlags.Linear,
            BorderTypes borderMode = BorderTypes.Constant, Scalar? borderValue = null)
        {
            Mat dstMat = new();
            Cv2.WarpPerspective(src.AsMat(), dstMat, m, dsize, flags, borderMode, borderValue);
            return dstMat.numpy();
        }

        /// <summary>
        /// Applies a generic geometrical transformation to an image.
        /// </summary>
        /// <param name="src">Source image.</param>
        /// <param name="map1">The first map of either (x,y) points or just x values having the type CV_16SC2, CV_32FC1, or CV_32FC2.</param>
        /// <param name="map2">The second map of y values having the type CV_16UC1, CV_32FC1, or none (empty map if map1 is (x,y) points), respectively.</param>
        /// <param name="interpolation">Interpolation method. The method INTER_AREA is not supported by this function.</param>
        /// <param name="borderMode">Pixel extrapolation method. When borderMode=BORDER_TRANSPARENT, 
        /// it means that the pixels in the destination image that corresponds to the "outliers" in 
        /// the source image are not modified by the function.</param>
        /// <param name="borderValue">Value used in case of a constant border. By default, it is 0.</param>
        /// <returns>Destination image. It has the same size as map1 and the same type as src</returns>
        public NDArray remap(NDArray src, NDArray map1, NDArray map2, InterpolationFlags interpolation = InterpolationFlags.Linear,
            BorderTypes borderMode = BorderTypes.Constant, Scalar? borderValue = null)
        {
            Mat dstMat = new();
            Cv2.Remap(src.AsMat(), dstMat, map1.AsMat(), map2.AsMat(), interpolation, 
                borderMode, borderValue);
            return dstMat.numpy();
        }

        /// <summary>
        /// Converts image transformation maps from one representation to another.
        /// </summary>
        /// <param name="map1">The first input map of type CV_16SC2 , CV_32FC1 , or CV_32FC2 .</param>
        /// <param name="map2">The second input map of type CV_16UC1 , CV_32FC1 , or none (empty matrix), respectively.</param>
        /// <param name="dstmap1Type">Type of the first output map that should be CV_16SC2 , CV_32FC1 , or CV_32FC2 .</param>
        /// <param name="nnInterpolation">Flag indicating whether the fixed-point maps are used for the nearest-neighbor or for a more complex interpolation.</param>
        /// <returns>The first and second output map that has the type dstmap1type and the same size as src.</returns>
        public (NDArray, NDArray) convertMaps(NDArray map1, NDArray map2, MatType dstmap1Type, bool nnInterpolation = false)
        {
            Mat dstMat1 = new();
            Mat dstMat2 = new();
            Cv2.ConvertMaps(map1.AsMat(), map2.AsMat(), dstMat1, dstMat2,dstmap1Type, nnInterpolation);
            return (dstMat1.numpy(), dstMat2.numpy());
        }

        /// <summary>
        /// Calculates an affine matrix of 2D rotation.
        /// </summary>
        /// <param name="center">Center of the rotation in the source image.</param>
        /// <param name="angle">Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).</param>
        /// <param name="scale">Isotropic scale factor.</param>
        /// <returns></returns>
        public NDArray getRotationMatrix2D(Point2f center, double angle, double scale)
        {
            return Cv2.GetRotationMatrix2D(center, angle, scale).numpy();
        }

        /// <summary>
        /// Inverts an affine transformation.
        /// </summary>
        /// <param name="m">Original affine transformation.</param>
        /// <returns>Output reverse affine transformation.</returns>
        public NDArray invertAffineTransform(NDArray m)
        {
            Mat dstMat = new();
            Cv2.InvertAffineTransform(m.AsMat(), dstMat);
            return dstMat.numpy();
        }

        /// <summary>
        /// Calculates a perspective transform from four pairs of the corresponding points.
        /// The function calculates the 3×3 matrix of a perspective transform.
        /// </summary>
        /// <param name="src">Coordinates of quadrangle vertices in the source image.</param>
        /// <param name="dst">Coordinates of the corresponding quadrangle vertices in the destination image.</param>
        /// <returns></returns>
        public NDArray getPerspectiveTransform(IEnumerable<Point2f> src, IEnumerable<Point2f> dst)
        {
            return Cv2.GetPerspectiveTransform(src, dst).numpy();
        }

        /// <summary>
        /// Calculates a perspective transform from four pairs of the corresponding points.
        /// The function calculates the 3×3 matrix of a perspective transform.
        /// </summary>
        /// <param name="src">Coordinates of quadrangle vertices in the source image.</param>
        /// <param name="dst">Coordinates of the corresponding quadrangle vertices in the destination image.</param>
        /// <returns></returns>
        public NDArray getPerspectiveTransform(NDArray src, NDArray dst)
        {
            return Cv2.GetPerspectiveTransform(src.AsMat(), dst.AsMat()).numpy();
        }

        /// <summary>
        /// Calculates an affine transform from three pairs of the corresponding points.
        /// The function calculates the 2×3 matrix of an affine transform.
        /// </summary>
        /// <param name="src">Coordinates of triangle vertices in the source image.</param>
        /// <param name="dst">Coordinates of the corresponding triangle vertices in the destination image.</param>
        /// <returns></returns>
        public NDArray getAffineTransform(IEnumerable<Point2f> src, IEnumerable<Point2f> dst)
        {
            return Cv2.GetAffineTransform(src, dst).numpy();
        }

        /// <summary>
        /// Calculates an affine transform from three pairs of the corresponding points.
        /// The function calculates the 2×3 matrix of an affine transform.
        /// </summary>
        /// <param name="src">Coordinates of triangle vertices in the source image.</param>
        /// <param name="dst">Coordinates of the corresponding triangle vertices in the destination image.</param>
        /// <returns></returns>
        public NDArray getAffineTransform(NDArray src, NDArray dst)
        {
            return Cv2.GetAffineTransform(src.AsMat(), dst.AsMat()).numpy();
        }

        /// <summary>
        /// Retrieves a pixel rectangle from an image with sub-pixel accuracy.
        /// </summary>
        /// <param name="image">Source image.</param>
        /// <param name="patchSize">Size of the extracted patch.</param>
        /// <param name="center">Floating point coordinates of the center of the extracted rectangle 
        /// within the source image. The center must be inside the image.</param>
        /// <param name="patchType">Depth of the extracted pixels. By default, they have the same depth as src.</param>
        /// <returns>Extracted patch that has the size patchSize and the same number of channels as src .</returns>
        public NDArray getRectSubPix(NDArray image, Size patchSize, Point2f center,
            int patchType = -1)
        {
            Mat dstMat = new();
            Cv2.GetRectSubPix(image.AsMat(), patchSize, center, dstMat, patchType);
            return dstMat.numpy();
        }
    }
}
