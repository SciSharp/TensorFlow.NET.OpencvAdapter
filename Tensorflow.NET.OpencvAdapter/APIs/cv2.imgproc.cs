using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
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

        /// <summary>
        /// Remaps an image to log-polar space.
        /// </summary>
        /// <param name="src">Source image</param>
        /// <param name="center">The transformation center; where the output precision is maximal</param>
        /// <param name="m">Magnitude scale parameter.</param>
        /// <param name="flags">A combination of interpolation methods, see cv::InterpolationFlags</param>
        /// <returns>Destination image</returns>
        public NDArray logPolar(NDArray src, Point2f center, double m, InterpolationFlags flags)
        {
            Mat dstMat = new();
            Cv2.LogPolar(src.AsMat(), dstMat, center, m, flags);
            return dstMat.numpy();
        }

        /// <summary>
        /// Remaps an image to polar space.
        /// </summary>
        /// <param name="src">Source image</param>
        /// <param name="center">The transformation center</param>
        /// <param name="maxRadius">Inverse magnitude scale parameter</param>
        /// <param name="flags">A combination of interpolation methods, see cv::InterpolationFlags</param>
        /// <returns>Destination image</returns>
        public NDArray linearPolar(NDArray src, Point2f center, double maxRadius, InterpolationFlags flags)
        {
            Mat dstMat = new();
            Cv2.LinearPolar(src.AsMat(), dstMat, center, maxRadius, flags);
            return dstMat.numpy();
        }

        /// <summary>
        /// Remaps an image to polar or semilog-polar coordinates space.
        /// </summary>
        /// <remarks>
        /// -  The function can not operate in-place.
        /// -  To calculate magnitude and angle in degrees #cartToPolar is used internally thus angles are measured from 0 to 360 with accuracy about 0.3 degrees.
        /// -  This function uses #remap. Due to current implementation limitations the size of an input and output images should be less than 32767x32767.
        /// </remarks>
        /// <param name="src">Source image.</param>
        /// <param name="dsize">The destination image size (see description for valid options).</param>
        /// <param name="center">The transformation center.</param>
        /// <param name="maxRadius">The radius of the bounding circle to transform. It determines the inverse magnitude scale parameter too.</param>
        /// <param name="interpolationFlags">interpolation methods.</param>
        /// <param name="warpPolarMode">interpolation methods.</param>
        /// <returns>Destination image. It will have same type as src.</returns>
        public NDArray warpPolar(NDArray src, Size dsize, Point2f center, double maxRadius, 
            InterpolationFlags interpolationFlags, WarpPolarMode warpPolarMode)
        {
            Mat dstMat = new();
            Cv2.WarpPolar(src.AsMat(), dstMat, dsize, center, maxRadius, interpolationFlags, warpPolarMode);
            return dstMat.numpy();
        }

        /// <summary>
        /// Calculates the integral of an image.
        /// The function calculates one or more integral images for the source image.
        /// </summary>
        /// <param name="src"></param>
        /// <param name="sdepth"></param>
        /// <returns>sum and sqsum</returns>
        public (NDArray, NDArray) integral(NDArray src, int sdepth = -1)
        {
            Mat sumMat = new();
            Mat sqsumMat = new();
            Cv2.Integral(src.AsMat(), sumMat, sqsumMat, sdepth);
            return (sumMat.numpy(), sqsumMat.numpy());
        }

        /// <summary>
        /// Calculates the integral of an image.
        /// The function calculates one or more integral images for the source image.
        /// </summary>
        /// <param name="src">input image as W×H, 8-bit or floating-point (32f or 64f).</param>
        /// <param name="sdepth">desired depth of the integral and the tilted integral images, CV_32S, CV_32F, or CV_64F.</param>
        /// <param name="sqdepth">desired depth of the integral image of squared pixel values, CV_32F or CV_64F.</param>
        /// <returns>sum, sqsum and titled, which respectively mean: 1. integral image as (W+1)×(H+1) , 32-bit integer or floating-point (32f or 64f). 
        /// 2. integral image for squared pixel values; it is (W+1)×(H+1), double-precision floating-point (64f) array. 
        /// 3. integral for the image rotated by 45 degrees; it is (W+1)×(H+1) array with the same data type as sum.</returns>
        public (NDArray, NDArray, NDArray) integral(NDArray src, int sdepth = -1, int sqdepth = -1)
        {
            Mat sumMat = new();
            Mat sqsumMat = new();
            Mat titledMat = new();
            Cv2.Integral(src.AsMat(), sumMat, sqsumMat, titledMat, sdepth, sqdepth);
            return (sumMat.numpy(), sqsumMat.numpy(), titledMat.numpy());
        }

        /// <summary>
        /// Adds an image to the accumulator.
        /// </summary>
        /// <param name="src">Input image as 1- or 3-channel, 8-bit or 32-bit floating point.</param>
        /// <param name="dst">Accumulator image with the same number of channels as input image, 32-bit or 64-bit floating-point.</param>
        /// <param name="mask">Optional operation mask.</param>
        /// <returns>dst</returns>
        public NDArray accumulate(NDArray src, NDArray? dst, NDArray mask)
        {
            if (dst is not null && !dst.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "dst needs that. Please consider change the adapter mode.");
            }
            var dstMat = dst is null ? new Mat() : dst.AsMat();
            Cv2.Accumulate(src.AsMat(), dstMat, mask.AsMat());
            if(dst is not null)
            {
                return dst;
            }
            else
            {
                return dstMat.numpy();
            }
        }

        /// <summary>
        /// Adds the square of a source image to the accumulator.
        /// </summary>
        /// <param name="src">Input image as 1- or 3-channel, 8-bit or 32-bit floating point.</param>
        /// <param name="dst">Accumulator image with the same number of channels as input image, 32-bit or 64-bit floating-point.</param>
        /// <param name="mask">Optional operation mask.</param>
        /// <returns></returns>
        public NDArray accumulateSquare(NDArray src, NDArray? dst, NDArray mask)
        {
            if (dst is not null && !dst.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "dst needs that. Please consider change the adapter mode.");
            }
            var dstMat = dst is null ? new Mat() : dst.AsMat();
            Cv2.AccumulateSquare(src.AsMat(), dstMat, mask.AsMat());
            if (dst is not null)
            {
                return dst;
            }
            else
            {
                return dstMat.numpy();
            }
        }

        /// <summary>
        /// Adds the per-element product of two input images to the accumulator.
        /// </summary>
        /// <param name="src1">First input image, 1- or 3-channel, 8-bit or 32-bit floating point.</param>
        /// <param name="src2">Second input image of the same type and the same size as src1</param>
        /// <param name="dst">Accumulator with the same number of channels as input images, 32-bit or 64-bit floating-point.</param>
        /// <param name="mask">Optional operation mask.</param>
        /// <returns>dst</returns>
        public NDArray accumulateProduct(NDArray src1, NDArray src2, NDArray? dst, NDArray mask)
        {
            if (dst is not null && !dst.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "dst needs that. Please consider change the adapter mode.");
            }
            var dstMat = dst is null ? new Mat() : dst.AsMat();
            Cv2.AccumulateProduct(src1.AsMat(), src2.AsMat(), dstMat, mask.AsMat());
            if (dst is not null)
            {
                return dst;
            }
            else
            {
                return dstMat.numpy();
            }
        }

        /// <summary>
        /// Updates a running average.
        /// </summary>
        /// <param name="src">Input image as 1- or 3-channel, 8-bit or 32-bit floating point.</param>
        /// <param name="dst">Accumulator image with the same number of channels as input image, 32-bit or 64-bit floating-point.</param>
        /// <param name="alpha">Weight of the input image.</param>
        /// <param name="mask">Optional operation mask.</param>
        /// <returns>dst</returns>
        public NDArray accumulateWeighted(NDArray src, NDArray? dst, double alpha, NDArray mask)
        {
            if (dst is not null && !dst.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "dst needs that. Please consider change the adapter mode.");
            }
            var dstMat = dst is null ? new Mat() : dst.AsMat();
            Cv2.AccumulateWeighted(src.AsMat(), dstMat, alpha, mask.AsMat());
            if (dst is not null)
            {
                return dst;
            }
            else
            {
                return dstMat.numpy();
            }
        }

        /// <summary>
        /// The function is used to detect translational shifts that occur between two images.
        /// 
        /// The operation takes advantage of the Fourier shift theorem for detecting the translational shift in
        /// the frequency domain.It can be used for fast image registration as well as motion estimation.
        /// For more information please see http://en.wikipedia.org/wiki/Phase_correlation.
        /// 
        /// Calculates the cross-power spectrum of two supplied source arrays. The arrays are padded if needed with getOptimalDFTSize.
        /// </summary>
        /// <param name="src1">Source floating point array (CV_32FC1 or CV_64FC1)</param>
        /// <param name="src2">Source floating point array (CV_32FC1 or CV_64FC1)</param>
        /// <param name="window">Floating point array with windowing coefficients to reduce edge effects (optional).</param>
        /// <returns>detected phase shift(sub-pixel) between the two arrays, and response, which is signal power within 
        /// the 5x5 centroid around the peak, between 0 and 1 (optional).</returns>
        public (Point2d, double) phaseCorrelate(NDArray src1, NDArray src2, NDArray window)
        {
            var retVal = Cv2.PhaseCorrelate(src1.AsMat(), src2.AsMat(), window.AsMat(), out var response);
            return (retVal, response);
        }

        /// <summary>
        /// Computes a Hanning window coefficients in two dimensions.
        /// </summary>
        /// <param name="dst">Destination array to place Hann coefficients in</param>
        /// <param name="winSize">The window size specifications</param>
        /// <param name="type">Created array type</param>
        /// <returns>dst</returns>
        public NDArray createHanningWindow(NDArray dst, Size winSize, MatType type)
        {
            if (dst is not null && !dst.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "dst needs that. Please consider change the adapter mode.");
            }
            var dstMat = dst is null ? new Mat() : dst.AsMat();
            Cv2.CreateHanningWindow(dstMat, winSize, type);
            if (dst is not null)
            {
                return dst;
            }
            else
            {
                return dstMat.numpy();
            }
        }

        /// <summary>
        /// Applies a fixed-level threshold to each array element.
        /// </summary>
        /// <param name="src">input array (single-channel, 8-bit or 32-bit floating point).</param>
        /// <param name="thresh">threshold value.</param>
        /// <param name="maxval">maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.</param>
        /// <param name="type">thresholding type (see the details below).</param>
        /// <returns>the computed threshold value when type == OTSU and output array of the same size and type as src</returns>
        public (double, NDArray) threshold(NDArray src, double thresh, double maxval, ThresholdTypes type)
        {
            Mat dstMat = new();
            var retVal = Cv2.Threshold(src.AsMat(), dstMat, thresh, maxval, type);
            return (retVal, dstMat.numpy());
        }

        /// <summary>
        /// Applies an adaptive threshold to an array.
        /// </summary>
        /// <param name="src">Source 8-bit single-channel image.</param>
        /// <param name="maxval">Non-zero value assigned to the pixels for which the condition is satisfied. See the details below.</param>
        /// <param name="adaptiveMethod">Adaptive thresholding algorithm to use, ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C .</param>
        /// <param name="thresholdType">Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV .</param>
        /// <param name="blockSize">Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.</param>
        /// <param name="c">Constant subtracted from the mean or weighted mean (see the details below). 
        /// Normally, it is positive but may be zero or negative as well.</param>
        /// <returns>Destination image of the same size and the same type as src</returns>
        public NDArray adaptiveThreshold(NDArray src, double maxval, AdaptiveThresholdTypes adaptiveMethod, 
            ThresholdTypes thresholdType, int blockSize, double c)
        {
            Mat dstMat = new();
            Cv2.AdaptiveThreshold(src.AsMat(), dstMat, maxval, adaptiveMethod, thresholdType, blockSize, c);
            return dstMat.numpy();
        }

        /// <summary>
        /// Blurs an image and downsamples it.
        /// </summary>
        /// <param name="src">input image.</param>
        /// <param name="dstSize">size of the output image; by default, it is computed as Size((src.cols+1)/2</param>
        /// <param name="borderType"></param>
        /// <returns>output image; it has the specified size and the same type as src.</returns>
        public NDArray pyrDown(NDArray src, Size? dstSize = null, BorderTypes borderType = BorderTypes.Default)
        {
            Mat dstMat = new();
            Cv2.PyrDown(src.AsMat(), dstMat, dstSize, borderType);
            return dstMat.numpy();
        }

        /// <summary>
        /// Upsamples an image and then blurs it.
        /// </summary>
        /// <param name="src">input image.</param>
        /// <param name="dstSize">size of the output image; by default, it is computed as Size(src.cols*2, (src.rows*2)</param>
        /// <param name="borderType"></param>
        /// <returns>output image. It has the specified size and the same type as src.</returns>
        public NDArray pyrUp(NDArray src, Size? dstSize = null, BorderTypes borderType = BorderTypes.Default)
        {
            Mat dstMat = new();
            Cv2.PyrUp(src.AsMat(), dstMat, dstSize, borderType);
            return dstMat.numpy();
        }

        /// <summary>
        /// Creates a predefined CLAHE object
        /// </summary>
        /// <param name="clipLimit"></param>
        /// <param name="tileGridSize"></param>
        /// <returns></returns>
        public CLAHE CreateCLAHE(double clipLimit = 40.0, Size? tileGridSize = null)
        {
            return Cv2.CreateCLAHE(clipLimit, tileGridSize);
        }

        /// <summary>
        /// Computes the "minimal work" distance between two weighted point configurations.
        ///
        /// The function computes the earth mover distance and/or a lower boundary of the distance between the
        /// two weighted point configurations.One of the applications described in @cite RubnerSept98,
        /// @cite Rubner2000 is multi-dimensional histogram comparison for image retrieval.EMD is a transportation
        /// problem that is solved using some modification of a simplex algorithm, thus the complexity is
        /// exponential in the worst case, though, on average it is much faster.In the case of a real metric
        /// the lower boundary can be calculated even faster (using linear-time algorithm) and it can be used
        /// to determine roughly whether the two signatures are far enough so that they cannot relate to the same object.
        /// </summary>
        /// <param name="signature1">First signature, a \f$\texttt{size1}\times \texttt{dims}+1\f$ floating-point matrix. 
        /// Each row stores the point weight followed by the point coordinates.The matrix is allowed to have
        /// a single column(weights only) if the user-defined cost matrix is used.The weights must be non-negative
        /// and have at least one non-zero value.</param>
        /// <param name="signature2">Second signature of the same format as signature1 , though the number of rows
        /// may be different.The total weights may be different.In this case an extra "dummy" point is added
        /// to either signature1 or signature2. The weights must be non-negative and have at least one non-zero value.</param>
        /// <param name="distType">Used metric.</param>
        public float EMD(NDArray signature1, NDArray signature2, DistanceTypes distType)
        {
            return Cv2.EMD(signature1.AsMat(), signature2.AsMat(), distType);
        }

        /// <summary>
        /// Computes the "minimal work" distance between two weighted point configurations.
        ///
        /// The function computes the earth mover distance and/or a lower boundary of the distance between the
        /// two weighted point configurations.One of the applications described in @cite RubnerSept98,
        /// @cite Rubner2000 is multi-dimensional histogram comparison for image retrieval.EMD is a transportation
        /// problem that is solved using some modification of a simplex algorithm, thus the complexity is
        /// exponential in the worst case, though, on average it is much faster.In the case of a real metric
        /// the lower boundary can be calculated even faster (using linear-time algorithm) and it can be used
        /// to determine roughly whether the two signatures are far enough so that they cannot relate to the same object.
        /// </summary>
        /// <param name="signature1">First signature, a \f$\texttt{size1}\times \texttt{dims}+1\f$ floating-point matrix. 
        /// Each row stores the point weight followed by the point coordinates.The matrix is allowed to have
        /// a single column(weights only) if the user-defined cost matrix is used.The weights must be non-negative
        /// and have at least one non-zero value.</param>
        /// <param name="signature2">Second signature of the same format as signature1 , though the number of rows
        /// may be different.The total weights may be different.In this case an extra "dummy" point is added
        /// to either signature1 or signature2. The weights must be non-negative and have at least one non-zero value.</param>
        /// <param name="distType">Used metric.</param>
        /// <param name="cost">User-defined size1 x size2 cost matrix. Also, if a cost matrix
        /// is used, lower boundary lowerBound cannot be calculated because it needs a metric function.</param>
        /// <returns></returns>
        public float EMD(NDArray signature1, NDArray signature2, DistanceTypes distType, NDArray? cost)
        {
            return Cv2.EMD(signature1.AsMat(), signature2.AsMat(), distType, cost.ToInputArray());
        }

        /// <summary>
        /// Computes the "minimal work" distance between two weighted point configurations.
        ///
        /// The function computes the earth mover distance and/or a lower boundary of the distance between the
        /// two weighted point configurations.One of the applications described in @cite RubnerSept98,
        /// @cite Rubner2000 is multi-dimensional histogram comparison for image retrieval.EMD is a transportation
        /// problem that is solved using some modification of a simplex algorithm, thus the complexity is
        /// exponential in the worst case, though, on average it is much faster.In the case of a real metric
        /// the lower boundary can be calculated even faster (using linear-time algorithm) and it can be used
        /// to determine roughly whether the two signatures are far enough so that they cannot relate to the same object.
        /// </summary>
        /// <param name="signature1">First signature, a \f$\texttt{size1}\times \texttt{dims}+1\f$ floating-point matrix. 
        /// Each row stores the point weight followed by the point coordinates.The matrix is allowed to have
        /// a single column(weights only) if the user-defined cost matrix is used.The weights must be non-negative
        /// and have at least one non-zero value.</param>
        /// <param name="signature2">Second signature of the same format as signature1 , though the number of rows
        /// may be different.The total weights may be different.In this case an extra "dummy" point is added
        /// to either signature1 or signature2. The weights must be non-negative and have at least one non-zero value.</param>
        /// <param name="distType">Used metric.</param>
        /// <param name="cost">User-defined size1 x size2 cost matrix. Also, if a cost matrix
        /// is used, lower boundary lowerBound cannot be calculated because it needs a metric function.</param>
        /// <param name="flow">Resultant size1 x size2 flow matrix: flow[i,j] is  a flow from i-th point of signature1
        /// to j-th point of signature2.</param>
        /// <returns>The second return value is `lowerBound`, which is the lower boundary of a distance between the two
        /// signatures that is a distance between mass centers.The lower boundary may not be calculated if
        /// the user-defined cost matrix is used, the total weights of point configurations are not equal, or
        /// if the signatures consist of weights only(the signature matrices have a single column). You ** must**
        /// initialize \*lowerBound.If the calculated distance between mass centers is greater or equal to
        /// \*lowerBound(it means that the signatures are far enough), the function does not calculate EMD.
        /// In any case \*lowerBound is set to the calculated distance between mass centers on return.
        /// Thus, if you want to calculate both distance between mass centers and EMD, \*lowerBound should be set to 0.</returns>
        public (float, float) EMD(NDArray signature1, NDArray signature2, DistanceTypes distType, NDArray? cost, NDArray? flow = null)
        {
            var retVal = Cv2.EMD(signature1.AsMat(), signature2.AsMat(), distType, cost.ToInputArray(), out var lowerBound, flow?.AsMat());
            return (retVal, lowerBound);
        }

        /// <summary>
        /// Performs a marker-based image segmentation using the watershed algorithm.
        /// </summary>
        /// <param name="image">Input 8-bit 3-channel image.</param>
        /// <param name="markers">Input/output 32-bit single-channel image (map) of markers. 
        /// It should have the same size as image.</param>
        /// <returns>markers</returns>
        public NDArray waterShed(NDArray image, NDArray markers)
        {
            if (!markers.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "waterShed needs that. Please consider change the adapter mode.");
            }
            Cv2.Watershed(image.AsMat(), markers.AsMat());
            return markers;
        }

        /// <summary>
        /// Performs initial step of meanshift segmentation of an image.
        /// </summary>
        /// <param name="src">The source 8-bit, 3-channel image.</param>
        /// <param name="sp">The spatial window radius.</param>
        /// <param name="sr">The color window radius.</param>
        /// <param name="maxLevel">Maximum level of the pyramid for the segmentation.</param>
        /// <param name="termcrit">Termination criteria: when to stop meanshift iterations.</param>
        /// <returns>The destination image of the same format and the same size as the source.</returns>
        public NDArray pyrMeanShiftFiltering(NDArray src, double sp, double sr, int maxLevel = 1, TermCriteria? termcrit = null)
        {
            Mat dstMat = new();
            Cv2.PyrMeanShiftFiltering(src.AsMat(), dstMat, sp, sr, maxLevel, termcrit);
            return dstMat.numpy();
        }

        /// <summary>
        /// Segments the image using GrabCut algorithm
        /// </summary>
        /// <param name="img">Input 8-bit 3-channel image.</param>
        /// <param name="mask">Input/output 8-bit single-channel mask. 
        /// The mask is initialized by the function when mode is set to GC_INIT_WITH_RECT. 
        /// Its elements may have Cv2.GC_BGD / Cv2.GC_FGD / Cv2.GC_PR_BGD / Cv2.GC_PR_FGD</param>
        /// <param name="rect">ROI containing a segmented object. The pixels outside of the ROI are 
        /// marked as "obvious background". The parameter is only used when mode==GC_INIT_WITH_RECT.</param>
        /// <param name="bgdModel">Temporary array for the background model. Do not modify it while you are processing the same image.</param>
        /// <param name="fgdModel">Temporary arrays for the foreground model. Do not modify it while you are processing the same image.</param>
        /// <param name="iterCount">Number of iterations the algorithm should make before returning the result. 
        /// Note that the result can be refined with further calls with mode==GC_INIT_WITH_MASK or mode==GC_EVAL .</param>
        /// <param name="mode">Operation mode that could be one of GrabCutFlag value.</param>
        /// <returns>(mask, bgdModel, fgdModel)</returns>
        public (NDArray, NDArray, NDArray) gradCut(NDArray img, NDArray mask, Rect rect, NDArray bgdModel, 
            NDArray fgdModel, int iterCount, GrabCutModes mode)
        {
            if (!mask.CanConvertToMatWithouyCopy() || !bgdModel.CanConvertToMatWithouyCopy()
                || !fgdModel.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "gradCut needs that. Please consider change the adapter mode.");
            }
            Cv2.GrabCut(img.AsMat(), mask.AsMat(), rect, bgdModel.AsMat(), fgdModel.AsMat(), iterCount, mode);
            return (mask, bgdModel, fgdModel);
        }

        /// <summary>
        /// Fills a connected component with the given color.
        /// </summary>
        /// <param name="image">Input/output 1- or 3-channel, 8-bit, or floating-point image. 
        /// It is modified by the function unless the FLOODFILL_MASK_ONLY flag is set in the 
        /// second variant of the function. See the details below.</param>
        /// <param name="seedPoint">Starting point.</param>
        /// <param name="newVal">New value of the repainted domain pixels.</param>
        /// <returns></returns>
        public (int, NDArray) floodFill(NDArray image, Point seedPoint, Scalar newVal)
        {
            if (!image.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "floodFill needs that. Please consider change the adapter mode.");
            }
            var retVal = Cv2.FloodFill(image.AsMat(), seedPoint, newVal);
            return (retVal, image);
        }

        /// <summary>
        /// Fills a connected component with the given color.
        /// </summary>
        /// <param name="image">Input/output 1- or 3-channel, 8-bit, or floating-point image. 
        /// It is modified by the function unless the FLOODFILL_MASK_ONLY flag is set in the 
        /// second variant of the function. See the details below.</param>
        /// <param name="seedPoint">Starting point.</param>
        /// <param name="newVal">New value of the repainted domain pixels.</param>
        /// <param name="loDiff">Maximal lower brightness/color difference between the currently 
        /// observed pixel and one of its neighbors belonging to the component, or a seed pixel 
        /// being added to the component.</param>
        /// <param name="upDiff">Maximal upper brightness/color difference between the currently 
        /// observed pixel and one of its neighbors belonging to the component, or a seed pixel 
        /// being added to the component.</param>
        /// <param name="flags">Operation flags. Lower bits contain a connectivity value, 
        /// 4 (default) or 8, used within the function. Connectivity determines which 
        /// neighbors of a pixel are considered. Using FloodFillFlags.MaskOnly will
        /// fill in the mask using the grey value 255 (white). </param>
        /// <returns>The second return value is Optional output parameter set by the function to the 
        /// minimum bounding rectangle of the repainted domain. The third return value is image.</returns>
        public (int, Rect, NDArray) floodFill(NDArray image, Point seedPoint, Scalar newVal,
            Scalar? loDiff = null, Scalar? upDiff = null, FloodFillFlags flags = FloodFillFlags.Link4)
        {
            if (!image.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "floodFill needs that. Please consider change the adapter mode.");
            }
            var retVal = Cv2.FloodFill(image.AsMat(), seedPoint, newVal, out var rect, loDiff, upDiff, flags);
            return (retVal, rect, image);
        }

        /// <summary>
        /// Fills a connected component with the given color.
        /// </summary>
        /// <param name="image">Input/output 1- or 3-channel, 8-bit, or floating-point image. 
        /// It is modified by the function unless the FLOODFILL_MASK_ONLY flag is set in the 
        /// second variant of the function. See the details below.</param>
        /// <param name="mask">(For the second function only) Operation mask that should be a single-channel 8-bit image, 
        /// 2 pixels wider and 2 pixels taller. The function uses and updates the mask, so you take responsibility of 
        /// initializing the mask content. Flood-filling cannot go across non-zero pixels in the mask. For example, 
        /// an edge detector output can be used as a mask to stop filling at edges. It is possible to use the same mask 
        /// in multiple calls to the function to make sure the filled area does not overlap.</param>
        /// <param name="seedPoint">Starting point.</param>
        /// <param name="newVal">New value of the repainted domain pixels.</param>
        /// <returns></returns>
        public (int, NDArray, NDArray) floodFill(NDArray image, NDArray mask, Point seedPoint, Scalar newVal)
        {
            if (!image.CanConvertToMatWithouyCopy() || !mask.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "floodFill needs that. Please consider change the adapter mode.");
            }
            var retVal = Cv2.FloodFill(image.AsMat(), mask.AsMat(), seedPoint, newVal);
            return (retVal, image, mask);
        }

        /// <summary>
        /// Fills a connected component with the given color.
        /// </summary>
        /// <param name="image">Input/output 1- or 3-channel, 8-bit, or floating-point image. 
        /// It is modified by the function unless the FLOODFILL_MASK_ONLY flag is set in the 
        /// second variant of the function. See the details below.</param>
        /// <param name="mask">(For the second function only) Operation mask that should be a single-channel 8-bit image, 
        /// 2 pixels wider and 2 pixels taller. The function uses and updates the mask, so you take responsibility of 
        /// initializing the mask content. Flood-filling cannot go across non-zero pixels in the mask. For example, 
        /// an edge detector output can be used as a mask to stop filling at edges. It is possible to use the same mask 
        /// in multiple calls to the function to make sure the filled area does not overlap.</param>
        /// <param name="seedPoint">Starting point.</param>
        /// <param name="newVal">New value of the repainted domain pixels.</param>
        /// <param name="loDiff">Maximal lower brightness/color difference between the currently 
        /// observed pixel and one of its neighbors belonging to the component, or a seed pixel 
        /// being added to the component.</param>
        /// <param name="upDiff">Maximal upper brightness/color difference between the currently 
        /// observed pixel and one of its neighbors belonging to the component, or a seed pixel 
        /// being added to the component.</param>
        /// <param name="flags">Operation flags. Lower bits contain a connectivity value, 
        /// 4 (default) or 8, used within the function. Connectivity determines which 
        /// neighbors of a pixel are considered. Using FloodFillFlags.MaskOnly will
        /// fill in the mask using the grey value 255 (white). </param>
        /// <returns>The second return value is Optional output parameter set by the function to the 
        /// minimum bounding rectangle of the repainted domain. The third return value is image. 
        /// The forth return value is maek.</returns>
        public (int, Rect, NDArray, NDArray) floodFill(NDArray image, NDArray mask, Point seedPoint, Scalar newVal,
            Scalar? loDiff = null, Scalar? upDiff = null, FloodFillFlags flags = FloodFillFlags.Link4)
        {
            if (!image.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "floodFill needs that. Please consider change the adapter mode.");
            }
            var retVal = Cv2.FloodFill(image.AsMat(), mask.AsMat(), seedPoint, newVal, out var rect, loDiff, upDiff, flags);
            return (retVal, rect, image, mask);
        }

        /// <summary>
        /// Performs linear blending of two images:
        /// dst(i,j) = weights1(i,j)*src1(i,j) + weights2(i,j)*src2(i,j)
        /// </summary>
        /// <param name="src1">It has a type of CV_8UC(n) or CV_32FC(n), where n is a positive integer.</param>
        /// <param name="src2">It has the same type and size as src1.</param>
        /// <param name="weights1">It has a type of CV_32FC1 and the same size with src1.</param>
        /// <param name="weights2">It has a type of CV_32FC1 and the same size with src1.</param>
        /// <returns>It is created if it does not have the same size and type with src1.</returns>
        public NDArray blendLinear(NDArray src1, NDArray src2, NDArray weights1, NDArray weights2)
        {
            Mat dstMat = new();
            Cv2.BlendLinear(src1.AsMat(), src2.AsMat(), weights1.AsMat(), weights2.AsMat(), dstMat);
            return dstMat.numpy();
        }

        /// <summary>
        /// Converts image from one color space to another
        /// </summary>
        /// <param name="src">The source image, 8-bit unsigned, 16-bit unsigned or single-precision floating-point</param>
        /// <param name="code">The color space conversion code</param>
        /// <param name="dstCn">The number of channels in the destination image; if the parameter is 0, the number of the 
        /// channels will be derived automatically from src and the code</param>
        /// <returns>The destination image; will have the same size and the same depth as src</returns>
        public NDArray cvtColor(NDArray src, ColorConversionCodes code, int dstCn = 0)
        {
            Mat dstMat = new();
            Cv2.CvtColor(src.AsMat(), dstMat, code, dstCn);
            return dstMat.numpy();
        }

        /// <summary>
        /// Converts an image from one color space to another where the source image is stored in two planes.
        /// This function only supports YUV420 to RGB conversion as of now.
        /// </summary>
        /// <param name="src1">8-bit image (#CV_8U) of the Y plane.</param>
        /// <param name="src2">image containing interleaved U/V plane.</param>
        /// <param name="code">Specifies the type of conversion. It can take any of the following values:
        /// - #COLOR_YUV2BGR_NV12
        /// - #COLOR_YUV2RGB_NV12
        /// - #COLOR_YUV2BGRA_NV12
        /// - #COLOR_YUV2RGBA_NV12
        /// - #COLOR_YUV2BGR_NV21
        /// - #COLOR_YUV2RGB_NV21
        /// - #COLOR_YUV2BGRA_NV21
        /// - #COLOR_YUV2RGBA_NV21</param>
        /// <returns>output image</returns>
        public NDArray cvtColorTwoPlane(NDArray src1, NDArray src2, ColorConversionCodes code)
        {
            Mat dstMat = new();
            Cv2.CvtColorTwoPlane(src1.AsMat(), src2.AsMat(), dstMat, code);
            return dstMat.numpy();
        }

        /// <summary>
        /// Applies a GNU Octave/MATLAB equivalent colormap on a given image.
        /// </summary>
        /// <param name="src">The source image, grayscale or colored of type CV_8UC1 or CV_8UC3.</param>
        /// <param name="colormap">colormap The colormap to apply</param>
        /// <returns>The result is the colormapped source image. Note: Mat::create is called on dst.</returns>
        public NDArray applyColorMap(NDArray src, NDArray userColor)
        {
            Mat dstMat = new();
            Cv2.ApplyColorMap(src.AsMat(), dstMat, userColor.AsMat());
            return dstMat.numpy();
        }

        /// <summary>
        /// Applies a GNU Octave/MATLAB equivalent colormap on a given image.
        /// </summary>
        /// <param name="src">The source image, grayscale or colored of type CV_8UC1 or CV_8UC3.</param>
        /// <param name="userColor">The colormap to apply of type CV_8UC1 or CV_8UC3 and size 256</param>
        /// <returns>The result is the colormapped source image. Note: Mat::create is called on dst.</returns>
        public NDArray applyColorMap(NDArray src, ColormapTypes colormap)
        {
            Mat dstMat = new();
            Cv2.ApplyColorMap(src.AsMat(), dstMat, colormap);
            return dstMat.numpy();
        }

        /// <summary>
        /// Draws a line segment connecting two points
        /// </summary>
        /// <param name="img">The image. </param>
        /// <param name="pt1X">First point's x-coordinate of the line segment. </param>
        /// <param name="pt1Y">First point's y-coordinate of the line segment. </param>
        /// <param name="pt2X">Second point's x-coordinate of the line segment. </param>
        /// <param name="pt2Y">Second point's y-coordinate of the line segment. </param>
        /// <param name="color">Line color. </param>
        /// <param name="thickness">Line thickness. [By default this is 1]</param>
        /// <param name="lineType">Type of the line. [By default this is LineType.Link8]</param>
        /// <param name="shift">Number of fractional bits in the point coordinates. [By default this is 0]</param>
        /// <returns>img</returns>
        public NDArray line(NDArray img, int pt1X, int pt1Y, int pt2X, int pt2Y, Scalar color,
            int thickness = 1, LineTypes lineType = LineTypes.Link8, int shift = 0)
        {
            if (!img.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "line needs that. Please consider change the adapter mode.");
            }
            Cv2.Line(img.AsMat(), pt1X, pt1Y, pt2X, pt2Y, color, thickness, lineType, shift);
            return img;
        }

        /// <summary>
        /// Draws a line segment connecting two points
        /// </summary>
        /// <param name="img">The image. </param>
        /// <param name="pt1">First point of the line segment. </param>
        /// <param name="pt2">Second point of the line segment. </param>
        /// <param name="color">Line color. </param>
        /// <param name="thickness">Line thickness. [By default this is 1]</param>
        /// <param name="lineType">Type of the line. [By default this is LineType.Link8]</param>
        /// <param name="shift">Number of fractional bits in the point coordinates. [By default this is 0]</param>
        /// <returns>img</returns>
        public NDArray line(NDArray img, Point pt1, Point pt2, Scalar color, int thickness = 1,
            LineTypes lineType = LineTypes.Link8, int shift = 0)
        {
            if (!img.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "line needs that. Please consider change the adapter mode.");
            }
            Cv2.Line(img.AsMat(), pt1, pt2, color, thickness, lineType, shift);
            return img;
        }

        /// <summary>
        /// Draws a arrow segment pointing from the first point to the second one.
        /// The function arrowedLine draws an arrow between pt1 and pt2 points in the image. 
        /// See also cv::line.
        /// </summary>
        /// <param name="img">Image.</param>
        /// <param name="pt1">The point the arrow starts from.</param>
        /// <param name="pt2">The point the arrow points to.</param>
        /// <param name="color">Line color.</param>
        /// <param name="thickness">Line thickness.</param>
        /// <param name="lineType">Type of the line, see cv::LineTypes</param>
        /// <param name="shift">Number of fractional bits in the point coordinates.</param>
        /// <param name="tipLength">The length of the arrow tip in relation to the arrow length</param>
        /// <returns>img</returns>
        public NDArray arrowedLine(NDArray img, Point pt1, Point pt2, Scalar color, int thickness = 1,
            LineTypes lineType = LineTypes.Link8, int shift = 0, double tipLength = 0.1)
        {
            if (!img.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "line needs that. Please consider change the adapter mode.");
            }
            Cv2.ArrowedLine(img.AsMat(), pt1, pt2, color, thickness, lineType, shift, tipLength);
            return img;
        }

        /// <summary>
        /// Draws simple, thick or filled rectangle
        /// </summary>
        /// <param name="img">Image. </param>
        /// <param name="pt1">One of the rectangle vertices. </param>
        /// <param name="pt2">Opposite rectangle vertex. </param>
        /// <param name="color">Line color (RGB) or brightness (grayscale image). </param>
        /// <param name="thickness">Thickness of lines that make up the rectangle. Negative values make the function to draw a filled rectangle. [By default this is 1]</param>
        /// <param name="lineType">Type of the line, see cvLine description. [By default this is LineType.Link8]</param>
        /// <param name="shift">Number of fractional bits in the point coordinates. [By default this is 0]</param>
        /// <returns>img</returns>
        public NDArray rectangle(NDArray img, Point pt1, Point pt2, Scalar color, int thickness = 1,
            LineTypes lineType = LineTypes.Link8, int shift = 0)
        {
            if (!img.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "line needs that. Please consider change the adapter mode.");
            }
            Cv2.Rectangle(img.AsMat(), pt1, pt2, color, thickness, lineType, shift);
            return img;
        }

        /// <summary>
        /// Draws simple, thick or filled rectangle
        /// </summary>
        /// <param name="img">Image. </param>
        /// <param name="rect">Rectangle.</param>
        /// <param name="color">Line color (RGB) or brightness (grayscale image). </param>
        /// <param name="thickness">Thickness of lines that make up the rectangle.
        /// Negative values make the function to draw a filled rectangle. [By default this is 1]</param>
        /// <param name="lineType">Type of the line, see cvLine description. [By default this is LineType.Link8]</param>
        /// <param name="shift">Number of fractional bits in the point coordinates. [By default this is 0]</param>
        /// <returns>img</returns>
        public NDArray rectangle(NDArray img, Rect rect, Scalar color, int thickness = 1,
            LineTypes lineType = LineTypes.Link8, int shift = 0)
        {
            if (!img.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "line needs that. Please consider change the adapter mode.");
            }
            Cv2.Rectangle(img.AsMat(), rect, color, thickness, lineType, shift);
            return img;
        }

        /// <summary>
        /// Draws a circle
        /// </summary>
        /// <param name="img">Image where the circle is drawn. </param>
        /// <param name="centerX">X-coordinate of the center of the circle. </param>
        /// <param name="centerY">Y-coordinate of the center of the circle. </param>
        /// <param name="radius">Radius of the circle. </param>
        /// <param name="color">Circle color. </param>
        /// <param name="thickness">Thickness of the circle outline if positive, otherwise indicates that a filled circle has to be drawn. [By default this is 1]</param>
        /// <param name="lineType">Type of the circle boundary. [By default this is LineType.Link8]</param>
        /// <param name="shift">Number of fractional bits in the center coordinates and radius value. [By default this is 0]</param>
        /// <returns>img</returns>
        public NDArray circle(NDArray img, int centerX, int centerY, int radius, Scalar color,
            int thickness = 1, LineTypes lineType = LineTypes.Link8, int shift = 0)
        {
            if (!img.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "line needs that. Please consider change the adapter mode.");
            }
            Cv2.Circle(img.AsMat(), centerX, centerY, radius, color, thickness, lineType, shift);
            return img;
        }

        /// <summary>
        /// Draws a circle
        /// </summary>
        /// <param name="img">Image where the circle is drawn. </param>
        /// <param name="center">Center of the circle. </param>
        /// <param name="radius">Radius of the circle. </param>
        /// <param name="color">Circle color. </param>
        /// <param name="thickness">Thickness of the circle outline if positive, otherwise indicates that a filled circle has to be drawn. [By default this is 1]</param>
        /// <param name="lineType">Type of the circle boundary. [By default this is LineType.Link8]</param>
        /// <param name="shift">Number of fractional bits in the center coordinates and radius value. [By default this is 0]</param>
        /// <returns>img</returns>
        public NDArray circle(NDArray img, Point center, int radius, Scalar color,
            int thickness = 1, LineTypes lineType = LineTypes.Link8, int shift = 0)
        {
            if (!img.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "line needs that. Please consider change the adapter mode.");
            }
            Cv2.Circle(img.AsMat(), center, radius, color, thickness, lineType, shift);
            return img;
        }

        /// <summary>
        /// Draws simple or thick elliptic arc or fills ellipse sector
        /// </summary>
        /// <param name="img">Image. </param>
        /// <param name="center">Center of the ellipse. </param>
        /// <param name="axes">Length of the ellipse axes. </param>
        /// <param name="angle">Rotation angle. </param>
        /// <param name="startAngle">Starting angle of the elliptic arc. </param>
        /// <param name="endAngle">Ending angle of the elliptic arc. </param>
        /// <param name="color">Ellipse color. </param>
        /// <param name="thickness">Thickness of the ellipse arc. [By default this is 1]</param>
        /// <param name="lineType">Type of the ellipse boundary. [By default this is LineType.Link8]</param>
        /// <param name="shift">Number of fractional bits in the center coordinates and axes' values. [By default this is 0]</param>
        /// <returns>img</returns>
        public NDArray ellipse(NDArray img, Point center, Size axes, double angle, double startAngle, double endAngle, Scalar color,
            int thickness = 1, LineTypes lineType = LineTypes.Link8, int shift = 0)
        {
            if (!img.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "line needs that. Please consider change the adapter mode.");
            }
            Cv2.Ellipse(img.AsMat(), center, axes, angle, startAngle, endAngle, color, thickness, lineType, shift);
            return img;
        }

        /// <summary>
        /// Draws simple or thick elliptic arc or fills ellipse sector
        /// </summary>
        /// <param name="img">Image. </param>
        /// <param name="box">The enclosing box of the ellipse drawn </param>
        /// <param name="color">Ellipse color. </param>
        /// <param name="thickness">Thickness of the ellipse boundary. [By default this is 1]</param>
        /// <param name="lineType">Type of the ellipse boundary. [By default this is LineType.Link8]</param>
        /// <returns>img</returns>
        public NDArray ellipse(NDArray img, RotatedRect box, Scalar color,
            int thickness = 1, LineTypes lineType = LineTypes.Link8)
        {
            if (!img.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "line needs that. Please consider change the adapter mode.");
            }
            Cv2.Ellipse(img.AsMat(), box, color, thickness, lineType);
            return img;
        }

        /// <summary>
        /// Draws a marker on a predefined position in an image.
        ///
        /// The function cv::drawMarker draws a marker on a given position in the image.For the moment several
        /// marker types are supported, see #MarkerTypes for more information.
        /// </summary>
        /// <param name="img">Image.</param>
        /// <param name="position">The point where the crosshair is positioned.</param>
        /// <param name="color">Line color.</param>
        /// <param name="markerType">The specific type of marker you want to use.</param>
        /// <param name="markerSize">The length of the marker axis [default = 20 pixels]</param>
        /// <param name="thickness">Line thickness.</param>
        /// <param name="lineType">Type of the line.</param>
        /// <returns>img</returns>
        public NDArray drawMarker(NDArray img, Point position, Scalar color, MarkerTypes markerType = MarkerTypes.Cross, 
            int markerSize = 20, int thickness = 1, LineTypes lineType = LineTypes.Link8)
        {
            if (!img.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "line needs that. Please consider change the adapter mode.");
            }
            Cv2.DrawMarker(img.AsMat(), position, color, markerType, markerSize, thickness, lineType);
            return img;
        }

        /// <summary>
        /// Fills a convex polygon.
        /// </summary>
        /// <param name="img">Image</param>
        /// <param name="pts">The polygon vertices</param>
        /// <param name="color">Polygon color</param>
        /// <param name="lineType">Type of the polygon boundaries</param>
        /// <param name="shift">The number of fractional bits in the vertex coordinates</param>
        /// <returns>img</returns>
        public NDArray fillConvexPoly(NDArray img, NDArray pts, Scalar color,
            LineTypes lineType = LineTypes.Link8, int shift = 0)
        {
            if (!img.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "line needs that. Please consider change the adapter mode.");
            }
            Cv2.FillConvexPoly(img.AsMat(), pts.AsMat(), color, lineType, shift);
            return img;
        }

        /// <summary>
        /// Fills the area bounded by one or more polygons
        /// </summary>
        /// <param name="img">Image</param>
        /// <param name="pts">Array of polygons, each represented as an array of points</param>
        /// <param name="color">Polygon color</param>
        /// <param name="lineType">Type of the polygon boundaries</param>
        /// <param name="shift">The number of fractional bits in the vertex coordinates</param>
        /// <param name="offset"></param>
        /// <returns>img</returns>
        public NDArray fillPoly(NDArray img, NDArray pts, Scalar color,
            LineTypes lineType = LineTypes.Link8, int shift = 0, Point? offset = null)
        {
            if (!img.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "line needs that. Please consider change the adapter mode.");
            }
            Cv2.FillPoly(img.AsMat(), pts.AsMat(), color, lineType, shift, offset);
            return img;
        }

        /// <summary>
        /// draws one or more polygonal curves
        /// </summary>
        /// <param name="img"></param>
        /// <param name="pts"></param>
        /// <param name="isClosed"></param>
        /// <param name="color"></param>
        /// <param name="thickness"></param>
        /// <param name="lineType"></param>
        /// <param name="shift"></param>
        /// <returns>img</returns>
        public NDArray polylines(NDArray img, NDArray pts, bool isClosed, Scalar color,
            int thickness = 1, LineTypes lineType = LineTypes.Link8, int shift = 0)
        {
            if (!img.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "line needs that. Please consider change the adapter mode.");
            }
            Cv2.Polylines(img.AsMat(), pts.AsMat(), isClosed, color, thickness, lineType, shift);
            return img;
        }

        /// <summary>
        /// draws contours in the image
        /// </summary>
        /// <param name="image">Destination image.</param>
        /// <param name="contours">All the input contours. Each contour is stored as a point vector.</param>
        /// <param name="contourIdx">Parameter indicating a contour to draw. If it is negative, all the contours are drawn.</param>
        /// <param name="color">Color of the contours.</param>
        /// <param name="thickness">Thickness of lines the contours are drawn with. If it is negative (for example, thickness=CV_FILLED ), 
        /// the contour interiors are drawn.</param>
        /// <param name="lineType">Line connectivity. </param>
        /// <param name="hierarchy">Optional information about hierarchy. It is only needed if you want to draw only some of the contours</param>
        /// <param name="maxLevel">Maximal level for drawn contours. If it is 0, only the specified contour is drawn. 
        /// If it is 1, the function draws the contour(s) and all the nested contours. If it is 2, the function draws the contours, 
        /// all the nested contours, all the nested-to-nested contours, and so on. This parameter is only taken into account 
        /// when there is hierarchy available.</param>
        /// <param name="offset">Optional contour shift parameter. Shift all the drawn contours by the specified offset = (dx, dy)</param>
        /// <returns>image</returns>
        public NDArray drawContours(NDArray image, IEnumerable<NDArray> contours,
            int contourIdx,
            Scalar color,
            int thickness = 1,
            LineTypes lineType = LineTypes.Link8,
            Mat? hierarchy = null,
            int maxLevel = int.MaxValue,
            Point? offset = null)
        {
            if (!image.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "line needs that. Please consider change the adapter mode.");
            }
            Cv2.DrawContours(image.AsMat(), contours.Select(x => x.AsMat()), contourIdx, color, thickness, 
                lineType, hierarchy, maxLevel, offset);
            return image;
        }

        /// <summary>
        /// Clips the line against the image rectangle
        /// </summary>
        /// <param name="imgSize">The image size</param>
        /// <param name="pt1">The first line point</param>
        /// <param name="pt2">The second line point</param>
        /// <returns></returns>
        public bool clipLine(Size imgSize, ref Point pt1, ref Point pt2)
        {
            return Cv2.ClipLine(imgSize, ref pt1, ref pt2);
        }

        /// <summary>
        /// Clips the line against the image rectangle
        /// </summary>
        /// <param name="imgRect">sThe image rectangle</param>
        /// <param name="pt1">The first line point</param>
        /// <param name="pt2">The second line point</param>
        /// <returns></returns>
        public bool clipLine(Rect imgRect, ref Point pt1, ref Point pt2)
        {
            return Cv2.ClipLine(imgRect, ref pt1, ref pt2);
        }

        /// <summary>
        /// Approximates an elliptic arc with a polyline.
        /// The function ellipse2Poly computes the vertices of a polyline that 
        /// approximates the specified elliptic arc. It is used by cv::ellipse.
        /// </summary>
        /// <param name="center">Center of the arc.</param>
        /// <param name="axes">Half of the size of the ellipse main axes. See the ellipse for details.</param>
        /// <param name="angle">Rotation angle of the ellipse in degrees. See the ellipse for details.</param>
        /// <param name="arcStart">Starting angle of the elliptic arc in degrees.</param>
        /// <param name="arcEnd">Ending angle of the elliptic arc in degrees.</param>
        /// <param name="delta">Angle between the subsequent polyline vertices. It defines the approximation</param>
        /// <returns>Output vector of polyline vertices.</returns>
        public Point[] ellipse2Poly(Point center, Size axes, int angle,
            int arcStart, int arcEnd, int delta)
        {
            return Cv2.Ellipse2Poly(center, axes, angle, arcStart, arcEnd, delta);
        }

        /// <summary>
        /// Approximates an elliptic arc with a polyline.
        /// The function ellipse2Poly computes the vertices of a polyline that 
        /// approximates the specified elliptic arc. It is used by cv::ellipse.
        /// </summary>
        /// <param name="center">Center of the arc.</param>
        /// <param name="axes">Half of the size of the ellipse main axes. See the ellipse for details.</param>
        /// <param name="angle">Rotation angle of the ellipse in degrees. See the ellipse for details.</param>
        /// <param name="arcStart">Starting angle of the elliptic arc in degrees.</param>
        /// <param name="arcEnd">Ending angle of the elliptic arc in degrees.</param>
        /// <param name="delta">Angle between the subsequent polyline vertices. It defines the approximation</param>
        /// <returns>Output vector of polyline vertices.</returns>
        public Point2d[] ellipse2Poly(Point2d center, Size2d axes, int angle,
            int arcStart, int arcEnd, int delta)
        {
            return Cv2.Ellipse2Poly(center, axes, angle, arcStart, arcEnd, delta);
        }

        /// <summary>
        /// renders text string in the image
        /// </summary>
        /// <param name="img">Image.</param>
        /// <param name="text">Text string to be drawn.</param>
        /// <param name="org">Bottom-left corner of the text string in the image.</param>
        /// <param name="fontFace">Font type, see #HersheyFonts.</param>
        /// <param name="fontScale">Font scale factor that is multiplied by the font-specific base size.</param>
        /// <param name="color">Text color.</param>
        /// <param name="thickness">Thickness of the lines used to draw a text.</param>
        /// <param name="lineType">Line type. See #LineTypes</param>
        /// <param name="bottomLeftOrigin">When true, the image data origin is at the bottom-left corner.
        /// Otherwise, it is at the top-left corner.</param>
        /// <returns>img</returns>
        public NDArray putText(NDArray img, string text, Point org,
            HersheyFonts fontFace, double fontScale, Scalar color,
            int thickness = 1, LineTypes lineType = LineTypes.Link8, bool bottomLeftOrigin = false)
        {
            if (!img.CanConvertToMatWithouyCopy())
            {
                throw new ValueError("Cannot convert the NDArray to Mat without copy but the method " +
                    "line needs that. Please consider change the adapter mode.");
            }
            Cv2.PutText(img.AsMat(), text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin);
            return img;
        }

        /// <summary>
        /// returns bounding box of the text string
        /// </summary>
        /// <param name="text">Input text string.</param>
        /// <param name="fontFace">Font to use, see #HersheyFonts.</param>
        /// <param name="fontScale">Font scale factor that is multiplied by the font-specific base size.</param>
        /// <param name="thickness">Thickness of lines used to render the text. See #putText for details.</param>
        /// <param name="baseLine">baseLine y-coordinate of the baseline relative to the bottom-most text</param>
        /// <returns>The size of a box that contains the specified text.</returns>
        public Size getTextSize(string text, HersheyFonts fontFace,
            double fontScale, int thickness, out int baseLine)
        {
            return Cv2.GetTextSize(text, fontFace, fontScale, thickness, out baseLine);
        }

        /// <summary>
        /// Calculates the font-specific size to use to achieve a given height in pixels.
        /// </summary>
        /// <param name="fontFace">Font to use, see cv::HersheyFonts.</param>
        /// <param name="pixelHeight">Pixel height to compute the fontScale for</param>
        /// <param name="thickness">Thickness of lines used to render the text.See putText for details.</param>
        /// <returns>The fontSize to use for cv::putText</returns>
        public double getFontScaleFromHeight(HersheyFonts fontFace, int pixelHeight, int thickness = 1)
        {
            return Cv2.GetFontScaleFromHeight(fontFace, pixelHeight, thickness);
        }
    }
}
