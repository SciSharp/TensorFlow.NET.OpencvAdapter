using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.NumPy;
using Tensorflow.OpencvAdapter.Extensions;

namespace Tensorflow.OpencvAdapter.APIs
{
    public partial class Cv2API
    {
        /// <summary>
        /// Restores the selected region in an image using the region neighborhood.
        /// </summary>
        /// <param name="src">Input 8-bit, 16-bit unsigned or 32-bit float 1-channel or 8-bit 3-channel image.</param>
        /// <param name="inpaintMask">Inpainting mask, 8-bit 1-channel image. Non-zero pixels indicate the area that needs to be inpainted.</param>
        /// <param name="inpaintRadius">Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.</param>
        /// <param name="flags">Inpainting method that could be cv::INPAINT_NS or cv::INPAINT_TELEA</param>
        /// <returns>Output image with the same size and type as src.</returns>
        public NDArray inpaint(NDArray src, NDArray inpaintMask, double inpaintRadius, InpaintMethod flags)
        {
            Mat dstMat = new();
            Cv2.Inpaint(src.AsMat(), inpaintMask.AsMat(), dstMat, inpaintRadius, flags);
            return dstMat.numpy();
        }

        /// <summary>
        /// Perform image denoising using Non-local Means Denoising algorithm 
        /// with several computational optimizations. Noise expected to be a gaussian white noise
        /// </summary>
        /// <param name="src">Input 8-bit 1-channel, 2-channel or 3-channel image.</param>
        /// <param name="h">
        /// Parameter regulating filter strength. Big h value perfectly removes noise but also removes image details, 
        /// smaller h value preserves details but also preserves some noise</param>
        /// <param name="templateWindowSize">
        /// Size in pixels of the template patch that is used to compute weights. Should be odd. Recommended value 7 pixels</param>
        /// <param name="searchWindowSize">
        /// Size in pixels of the window that is used to compute weighted average for given pixel. 
        /// Should be odd. Affect performance linearly: greater searchWindowsSize - greater denoising time. Recommended value 21 pixels</param>
        /// <returns>Output image with the same size and type as src .</returns>
        public NDArray fastNlMeansDenoising(NDArray src, float h = 3, int templateWindowSize = 7, int searchWindowSize = 21)
        {
            Mat dstMat = new();
            Cv2.FastNlMeansDenoising(src.AsMat(), dstMat, h, templateWindowSize, searchWindowSize);
            return dstMat.numpy();
        }

        /// <summary>
        /// Modification of fastNlMeansDenoising function for colored images
        /// </summary>
        /// <param name="src">Input 8-bit 3-channel image.</param>
        /// <param name="h">Parameter regulating filter strength for luminance component. 
        /// Bigger h value perfectly removes noise but also removes image details, smaller h value preserves details but also preserves some noise</param>
        /// <param name="hColor">The same as h but for color components. For most images value equals 10 will be enought 
        /// to remove colored noise and do not distort colors</param>
        /// <param name="templateWindowSize">
        /// Size in pixels of the template patch that is used to compute weights. Should be odd. Recommended value 7 pixels</param>
        /// <param name="searchWindowSize">
        /// Size in pixels of the window that is used to compute weighted average for given pixel. Should be odd. 
        /// Affect performance linearly: greater searchWindowsSize - greater denoising time. Recommended value 21 pixels</param>
        /// <returns>Output image with the same size and type as src.</returns>
        public NDArray fastNlMeansDenoisingColored(NDArray src, float h = 3, float hColor = 3,
            int templateWindowSize = 7, int searchWindowSize = 21)
        {
            Mat dstMat = new();
            Cv2.FastNlMeansDenoisingColored(src.AsMat(), dstMat, h, hColor, templateWindowSize, searchWindowSize);
            return dstMat.numpy();
        }

        /// <summary>
        /// Modification of fastNlMeansDenoising function for images sequence where consequtive images have been captured 
        /// in small period of time. For example video. This version of the function is for grayscale images or for manual manipulation with colorspaces.
        /// </summary>
        /// <param name="srcImgs">Input 8-bit 1-channel, 2-channel or 3-channel images sequence. All images should have the same type and size.</param>
        /// <param name="imgToDenoiseIndex">Target image to denoise index in srcImgs sequence</param>
        /// <param name="temporalWindowSize">Number of surrounding images to use for target image denoising. 
        /// Should be odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to imgToDenoiseIndex - temporalWindowSize / 2 
        /// from srcImgs will be used to denoise srcImgs[imgToDenoiseIndex] image.</param>
        /// <param name="h">Parameter regulating filter strength for luminance component. Bigger h value perfectly removes noise but also removes image details, 
        /// smaller h value preserves details but also preserves some noise</param>
        /// <param name="templateWindowSize">Size in pixels of the template patch that is used to compute weights. Should be odd. Recommended value 7 pixels</param>
        /// <param name="searchWindowSize">Size in pixels of the window that is used to compute weighted average for given pixel. 
        /// Should be odd. Affect performance linearly: greater searchWindowsSize - greater denoising time. Recommended value 21 pixels</param>
        /// <returns> Output image with the same size and type as srcImgs images.</returns>
        public NDArray fastNlMeansDenoisingMulti(IEnumerable<NDArray> srcImgs, int imgToDenoiseIndex, int temporalWindowSize,
            float h = 3, int templateWindowSize = 7, int searchWindowSize = 21)
        {
            Mat dstMat = new();
            Cv2.FastNlMeansDenoisingMulti(srcImgs.Select(x => x.AsMat()), dstMat, imgToDenoiseIndex, temporalWindowSize, 
                h, templateWindowSize, searchWindowSize);
            return dstMat.numpy();
        }

        /// <summary>
        /// Primal-dual algorithm is an algorithm for solving special types of variational problems 
        /// (that is, finding a function to minimize some functional). As the image denoising, 
        /// in particular, may be seen as the variational problem, primal-dual algorithm then 
        /// can be used to perform denoising and this is exactly what is implemented.
        /// </summary>
        /// <param name="observations">This array should contain one or more noised versions 
        /// of the image that is to be restored.</param>
        /// <param name="lambda">Corresponds to \f$\lambda\f$ in the formulas above. 
        /// As it is enlarged, the smooth (blurred) images are treated more favorably than 
        /// detailed (but maybe more noised) ones. Roughly speaking, as it becomes smaller, 
        /// the result will be more blur but more sever outliers will be removed.</param>
        /// <param name="niters"> Number of iterations that the algorithm will run. 
        /// Of course, as more iterations as better, but it is hard to quantitatively 
        /// refine this statement, so just use the default and increase it if the results are poor.</param>
        /// <returns>Here the denoised image will be stored. There is no need to 
        /// do pre-allocation of storage space, as it will be automatically allocated, if necessary.</returns>
        public NDArray denoiseTVL1(IEnumerable<NDArray> observations, double lambda = 1.0, int niters = 30)
        {
            Mat dstMat = new();
            Cv2.DenoiseTVL1(observations.Select(x => x.AsMat()), dstMat, lambda, niters);
            return dstMat.numpy();
        }

        /// <summary>
        /// Transforms a color image to a grayscale image. It is a basic tool in digital 
        /// printing, stylized black-and-white photograph rendering, and in many single 
        /// channel image processing applications @cite CL12 .
        /// </summary>
        /// <param name="src">Input 8-bit 3-channel image.</param>
        /// <returns>grayscale Mat and color boost Mat</returns>
        public (NDArray, NDArray) decolor(NDArray src)
        {
            Mat grayscaleMat = new();
            Mat colorBoostMat = new();
            Cv2.Decolor(src.AsMat(), grayscaleMat, colorBoostMat);
            return (grayscaleMat.numpy(), colorBoostMat.numpy());
        }

        /// <summary>
        /// Image editing tasks concern either global changes (color/intensity corrections, 
        /// filters, deformations) or local changes concerned to a selection. Here we are 
        /// interested in achieving local changes, ones that are restricted to a region 
        /// manually selected (ROI), in a seamless and effortless manner. The extent of 
        /// the changes ranges from slight distortions to complete replacement by novel 
        /// content @cite PM03 .
        /// </summary>
        /// <param name="src">Input 8-bit 3-channel image.</param>
        /// <param name="dst">Input 8-bit 3-channel image.</param>
        /// <param name="mask">Input 8-bit 1 or 3-channel image.</param>
        /// <param name="p">Point in dst image where object is placed.</param>
        /// <param name="flags">Cloning method</param>
        /// <returns>Output image with the same size and type as dst.</returns>
        public NDArray seamlessClone(NDArray src, NDArray dst, NDArray? mask, Point p, SeamlessCloneMethods flags)
        {
            Mat blendMat = new();
            Cv2.SeamlessClone(src.AsMat(), dst.AsMat(), mask.ToInputArray(), p, blendMat, flags);
            return blendMat.numpy();
        }

        /// <summary>
        /// Given an original color image, two differently colored versions of this 
        /// image can be mixed seamlessly. Multiplication factor is between 0.5 to 2.5.
        /// </summary>
        /// <param name="src">Input 8-bit 3-channel image.</param>
        /// <param name="mask">Input 8-bit 1 or 3-channel image.</param>
        /// <param name="redMul">R-channel multiply factor.</param>
        /// <param name="greenMul">G-channel multiply factor.</param>
        /// <param name="blueMul">B-channel multiply factor.</param>
        /// <returns>Output image with the same size and type as src.</returns>
        public NDArray colorChange(NDArray src, NDArray? mask, float redMul = 1.0f, float greenMul = 1.0f, float blueMul = 1.0f)
        {
            Mat dstMat = new();
            Cv2.ColorChange(src.AsMat(), mask.ToInputArray(), dstMat, redMul, greenMul, blueMul);
            return dstMat.numpy();
        }

        /// <summary>
        /// Applying an appropriate non-linear transformation to the gradient field inside 
        /// the selection and then integrating back with a Poisson solver, modifies locally 
        /// the apparent illumination of an image.
        /// </summary>
        /// <param name="src">Input 8-bit 3-channel image.</param>
        /// <param name="mask">Input 8-bit 1 or 3-channel image.</param>
        /// <param name="alpha">Value ranges between 0-2.</param>
        /// <param name="beta">Value ranges between 0-2.</param>
        /// <returns>Output image with the same size and type as src.</returns>
        /// <remarks>
        /// This is useful to highlight under-exposed foreground objects or to reduce specular reflections.
        /// </remarks>
        public NDArray illuminationChange(NDArray src, NDArray? mask, float alpha = 0.2f, float beta = 0.4f)
        {
            Mat dstMat = new();
            Cv2.IlluminationChange(src.AsMat(), mask.ToInputArray(), dstMat, alpha, beta);
            return dstMat.numpy();
        }

        /// <summary>
        /// By retaining only the gradients at edge locations, before integrating with the 
        /// Poisson solver, one washes out the texture of the selected region, giving its 
        /// contents a flat aspect. Here Canny Edge Detector is used.
        /// </summary>
        /// <param name="src">Input 8-bit 3-channel image.</param>
        /// <param name="mask">Input 8-bit 1 or 3-channel image.</param>
        /// <param name="lowThreshold">Range from 0 to 100.</param>
        /// <param name="highThreshold">Value &gt; 100.</param>
        /// <param name="kernelSize">The size of the Sobel kernel to be used.</param>
        /// <returns>Output image with the same size and type as src.</returns>
        public NDArray textureFlattening(NDArray src, NDArray? mask, float lowThreshold = 30, float highThreshold = 45,
            int kernelSize = 3)
        {
            Mat dstMat = new();
            Cv2.TextureFlattening(src.AsMat(), mask.ToInputArray(), dstMat, lowThreshold, highThreshold, kernelSize);
            return dstMat.numpy();
        }

        /// <summary>
        /// Filtering is the fundamental operation in image and video processing. 
        /// Edge-preserving smoothing filters are used in many different applications @cite EM11 .
        /// </summary>
        /// <param name="src">Input 8-bit 3-channel image.</param>
        /// <param name="flags">Edge preserving filters</param>
        /// <param name="sigmaS">Range between 0 to 200.</param>
        /// <param name="sigmaR">Range between 0 to 1.</param>
        /// <returns>Output 8-bit 3-channel image.</returns>
        public NDArray edgePreservingFilter(NDArray src, EdgePreservingMethods flags = EdgePreservingMethods.RecursFilter,
            float sigmaS = 60, float sigmaR = 0.4f)
        {
            Mat dstMat = new();
            Cv2.EdgePreservingFilter(src.AsMat(), dstMat, flags, sigmaS, sigmaR);
            return dstMat.numpy();
        }

        /// <summary>
        /// This filter enhances the details of a particular image.
        /// </summary>
        /// <param name="src">Input 8-bit 3-channel image.</param>
        /// <param name="dst">Output image with the same size and type as src.</param>
        /// <param name="sigmaS">Range between 0 to 200.</param>
        /// <param name="sigmaR">Range between 0 to 1.</param>
        public NDArray detailEnhance(NDArray src, float sigmaS = 60, float sigmaR = 0.4f)
        {
            Mat dstMat = new();
            Cv2.DetailEnhance(src.AsMat(), dstMat, sigmaS, sigmaR);
            return dstMat.numpy();
        }

        /// <summary>
        /// Pencil-like non-photorealistic line drawing
        /// </summary>
        /// <param name="src">Input 8-bit 3-channel image.</param>
        /// <param name="sigmaS">Range between 0 to 200.</param>
        /// <param name="sigmaR">Range between 0 to 1.</param>
        /// <param name="shadeFactor">Range between 0 to 0.1.</param>
        /// <returns>output 8-bit 1-channel image and output image with the same size and type as src</returns>
        public (NDArray, NDArray) pencilSketch(NDArray src, float sigmaS = 60, float sigmaR = 0.07f, float shadeFactor = 0.02f)
        {
            Mat dst1Mat = new();
            Mat dst2Mat = new();
            Cv2.PencilSketch(src.AsMat(), dst1Mat, dst2Mat, sigmaS, sigmaR, shadeFactor);
            return (dst1Mat.numpy(), dst2Mat.numpy());
        }

        /// <summary>
        /// Stylization aims to produce digital imagery with a wide variety of effects 
        /// not focused on photorealism. Edge-aware filters are ideal for stylization, 
        /// as they can abstract regions of low contrast while preserving, or enhancing, 
        /// high-contrast features.
        /// </summary>
        /// <param name="src">Input 8-bit 3-channel image.</param>
        /// <param name="sigmaS">Range between 0 to 200.</param>
        /// <param name="sigmaR">Range between 0 to 1.</param>
        /// <returns>Output image with the same size and type as src.</returns>
        public NDArray stylization(NDArray src, float sigmaS = 60, float sigmaR = 0.45f)
        {
            Mat dstMat = new();
            Cv2.Stylization(src.AsMat(), dstMat, sigmaS, sigmaR);
            return dstMat.numpy();
        }
    }
}
