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
        /// Creates a window.
        /// </summary>
        /// <param name="winName">Name of the window in the window caption that may be used as a window identifier.</param>
        /// <param name="flags">
        /// Flags of the window. Currently the only supported flag is CV WINDOW AUTOSIZE. If this is set, 
        /// the window size is automatically adjusted to fit the displayed image (see imshow ), and the user can not change the window size manually.
        /// </param>
        public void namedWindow(string winName, WindowFlags flags = WindowFlags.Normal)
        {
            Cv2.NamedWindow(winName, flags);
        }

        /// <summary>
        /// Destroys the specified window.
        /// </summary>
        /// <param name="winName"></param>
        public void destroyWindow(string winName)
        {
            Cv2.DestroyWindow(winName);
        }

        /// <summary>
        /// Destroys all of the HighGUI windows.
        /// </summary>
        public void destroyAllWindows()
        {
            Cv2.DestroyAllWindows();
        }

        public void startWindowThread()
        {
            Cv2.StartWindowThread();
        }

        /// <summary>
        /// Waits for a pressed key.
        /// Similar to #waitKey, but returns full key code. 
        /// Key code is implementation specific and depends on used backend: QT/GTK/Win32/etc
        /// </summary>
        /// <param name="delay">Delay in milliseconds. 0 is the special value that means ”forever”</param>
        /// <returns>Returns the code of the pressed key or -1 if no key was pressed before the specified time had elapsed.</returns>
        public void waitKeyEx(int delay = 0)
        {
            Cv2.WaitKeyEx(delay);
        }

        /// <summary>
        /// Waits for a pressed key. 
        /// </summary>
        /// <param name="delay">Delay in milliseconds. 0 is the special value that means ”forever”</param>
        /// <returns>Returns the code of the pressed key or -1 if no key was pressed before the specified time had elapsed.</returns>
        public void waitKey(int delay = 0)
        {
            Cv2.WaitKey(delay);
        }

        /// <summary>
        /// Displays the image in the specified window
        /// </summary>
        /// <param name="winName">Name of the window.</param>
        /// <param name="mat">Image to be shown.</param>
        public void imshow(string winName, NDArray mat)
        {
            Cv2.ImShow(winName, mat.AsMat());
        }

        /// <summary>
        /// Resizes window to the specified size
        /// </summary>
        /// <param name="winName">Window name</param>
        /// <param name="width">The new window width</param>
        /// <param name="height">The new window height</param>
        public void resizeWindow(string winName, int width, int height)
        {
            Cv2.ResizeWindow(winName, width, height);
        }

        /// <summary>
        /// Resizes window to the specified size
        /// </summary>
        /// <param name="winName">Window name</param>
        /// <param name="size">The new window size</param>
        public void resizeWindow(string winName, Size size)
        {
            Cv2.ResizeWindow(winName, size);
        }

        /// <summary>
        /// Moves window to the specified position
        /// </summary>
        /// <param name="winName">Window name</param>
        /// <param name="x">The new x-coordinate of the window</param>
        /// <param name="y">The new y-coordinate of the window</param>
        public void moveWindow(string winName, int x, int y)
        {
            Cv2.ResizeWindow(winName, x, y);
        }

        /// <summary>
        /// Changes parameters of a window dynamically.
        /// </summary>
        /// <param name="winName">Name of the window.</param>
        /// <param name="propId">Window property to retrieve.</param>
        /// <param name="propValue">New value of the window property.</param>
        public void setWindowProperty(string winName, WindowPropertyFlags propId, double propValue)
        {
            Cv2.SetWindowProperty(winName, propId, propValue);
        }

        /// <summary>
        /// Updates window title
        /// </summary>
        /// <param name="winName">Name of the window</param>
        /// <param name="title">New title</param>
        public void setWindowTitle(string winName, string title)
        {
            Cv2.SetWindowTitle(winName, title);
        }

        /// <summary>
        /// Provides parameters of a window.
        /// </summary>
        /// <param name="winName">Name of the window.</param>
        /// <param name="propId">Window property to retrieve.</param>
        /// <returns></returns>
        public double getWindowProperty(string winName, WindowPropertyFlags propId)
        {
            return Cv2.GetWindowProperty(winName, propId);
        }

        /// <summary>
        /// Provides rectangle of image in the window.
        /// The function getWindowImageRect returns the client screen coordinates, width and height of the image rendering area.
        /// </summary>
        /// <param name="winName">Name of the window.</param>
        /// <returns></returns>
        public Rect GetWindowImageRect(string winName)
        {
            return GetWindowImageRect(winName);
        }

        /// <summary>
        /// Sets the callback function for mouse events occuring within the specified window.
        /// </summary>
        /// <param name="windowName">Name of the window. </param>
        /// <param name="onMouse">Reference to the function to be called every time mouse event occurs in the specified window. </param>
        /// <param name="userData"></param>
        public void setMouseCallback(string windowName, MouseCallback onMouse, IntPtr userData = default)
        {
            Cv2.SetMouseCallback(windowName, onMouse, userData);
        }

        /// <summary>
        /// Gets the mouse-wheel motion delta, when handling mouse-wheel events cv::EVENT_MOUSEWHEEL and cv::EVENT_MOUSEHWHEEL.
        /// 
        /// For regular mice with a scroll-wheel, delta will be a multiple of 120. The value 120 corresponds to 
        /// a one notch rotation of the wheel or the threshold for action to be taken and one such action should 
        /// occur for each delta.Some high-precision mice with higher-resolution freely-rotating wheels may 
        /// generate smaller values. 
        /// 
        /// For cv::EVENT_MOUSEWHEEL positive and negative values mean forward and backward scrolling, 
        /// respectively.For cv::EVENT_MOUSEHWHEEL, where available, positive and negative values mean right and 
        /// left scrolling, respectively.
        /// </summary>
        /// <param name="flags">The mouse callback flags parameter.</param>
        /// <returns></returns>
        public int getMouseWheelDelta(MouseEventFlags flags)
        {
            return Cv2.GetMouseWheelDelta(flags);
        }

        /// <summary>
        /// Selects ROI on the given image.
        /// Function creates a window and allows user to select a ROI using mouse.
        /// Controls: use `space` or `enter` to finish selection, use key `c` to cancel selection (function will return the zero cv::Rect).
        /// </summary>
        /// <param name="windowName">name of the window where selection process will be shown.</param>
        /// <param name="img">image to select a ROI.</param>
        /// <param name="showCrosshair">if true crosshair of selection rectangle will be shown.</param>
        /// <param name="fromCenter">if true center of selection will match initial mouse position. In opposite case a corner of
        /// selection rectangle will correspond to the initial mouse position.</param>
        /// <returns>selected ROI or empty rect if selection canceled.</returns>
        public Rect selectROI(string windowName, NDArray img, bool showCrosshair = true, bool fromCenter = false)
        {
            return Cv2.SelectROI(windowName, img.AsMat(), showCrosshair, fromCenter);
        }

        /// <summary>
        /// Selects ROI on the given image.
        /// Function creates a window and allows user to select a ROI using mouse.
        /// Controls: use `space` or `enter` to finish selection, use key `c` to cancel selection (function will return the zero cv::Rect).
        /// </summary>
        /// <param name="img">image to select a ROI.</param>
        /// <param name="showCrosshair">if true crosshair of selection rectangle will be shown.</param>
        /// <param name="fromCenter">if true center of selection will match initial mouse position. In opposite case a corner of
        /// selection rectangle will correspond to the initial mouse position.</param>
        /// <returns>selected ROI or empty rect if selection canceled.</returns>
        public Rect selectROI(NDArray img, bool showCrosshair = true, bool fromCenter = false)
        {
            return Cv2.SelectROI(img.AsMat(), showCrosshair, fromCenter);
        }

        /// <summary>
        /// Selects ROIs on the given image.
        /// Function creates a window and allows user to select a ROIs using mouse.
        /// Controls: use `space` or `enter` to finish current selection and start a new one,
        /// use `esc` to terminate multiple ROI selection process.
        /// </summary>
        /// <param name="windowName">name of the window where selection process will be shown.</param>
        /// <param name="img">image to select a ROI.</param>
        /// <param name="showCrosshair">if true crosshair of selection rectangle will be shown.</param>
        /// <param name="fromCenter">if true center of selection will match initial mouse position. In opposite case a corner of
        /// selection rectangle will correspond to the initial mouse position.</param>
        /// <returns>selected ROIs.</returns>
        public static Rect[] selectROIs(string windowName, NDArray img, bool showCrosshair = true, bool fromCenter = false)
        {
            return Cv2.SelectROIs(windowName, img.AsMat(), showCrosshair, fromCenter);
        }

        /// <summary>
        /// Creates a trackbar and attaches it to the specified window.
        /// The function createTrackbar creates a trackbar(a slider or range control) with the specified name 
        /// and range, assigns a variable value to be a position synchronized with the trackbar and specifies 
        /// the callback function onChange to be called on the trackbar position change.The created trackbar is 
        /// displayed in the specified window winName.
        /// </summary>
        /// <param name="trackbarName">Name of the created trackbar.</param>
        /// <param name="winName">Name of the window that will be used as a parent of the created trackbar.</param>
        /// <param name="value">Optional pointer to an integer variable whose value reflects the position of the slider.Upon creation,
        ///  the slider position is defined by this variable.</param>
        /// <param name="count">Maximal position of the slider. The minimal position is always 0.</param>
        /// <param name="onChange">Pointer to the function to be called every time the slider changes position. 
        /// This function should be prototyped as void Foo(int, void\*); , where the first parameter is the trackbar 
        /// position and the second parameter is the user data(see the next parameter). If the callback is 
        /// the NULL pointer, no callbacks are called, but only value is updated.</param>
        /// <param name="userData">User data that is passed as is to the callback. It can be used to handle trackbar events without using global variables.</param>
        /// <returns></returns>
        public int createTrackbar(string trackbarName, string winName,
            ref int value, int count, TrackbarCallbackNative? onChange = null, IntPtr userData = default)
        {
            return Cv2.CreateTrackbar(trackbarName, winName, ref value, count, onChange, userData);
        }

        /// <summary>
        /// Creates a trackbar and attaches it to the specified window.
        /// The function createTrackbar creates a trackbar(a slider or range control) with the specified name 
        /// and range, assigns a variable value to be a position synchronized with the trackbar and specifies 
        /// the callback function onChange to be called on the trackbar position change.The created trackbar is 
        /// displayed in the specified window winName.
        /// </summary>
        /// <param name="trackbarName">Name of the created trackbar.</param>
        /// <param name="winName">Name of the window that will be used as a parent of the created trackbar.</param>
        /// <param name="count">Maximal position of the slider. The minimal position is always 0.</param>
        /// <param name="onChange">Pointer to the function to be called every time the slider changes position. 
        /// This function should be prototyped as void Foo(int, void\*); , where the first parameter is the trackbar 
        /// position and the second parameter is the user data(see the next parameter). If the callback is 
        /// the NULL pointer, no callbacks are called, but only value is updated.</param>
        /// <param name="userData">User data that is passed as is to the callback. It can be used to handle trackbar events without using global variables.</param>
        /// <returns></returns>
        public int createTrackbar(string trackbarName, string winName,
            int count, TrackbarCallbackNative? onChange = null, IntPtr userData = default)
        {
            return Cv2.CreateTrackbar(trackbarName, winName, count, onChange, userData);
        }

        /// <summary>
        /// Returns the trackbar position.
        /// </summary>
        /// <param name="trackbarName">Name of the trackbar.</param>
        /// <param name="winName">Name of the window that is the parent of the trackbar.</param>
        /// <returns>trackbar position</returns>
        public int getTrackbarPos(string trackbarName, string winName)
        {
            return Cv2.GetTrackbarPos(trackbarName, winName);
        }

        /// <summary>
        /// Sets the trackbar position.
        /// </summary>
        /// <param name="trackbarName">Name of the trackbar.</param>
        /// <param name="winName">Name of the window that is the parent of trackbar.</param>
        /// <param name="pos">New position.</param>
        public void setTrackbarPos(string trackbarName, string winName, int pos)
        {
            Cv2.SetTrackbarPos(trackbarName, winName, pos);
        }

        /// <summary>
        /// Sets the trackbar maximum position.
        /// The function sets the maximum position of the specified trackbar in the specified window.
        /// </summary>
        /// <param name="trackbarName">Name of the trackbar.</param>
        /// <param name="winName">Name of the window that is the parent of trackbar.</param>
        /// <param name="maxVal">New maximum position.</param>
        public void setTrackbarMax(string trackbarName, string winName, int maxVal)
        {
            Cv2.SetTrackbarMax(trackbarName, winName, maxVal);
        }

        /// <summary>
        /// Sets the trackbar minimum position.
        /// The function sets the minimum position of the specified trackbar in the specified window.
        /// </summary>
        /// <param name="trackbarName">Name of the trackbar.</param>
        /// <param name="winName">Name of the window that is the parent of trackbar.</param>
        /// <param name="minVal">New minimum position.</param>
        public void setTrackbarMin(string trackbarName, string winName, int maxVal)
        {
            Cv2.SetTrackbarMin(trackbarName, winName, maxVal);
        }

        public IntPtr getWindowHandle(string windowName)
        {
            return Cv2.GetWindowHandle(windowName);
        }
    }
}
