# TensorFlow.OpencvAdapter
An efficient library which enables using tensorflow.net with opencvsharp. It reuses the memory of Mat to provide a good performance.

## Introduction

[Tensorflow.NET](https://github.com/SciSharp/TensorFlow.NET) is the dotnet binding of tensorflow, which is a deep-learning framework. [OpencvSharp](https://github.com/shimat/opencvsharp) is a good framework to provide opencv APIs in .NET. 

Tensorflow.NET uses `NDArray/Tensor` as data structure, while OpencvSharp uses `Mat` as data structure. This once became a gap between Tensorflow.NET and OpencvSharp, causing some inconvenience for CV works with Tensorflow.NET.

The aim of Tensorflow.OpencvAdapter is to make the two libraries compatible, and provide an extra set API with Tensorflow.NET style. With Tensorflow.OpencvAdapter, a `Mat` can be converted to a NDArray without memory copying and vice versa.

## Usages

There are currently two ways to use Tensorflow.OpencvAdapter to combine Tensorflow.NET and OpencvSharp:

1. Do all the opencv manipulations with `Mat` and finnaly convert them to NDArrays (without copying):

```cs
Mat m = ...
NDArray array1 = m.ToNDArray(copy: false); // C# style API
NDArray array2 = m.numpy(); // python style API
Mat n1 = array1.AsMat(); // Convert back to Mat without copying
Mat n2 = array1.ToMat(copy: true); // Convert back to Mat with copying
```

2. Use the cv2 APIs provided in `Tensorflow.OpencvAdapter`, which are in Tensorflow.NET style (python style). In this way the abstraction of `Mat` can be hided to some degrees.

```cs
using static Tensorflow.OpencvAPIs;

NDArray img = cv2.imread("xxx.jpg");
img = cv2.resize(img, new Size(2, 3));
```

## Installation

Currently the Tensorflow.OpencvAdapter has not been published. It will be published along with `Tensorflow.NET v1.0.0`.

## API progress rate

- [x] Conversion between `Mat` and `NDArray`
- [x] cv2.core APIs
- [x] cv2.imgproc APIs
- [x] cv2.photo APIs
- [x] cv2.video APIs
- [x] cv2.video APIs
- [x] cv2.imgcodecs APIs
- [x] cv2.objdetect APIs
- [x] cv2.highgui APIs
- [ ] cv2.features2d APIs
- [ ] cv2.calib3d APIs
