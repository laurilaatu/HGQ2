HGQ2: High Granularity Quantization 2
=============================================

[![LGPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0.en.html)
[![Documentation](https://github.com/calad0i/HGQ2/actions/workflows/sphinx-build.yml/badge.svg)](https://calad0i.github.io/HGQ2/)

This is a refactored version of the [HGQ](https://github.com/calad0i/HGQ) library: a quantization-aware training framework targeting realtime deep learning applications. Besides all the features provided by the original HGQ library, this version includes the following improvements:

- **Scalability**: HGQ2 is built on Keras v3 with all layers with proper supports for all backends: TensorFlow, JAX, and PyTorch. As XLA compilation is also supported, which can significantly speed up the training process. Besides GPU acceleration, HGQ2 also supports TPU acceleration for TensorFlow and JAX backends. Training speed on HGQ2 can be 1.2-5 times faster than the original HGQ library, depending on the model and the backend.
- **Flexibility**: Effective Bit-Operations (EBOP) based resource estimation can now be turned off, and cross layer talking is fully eliminated by moving the datalane quantizer location. This allows the user to mix HGQ2 layers with vanilla Keras layers without any restrictions. (Use with caution though, if you want to put the final model on hardware!)
- **Quantizers**:
  - _Fixed-point_: While the original HGQ library only optimizes the number of floating bits with one way of parameterizing the fixed-point numbers, HGQ2 supports multiple ways of parametrizing them, and allows of optimizing any part of them via gradients.
  - _Minifloat_: Training with minifloat quantization is supported, also with surrogate gradients support (alpha quality).

- **More Layers**: HGQ2 supports more layers than the original HGQ library, including the powerful `EinsumDense(BatchNorm)` layer and the `MultiHeadAttention` layer with bit-accurate softmax and scaled dot-product attention (alpha quality).


## Installation

```bash
pip install -e . # Install HGQ2 as local editable package. If you want to install it as a regular package, remove the `-e` flag.
```

You will need to use `hls4ml` from this [fork](https://github.com/calad0i/hls4ml/tree/da4ml-v2) to generate the HLS projects. Minifloat quantization is not supported by the backend yet.

## Usage

Please refer to the [documentation](https://calad0i.github.io/HGQ2/) for more details on how to use the library.
