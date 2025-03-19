# QKeras

## Compatibility layer

```{warning}
Currently the compatibility layer is in alpha stage and **must** not be for any purpose other than testing or development. Please do not use it in production. In general, we recommend using `HGQ2` directly for all new models, as the compatibility layer only exposes a small subset of the `HGQ2` functionality, and we cannot guarantee that it will be maintained in the future. If you are using `QKeras` for L1 trigger applications, we recommend using `HGQ2` directly, as it is designed for this purpose and will be more efficient than `QKeras`.
```

This framework is designed as a alternative to `QKeras` for ultra-low latency applications, mainly L1 triggers at collider experiments. As `Qkeras` is based on keras v2, HGQ2 may not coexist with `QKeras` in the same environment, thus no interoperability could be expected.

However, as most `QKeras` models targeting `hls4ml` may be represented 1:1 in HGQ2, we implemented a minimal `QKeras` compatibility layer on top of HGQ2. However, there are certain layers in `QKeras` that are not supported by HGQ2, such as the RNN family and hard activation layers. Operations not supported by `hls4ml`, such as `auto_po2` integer bits assignment with float scaling factors, are also not supported.

## Installation

HGQ2 provides a importable `qkeras` package directly, no extra procedure is required.
