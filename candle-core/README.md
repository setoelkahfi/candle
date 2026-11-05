# onde-candle-core

Tensors, devices and autograd for Rust, from the Onde Inference build of candle.

This is the Onde Inference build of [candle](https://github.com/huggingface/candle),
published so that crates which need it (the Onde engine and `onde-mistralrs`) can
depend on a released version instead of a git revision. It tracks Hugging Face
candle and keeps the same API and the same import names.

## Use it

```toml
[dependencies]
candle = { package = "onde-candle-core", version = "0.11" }
# features: metal, accelerate, mkl, cuda, cudnn
```

The library is named `candle_core`. With the rename above you write `use candle::Tensor;`,
the same as with upstream candle.

## What differs from upstream 0.11.0

- Runtime-dispatched CPU kernels, including a 256-bit AVX-VNNI/AVX2 tier and an
  expert-bucketed indexed matmul for MoE prefill.
- Metal strided reduces over more than four dimensions are correct. The bug was in
  `candle-metal-kernels` 0.11.0 (the Metal feature gives a wrong gradient for expand followed
  by reshape, as in grouped-query attention). 0.11.2 requires `candle-metal-kernels` 0.11.1,
  the upstream release with the fix.

## License

MIT or Apache-2.0, at your option. The code is candle's, copyright its authors.
