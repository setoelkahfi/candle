# onde-candle-nn

Neural network layers, optimizers and activations on top of `onde-candle-core`, from the
Onde Inference build of candle.

This is the Onde Inference build of [candle](https://github.com/huggingface/candle),
published so that crates which need it (the Onde engine and `onde-mistralrs`) can
depend on a released version instead of a git revision. It tracks Hugging Face
candle and keeps the same API and the same import names.

## Use it

```toml
[dependencies]
candle = { package = "onde-candle-core", version = "0.11" }
candle-nn = { package = "onde-candle-nn", version = "0.11" }
```

The library is named `candle_nn`, so `use candle_nn::Linear;` works as with upstream.
Features `metal`, `accelerate` and `mkl` follow the ones on `onde-candle-core`.

## License

MIT or Apache-2.0, at your option. The code is candle's, copyright its authors.
