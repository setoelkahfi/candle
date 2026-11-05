# onde-candle-metal-kernels

The Metal kernels behind the `metal` feature of `onde-candle-core`, from the Onde Inference
build of candle. Most people use it through that crate rather than directly.

This is the Onde Inference build of [candle](https://github.com/huggingface/candle),
published so that crates which need it (the Onde engine and `onde-mistralrs`) can
depend on a released version instead of a git revision. It tracks Hugging Face
candle and keeps the same API and the same import names.

## What differs from upstream 0.11.0

0.11.1 fixes `get_strided_idx_fallback` in `reduce.metal`. It consumed the dimensions after
the fourth from the wrong end, so a strided reduce over a tensor with more than four
dimensions read the wrong elements. Backward through a grouped-query-attention head expand
gave a gradient off by a relative 1.29 before the fix.

## License

MIT or Apache-2.0, at your option. The code is candle's, copyright its authors.
