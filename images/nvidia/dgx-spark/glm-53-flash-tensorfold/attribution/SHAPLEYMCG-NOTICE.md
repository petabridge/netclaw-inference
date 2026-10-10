# ShapleyMcG attribution notice

`incoai`'s drafter and the **default** target checkpoint this image serves are not ShapleyMcG
work: the default target, `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`, is Apache-2.0. This
notice covers the **alternate** checkpoint the vendored recipe can still serve by setting
`MODEL_ID` (`recipe/scripts/config.sh`):

[`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw),
a byte-identical mirror of `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (the recipe's default before
v1.3.3), is licensed under the
[ShapleyMCG License 1.0](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/blob/main/LICENSE),
an attribution-required, source-available license. The checkpoint is not included in this
image. That license requires the following notice in any README or publication that uses the
Work, reproduced verbatim from the upstream recipe's `CREDITS.md` and `NOTICE` at commit
`05397e24ee1fa0a60df11ff5dcb7a68f45cff6cd`:

> This work includes or was produced using ShapleyMcg, created by Brandon M. Music
> (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg
> License v1.0, an attribution-required license that grants no rights to the person known as
> "0xSero." Use of ShapleyMcg without this attribution is unlicensed.

Citation:

```bibtex
@misc{music2026shapleymcg,
  author = {Music, Brandon M.},
  title  = {ShapleyMCG: An Auditable Calibration-to-Encoding Pipeline for
            Low-Bit Mixture-of-Experts Models},
  year   = {2026},
  url    = {https://github.com/brandonmmusic-max/shapleymcg},
  note   = {Licensed under the ShapleyMcg License v1.0}
}
```
