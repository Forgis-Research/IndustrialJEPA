# TimesFM-2.5 Loading Failure (Phase 1 Gate)

**Date**: 2026-04-26
**Time budget used**: ~25 minutes

## Summary

TimesFM-2.5-200M is NOT loadable via the installed `timesfm` package (v1.3.0),
which is the latest available on PyPI. Tried both variants. Skipping 2.5 per
hard rule: "document failure, skip 2.5, use freed time to extend other baselines."

## Root Cause

The `timesfm` 1.3.0 package hardcodes:
```python
checkpoint_path = os.path.join(snapshot_download(repo_id), "torch_model.ckpt")
```

The `google/timesfm-2.5-200m-pytorch` HF repo contains only `model.safetensors`
(925 MB) - no `torch_model.ckpt`. Manual loading was attempted but failed due to
architecture differences between the 1.0 and 2.5 model implementations:

- 1.0 uses `stacked_transformer.layers.N.self_attn.*` key names
- 2.5 uses `stacked_xf.N.attn.*` key names
- 1.0 PatchedTimeSeriesDecoder has 253 keys; 2.5 safetensors has 232 keys
- 1.0 has bias terms; 2.5 does not (different attention implementation)
- 2.5 has `output_projection_point/quantiles`; 1.0 has `horizon_ff_layer`

A complete key remapping would require reimplementing the 2.5 attention block
(`stacked_xf`) which is not in the `timesfm` 1.3.0 package.

## Transformers Variant

`google/timesfm-2.5-200m-transformers` was checked:
- Files: README.md, config.json, model.safetensors
- `AutoConfig.from_pretrained(..., trust_remote_code=True)` fails with
  `'timesfm2_5'` not recognized - the custom model type is not registered
  in the installed `transformers` package.

## Outcome

TimesFM-2.5 skipped. TimesFM-1.0-200M remains the canonical TimesFM baseline
(4 datasets already done). Phase 4 will extend all three existing baselines
(MOMENT-1-large, TimesFM-1.0-200M, Moirai-1.1-R-base) to all 11 datasets.

## Paper Impact

The paper will note that TimesFM-2.5 was evaluated but could not be loaded
due to a checkpoint format incompatibility with the publicly available
`timesfm` Python package (v1.3.0), and that TimesFM-1.0-200M is used instead.
Honest negative: acknowledged in Section 5.1 / appendix.
