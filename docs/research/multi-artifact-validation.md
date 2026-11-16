# Multi-artifact Ollama model validation

**Date:** 2026-08-31  
**Status:** Phase 0 complete; proceed with a constrained model + projector implementation

## Decision

StudioLink can safely support an Ollama manifest containing exactly:

- one `application/vnd.ollama.image.model` GGUF; and
- zero or one `application/vnd.ollama.image.projector` GGUF.

Both files must be imported into the same LM Studio `user/repo` directory. The projector alias must start with `mmproj-` and end in `.gguf`. A digest-keyed suffix is accepted by LM Studio, so aliases do not need to retain the original upstream filename.

The first implementation must reject, rather than partially sync:

- manifests with more than one model layer;
- manifests with more than one projector layer;
- manifests containing an Ollama adapter layer;
- manifests containing an Ollama draft layer; and
- manifests whose only model representation is an Ollama tensor layer.

An Ollama draft layer is not part of the importable model bundle. LM Studio treats speculative draft models as separately selected model resources, not as files automatically associated by `lms import`. Phase 1 should report a draft layer as an unsupported semantic extension and block sync instead of claiming exact Ollama-to-LM-Studio parity.

## Primary-source findings

### Ollama manifest semantics

Ollama assigns distinct media types to model, projector, adapter, draft, and tensor data. Its current loader maps model layers to `ModelPath`, projectors to a list of `ProjectorPaths`, adapters to `AdapterPaths`, and a draft layer to `DraftPath`.[^ollama-loader] The current manifest package separately defines tensor and draft media types.[^ollama-layer-types]

These are semantically active layers, not incidental metadata. Ollama passes adapter and projector path lists into its runner, and represents the draft as a separate model path.[^ollama-runner] Ollama's model-list implementation also adds vision capability when at least one projector layer is present.[^ollama-vision]

The public `llava:latest` registry manifest retrieved on 2026-08-31 contains one model layer followed by one projector layer, then license, template, and parameter metadata.[^llava-manifest] An exact copy is retained at [`tests/fixtures/ollama/llava-latest-manifest.json`](../../tests/fixtures/ollama/llava-latest-manifest.json). `bakllava:latest` had the same model-plus-projector shape. By contrast, `qwen3-vl:latest` used one unified model layer and no projector, confirming that projector support must be conditional rather than assumed for every vision model.

### LM Studio import semantics

The current `lms import` interface accepts one file path per invocation.[^lms-import-source] It constructs the destination as:

```text
<models folder>/<user>/<repo>/<source basename>
```

and preserves that basename for copy, hard-link, and symbolic-link imports.[^lms-target-source] The official CLI documentation likewise describes importing one local model file and provides no batch or artifact-role option.[^lms-import-docs]

LM Studio's documented speculative-decoding interface selects a separate draft model resource. It does not document automatic association of a bundled draft file.[^lmstudio-draft]

No current LM Studio documentation or `lms load --help` option exposes an Ollama-style LoRA/adapter association. `lms import` can place an arbitrary file, but placement alone is not evidence that LM Studio can apply an Ollama adapter to its base model. Adapter support is therefore outside the safe initial scope.

## End-to-end validation

Validation used:

- LM Studio `0.4.23+1`;
- `lms` commit `07b7252d6de26a3a58c1bb80ed7e75a2b17f5eb6`;
- the installed Ministral 3 3B GGUF and its projector; and
- temporary hard links, so no model bytes were copied and no model was loaded.

Temporary aliases were given the planned StudioLink naming shape:

```text
ministral-3-3b-validation-deadbeef.gguf
mmproj-ministral-3-3b-validation-cafebabe.gguf
```

Each alias was imported separately with:

```text
lms import <alias> --yes --user-repo studiolink-validation/mmproj-pair --hard-link
```

After both imports, `lms ls --json` returned exactly one inventory entry:

```json
{
  "type": "llm",
  "modelKey": "mmproj-pair",
  "publisher": "studiolink-validation",
  "path": "studiolink-validation/mmproj-pair/ministral-3-3b-validation-deadbeef.gguf",
  "sizeBytes": 2986795328,
  "architecture": "mistral3",
  "vision": true
}
```

The reported size was exactly the base file plus the projector file. The installed LM Studio index identified the `mmproj-...gguf` file as the model's `visionAdapter`. This confirms all of the assumptions needed for one-projector support:

1. separate `lms import` calls can build one logical model bundle;
2. same-directory placement is sufficient;
3. an `mmproj-` prefix is recognized;
4. digest-keyed renamed files remain associated; and
5. `lms ls --json` is a viable public reconciliation seam for the resulting model.

### Multiple-projector negative test

A second reversible experiment imported one model and two `mmproj-*.gguf` files into the same repository. LM Studio returned one vision model but indexed only one projector:

```json
{
  "entryPoint": "ministral-multi-validation.gguf",
  "visionAdapter": "mmproj-ministral-multi-two.gguf",
  "allFiles": [
    "ministral-multi-validation.gguf",
    "mmproj-ministral-multi-two.gguf"
  ]
}
```

The other projector was ignored. The public inventory did not explain or expose the ambiguity. StudioLink must therefore reject multiple projectors until LM Studio provides a documented selection mechanism.

All temporary aliases, imported targets, and resulting inventory entries were removed after validation.

## Supported-layer matrix

| Ollama layer | Initial handling | Reason |
|---|---|---|
| `image.model` | Import; exactly one required | Confirmed GGUF entry point |
| `image.projector` | Import when exactly one; `mmproj-` alias | Confirmed as LM Studio `visionAdapter` |
| Second projector | Reject model | LM Studio silently selected only one |
| `image.adapter` | Reject model | Semantically required by Ollama; no documented LM Studio association seam |
| `image.draft` | Reject model | LM Studio selects draft models separately; `lms import` cannot preserve the bundle relationship |
| `image.tensor` | Reject model | Not a standalone GGUF import supported by `lms import` |
| template/system/params/license/config | Do not import as artifacts | Manifest metadata, not LM Studio model files |

## Consequences for Phase 1

1. Add artifacts without changing the user-facing model selection seam.
2. Treat model and projector as required artifacts for readiness and sync.
3. Preserve the existing primary model alias naming; prefix projector aliases with `mmproj-`.
4. Import both into the same `user_repo` using separate CLI calls.
5. Do not allow `--direct` for projector bundles because the opaque blob basename loses the required `mmproj-` role marker.
6. Do not claim support for adapters, drafts, tensor-only manifests, or multiple projectors.
7. Add `lms ls --json` inventory parsing before relying on inferred destination paths for multi-artifact reconciliation.

## Remaining uncertainty

The validation covers current GGUF model-plus-projector bundles on LM Studio 0.4.23. LM Studio's projector pairing rule is observable but not documented as a stable interface. StudioLink should retain integration tests around inventory behavior and fail clearly if a future LM Studio version stops reporting the imported bundle as vision-capable.

[^ollama-loader]: Ollama source, [`server/images.go` lines 717–740](https://github.com/ollama/ollama/blob/f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a/server/images.go#L717-L740).
[^ollama-layer-types]: Ollama source, [`manifest/layer.go` lines 20–24](https://github.com/ollama/ollama/blob/f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a/manifest/layer.go#L20-L24).
[^ollama-runner]: Ollama source, [`server/sched.go` runner construction](https://github.com/ollama/ollama/blob/f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a/server/sched.go#L577) and [`server/routes.go` draft model path](https://github.com/ollama/ollama/blob/f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a/server/routes.go#L2368).
[^ollama-vision]: Ollama source, [`server/model_list_cache.go` lines 341–391](https://github.com/ollama/ollama/blob/f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a/server/model_list_cache.go#L341-L391).
[^llava-manifest]: Ollama registry, [`library/llava:latest` manifest](https://registry.ollama.ai/v2/library/llava/manifests/latest), retrieved 2026-08-31.
[^lms-import-source]: LM Studio CLI source, [single `<file-path>` argument](https://github.com/lmstudio-ai/lms/blob/07b7252d6de26a3a58c1bb80ed7e75a2b17f5eb6/src/subcommands/importCmd.ts#L71-L74).
[^lms-target-source]: LM Studio CLI source, [destination path construction](https://github.com/lmstudio-ai/lms/blob/07b7252d6de26a3a58c1bb80ed7e75a2b17f5eb6/src/subcommands/importCmd.ts#L176) and [hard-link implementation](https://github.com/lmstudio-ai/lms/blob/07b7252d6de26a3a58c1bb80ed7e75a2b17f5eb6/src/subcommands/importCmd.ts#L247-L252).
[^lms-import-docs]: LM Studio documentation, [`lms import`](https://lmstudio.ai/docs/cli/local-models/import).
[^lmstudio-draft]: LM Studio documentation, [Speculative Decoding](https://lmstudio.ai/docs/app/advanced/speculative-decoding).
