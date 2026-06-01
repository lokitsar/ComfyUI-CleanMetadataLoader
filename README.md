# ComfyUI Clean Metadata Loader

Custom nodes for removing ComfyUI's embedded `workflow` metadata while preserving readable generation metadata such as prompt, seed, sampler, scheduler, model, and other text chunks.

## Why

Existing nodes such as ComfyUI's folder dataset loader, KJNodes' folder loader, and WAS Node Suite's batch loader cover directory loading. These nodes add the missing privacy/export step: keep useful generation metadata, but strip the embedded ComfyUI workflow payload before reuse or sharing.

The default behavior removes only the `workflow` metadata chunk. It preserves `prompt`, `parameters`, and other readable metadata so sites such as Civitai can still display generation details.

When `extract_generation_metadata` is enabled, the nodes inspect the embedded workflow before stripping it. They try to follow `KSampler -> positive/negative conditioning -> CLIPTextEncode -> upstream text node` links, then write the recovered prompt into normal readable PNG fields:

- `parameters`
- `positive_prompt`
- `negative_prompt`
- `Civitai resources`

The `parameters` field is written in the common Automatic1111-style format because Civitai and many metadata readers recognize it:

```text
positive prompt
Negative prompt: negative prompt
Steps: 12, Sampler: euler, CFG scale: 1.0, Seed: 123, Size: 768x1344, Model: model.safetensors
Loras: path/to/lora.safetensors:0.75
```

The node also sanitizes the retained ComfyUI `prompt` JSON so `CLIPTextEncode.text`, `KSampler.seed`, and save-node prompt fields become literal readable values instead of links to custom nodes.

## Install

Copy this folder into:

```text
ComfyUI/custom_nodes/ComfyUI-CleanMetadataLoader
```

Then restart ComfyUI.

## Nodes

### `image/metadata -> Strip ComfyUI Workflow From Directory`

Use this to fix a folder of already-generated images before sharing or re-uploading.

Inputs:

- `directory`: folder containing source images.
- `pattern`: semicolon-separated filename globs, for example `*.png;*.jpg;*.webp`.
- `recursive`: include subfolders.
- `output_directory`: where cleaned copies are saved. Empty means save next to the source files.
- `clean_suffix`: suffix added before the file extension. Use the default first so originals remain untouched.
- `remove_prompt_json`: also remove ComfyUI's `prompt` JSON. Leave this `False` for Civitai-style readable generation metadata.
- `extract_generation_metadata`: recover prompts from the workflow/custom text nodes and write them into readable metadata before stripping `workflow`.

Outputs:

- `report_json`: JSON report for every processed image.
- `processed_count`: number of matched images.
- `workflow_removed_count`: number of images where a `workflow` chunk was removed.

### `image/metadata -> Clean Metadata Loader`

Legacy aliases are also registered as `Clean Metadata Directory Loader (Legacy)` and `Load Clean Image(s) From Directory (Legacy)` so older saved workflows still load.

This loads one image at a time from a directory and can optionally save a workflow-stripped copy of that selected image. It intentionally loads a single file per execution to avoid ComfyUI batch errors when directory images have different dimensions.

You can also connect an `IMAGE` input, for example from ComfyUI's standard `Load Image` node. When `image` is connected, it overrides the directory loader. Because ComfyUI image tensors do not carry the source PNG metadata, fill `source_path` with the original image path when you want this node to strip workflow metadata from that file. If `source_path` is blank, the node passes the connected image through and reports that there was no metadata file to clean.

Inputs:

- `directory`: folder containing source images.
- `pattern`: semicolon-separated filename globs, for example `*.png;*.jpg;*.webp`.
- `mode`: `incremental`, `index`, or `random`.
- `index`: starting image index for `incremental`, or exact wrapped index for `index`.
- `batch_size`: kept only for compatibility with older saved workflows; the node still loads one image per execution.
- `seed`: random seed used by `random`.
- `recursive`: include subfolders.
- `save_clean_copy`: writes cleaned image files when enabled.
- `output_directory`: where cleaned copies are saved. Empty means save next to the source files.
- `clean_suffix`: suffix added before the file extension.
- `remove_prompt_json`: also remove ComfyUI's `prompt` JSON. Leave this `False` for Civitai-style readable generation metadata.
- `extract_generation_metadata`: recover prompts from the workflow/custom text nodes and write them into readable metadata before stripping `workflow`.
- `source_path`: optional original image path used when an `IMAGE` input is connected.
- `image`: optional image input. When connected, it overrides `directory`.

Outputs:

- `image`: ComfyUI image tensor.
- `path`: cleaned path if cleaning is enabled, otherwise source path.
- `metadata_json`: JSON report containing source path, cleaned path, selected index, total count, removed keys, and preserved metadata summary.
- `index`: selected file index.
- `total_count`: number of matching images in the directory.

## Metadata Behavior

For PNG files, the nodes preserve text metadata chunks except `workflow` by default, and also carry through ICC, EXIF, and DPI data when Pillow exposes them. Set `remove_prompt_json=True` only if you also want to remove ComfyUI's `prompt` JSON. Set `extract_generation_metadata=True` to rescue prompt text from custom text nodes before `workflow` is removed. For JPEG and WebP files, the nodes preserve EXIF/ICC/DPI where supported by Pillow; those formats do not normally contain ComfyUI workflow chunks in the same PNG text-field style.

The cleaned copy is the shareable file with ComfyUI workflow metadata removed and readable generation metadata retained.
