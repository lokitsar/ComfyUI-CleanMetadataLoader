import fnmatch
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, PngImagePlugin

try:
    import folder_paths
except ImportError:
    folder_paths = None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
WORKFLOW_METADATA_KEYS = {"workflow"}
PROMPT_METADATA_KEYS = {"prompt"}
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
TEXT_INPUT_NAMES = {
    "text",
    "prompt",
    "positive",
    "positive_prompt",
    "string",
    "value",
    "manual_text",
    "text_0",
}
POSITIVE_INPUT_NAMES = {"positive"}
NEGATIVE_INPUT_NAMES = {"negative"}


def _bool(value):
    return bool(value)


def _list_images(directory, pattern, recursive):
    root = Path(directory).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    files = root.rglob("*") if recursive else root.glob("*")
    matches = []
    patterns = [part.strip() for part in pattern.split(";") if part.strip()] or ["*"]

    for path in files:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if any(fnmatch.fnmatch(path.name, pat) for pat in patterns):
            matches.append(path)

    return sorted(matches, key=lambda item: str(item).lower())


def _comfy_input_images(allow_blank=False):
    if folder_paths is None:
        return [""] if allow_blank else []
    input_dir = folder_paths.get_input_directory()
    images = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    images = folder_paths.filter_files_content_types(images, ["image"])
    images = sorted(images)
    return ([""] + images) if allow_blank else images


def _resolve_input_image(image_name):
    if not image_name:
        return None
    if folder_paths is not None:
        return Path(folder_paths.get_annotated_filepath(image_name))
    return Path(image_name).expanduser()


def _counter_key(directory, pattern, recursive):
    return f"{Path(directory).expanduser()}|{pattern}|{recursive}"


def _select_file(files, mode, index, seed, counter_key, counters):
    if not files:
        raise FileNotFoundError("No image files matched the directory and pattern.")

    if mode == "incremental":
        current_index = counters.get(counter_key, index) % len(files)
        counters[counter_key] = current_index + 1
        return files[current_index], current_index

    if mode == "random":
        rng = random.Random(seed)
        current_index = rng.randrange(len(files))
        return files[current_index], current_index

    current_index = index % len(files)
    return files[current_index], current_index


def _metadata_dict(image):
    metadata = {}

    for key, value in image.info.items():
        if key in {"exif", "icc_profile"}:
            metadata[key] = f"<{len(value)} bytes>"
        elif isinstance(value, bytes):
            metadata[key] = value.decode("utf-8", errors="replace")
        else:
            metadata[key] = str(value)

    return metadata


def _loads_json(value):
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _text_value(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return ""


def _node_title(node):
    title = node.get("title") or node.get("type") or node.get("class_type") or "node"
    return str(title)


def _workflow_graph(workflow):
    nodes = {str(node.get("id")): node for node in workflow.get("nodes", []) if isinstance(node, dict)}
    links = {}

    for link in workflow.get("links", []):
        if isinstance(link, list) and len(link) >= 5:
            links[str(link[0])] = {
                "source_node": str(link[1]),
                "source_slot": link[2],
                "target_node": str(link[3]),
                "target_slot": link[4],
            }
        elif isinstance(link, dict):
            link_id = link.get("id")
            if link_id is not None:
                links[str(link_id)] = {
                    "source_node": str(link.get("origin_id") or link.get("source_node") or link.get("from_node")),
                    "source_slot": link.get("origin_slot") or link.get("source_slot") or link.get("from_slot"),
                    "target_node": str(link.get("target_id") or link.get("target_node") or link.get("to_node")),
                    "target_slot": link.get("target_slot") or link.get("to_slot"),
                }

    return nodes, links


def _input_link_by_name(node, input_names):
    for index, item in enumerate(node.get("inputs", []) or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).lower()
        if name in input_names and item.get("link") is not None:
            return str(item.get("link"))

    return None


def _linked_source_node_id(links, link_id):
    link = links.get(str(link_id))
    if not link:
        return None
    return link.get("source_node")


def _node_text_candidates(node):
    candidates = []

    widgets = node.get("widgets_values", [])
    if isinstance(widgets, list):
        for value in widgets:
            text = _text_value(value)
            if text:
                candidates.append(text)
    elif isinstance(widgets, dict):
        for key, value in widgets.items():
            if str(key).lower() in TEXT_INPUT_NAMES:
                text = _text_value(value)
                if text:
                    candidates.append(text)

    for item in node.get("inputs", []) or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("name", "")).lower() in TEXT_INPUT_NAMES:
            text = _text_value(item.get("value"))
            if text:
                candidates.append(text)

    return candidates


def _best_text_candidate(candidates):
    useful = [item for item in candidates if item and not item.startswith("[") and not item.startswith("{")]
    return max(useful or candidates or [""], key=len).strip()


def _trace_workflow_text_from_node(node_id, nodes, links, visited=None):
    visited = visited or set()
    if node_id is None or node_id in visited:
        return ""
    visited.add(node_id)

    node = nodes.get(str(node_id))
    if not node:
        return ""

    text_link = _input_link_by_name(node, TEXT_INPUT_NAMES)
    if text_link is not None:
        linked_text = _trace_workflow_text_from_node(_linked_source_node_id(links, text_link), nodes, links, visited)
        if linked_text:
            return linked_text

    candidates = _node_text_candidates(node)
    if candidates:
        return _best_text_candidate(candidates)

    upstream_text = []
    for item in node.get("inputs", []) or []:
        if isinstance(item, dict) and item.get("link") is not None:
            linked_text = _trace_workflow_text_from_node(_linked_source_node_id(links, item.get("link")), nodes, links, visited)
            if linked_text:
                upstream_text.append(linked_text)

    return _best_text_candidate(upstream_text)


def _extract_workflow_prompts(workflow):
    if not isinstance(workflow, dict):
        return {}

    nodes, links = _workflow_graph(workflow)
    result = {"positive": "", "negative": "", "sources": {}}

    for node_id, node in nodes.items():
        node_type = str(node.get("type", ""))
        if "ksampler" not in node_type.lower() and node_type.lower() not in {"samplercustomadvanced"}:
            continue

        for label, input_names in (("positive", POSITIVE_INPUT_NAMES), ("negative", NEGATIVE_INPUT_NAMES)):
            link_id = _input_link_by_name(node, input_names)
            source_id = _linked_source_node_id(links, link_id) if link_id else None
            text = _trace_workflow_text_from_node(source_id, nodes, links)
            if text and not result[label]:
                result[label] = text
                result["sources"][label] = f"{_node_title(nodes.get(str(source_id), {}))} (ID {source_id})"

    if result["positive"] or result["negative"]:
        return result

    clip_texts = []
    for node_id, node in nodes.items():
        if "cliptextencode" in str(node.get("type", "")).lower():
            text = _trace_workflow_text_from_node(node_id, nodes, links)
            if text:
                clip_texts.append((node_id, text))

    if clip_texts:
        result["positive"] = clip_texts[0][1]
        result["sources"]["positive"] = f"CLIPTextEncode (ID {clip_texts[0][0]})"
    if len(clip_texts) > 1:
        result["negative"] = clip_texts[1][1]
        result["sources"]["negative"] = f"CLIPTextEncode (ID {clip_texts[1][0]})"

    return result


def _api_prompt_graph(prompt):
    if not isinstance(prompt, dict):
        return {}
    return {str(node_id): node for node_id, node in prompt.items() if isinstance(node, dict)}


def _resolve_api_value(value, graph, input_name=None, visited=None):
    if not isinstance(value, list) or not value:
        return value

    visited = visited or set()
    node_id = str(value[0])
    if node_id in visited:
        return value
    visited.add(node_id)

    node = graph.get(node_id)
    if not node:
        return value

    inputs = node.get("inputs", {})
    if not isinstance(inputs, dict):
        return value

    if input_name and input_name in inputs:
        return _resolve_api_value(inputs[input_name], graph, input_name, visited)

    preferred_inputs = ("seed", "value", "manual_text", "text", "text_0", "model_name", "unet_name", "width", "height")
    for name in preferred_inputs:
        if name in inputs:
            resolved = _resolve_api_value(inputs[name], graph, name, visited)
            if not isinstance(resolved, list):
                return resolved

    for resolved in (_resolve_api_value(item, graph, None, visited) for item in inputs.values()):
        if not isinstance(resolved, list):
            return resolved

    return value


def _trace_api_prompt_text(node_id, graph, visited=None):
    visited = visited or set()
    if node_id is None or str(node_id) in visited:
        return ""
    node_id = str(node_id)
    visited.add(node_id)

    node = graph.get(node_id)
    if not node:
        return ""

    inputs = node.get("inputs", {})
    if isinstance(inputs, dict):
        for name, value in inputs.items():
            if str(name).lower() in TEXT_INPUT_NAMES and isinstance(value, list) and value:
                linked = _trace_api_prompt_text(value[0], graph, visited)
                if linked:
                    return linked

        for name, value in inputs.items():
            if str(name).lower() in TEXT_INPUT_NAMES:
                text = _text_value(value)
                if text:
                    return text

        candidates = []
        for value in inputs.values():
            text = _text_value(value)
            if text:
                candidates.append(text)
        if candidates:
            return _best_text_candidate(candidates)

    return ""


def _extract_api_prompt_prompts(prompt):
    graph = _api_prompt_graph(prompt)
    result = {"positive": "", "negative": "", "sources": {}}

    for node_id, node in graph.items():
        class_type = str(node.get("class_type", ""))
        if "ksampler" not in class_type.lower() and class_type.lower() not in {"samplercustomadvanced"}:
            continue

        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue

        for label in ("positive", "negative"):
            value = inputs.get(label)
            if isinstance(value, list) and value:
                text = _trace_api_prompt_text(value[0], graph)
                if text and not result[label]:
                    result[label] = text
                    result["sources"][label] = f"{graph.get(str(value[0]), {}).get('class_type', 'node')} (ID {value[0]})"

    return result


def _first_ksampler_node(graph):
    for node_id, node in graph.items():
        class_type = str(node.get("class_type", "")).lower()
        if "ksampler" in class_type:
            return node_id, node
    return None, None


def _basename_without_extension(value):
    if not value or not isinstance(value, str):
        return ""
    return Path(value.replace("\\", "/")).stem


def _extract_loras(graph):
    loras = []
    for node in graph.values():
        inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
        if not isinstance(inputs, dict):
            continue

        for key, value in inputs.items():
            if not key.lower().startswith("lora_") or not isinstance(value, str) or value.lower() == "none":
                continue

            suffix = key.split("_", 1)[1]
            strength = inputs.get(f"strength_{suffix}", 1.0)
            loras.append(
                {
                    "name": value,
                    "trigger": _basename_without_extension(value),
                    "strength": strength,
                }
            )

    return loras


def _extract_model_name(graph, sampler_node):
    def find_model_name(node_id, visited=None):
        visited = visited or set()
        node_id = str(node_id)
        if node_id in visited:
            return ""
        visited.add(node_id)

        node = graph.get(node_id)
        inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
        if not isinstance(inputs, dict):
            return ""

        for name in ("ckpt_name", "model_name", "unet_name"):
            value = inputs.get(name)
            if isinstance(value, str) and value.lower() != "none":
                return value
            if isinstance(value, list) and value:
                found = find_model_name(value[0], visited)
                if found:
                    return found

        model_link = inputs.get("model")
        if isinstance(model_link, list) and model_link:
            found = find_model_name(model_link[0], visited)
            if found:
                return found

        return ""

    if sampler_node:
        model_link = sampler_node.get("inputs", {}).get("model")
        if isinstance(model_link, list) and model_link:
            found = find_model_name(model_link[0])
            if found:
                return found

    for node in graph.values():
        class_type = str(node.get("class_type", "")).lower() if isinstance(node, dict) else ""
        if not any(token in class_type for token in ("checkpoint", "unet", "diffusionmodel", "modelselector")):
            continue
        inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
        if not isinstance(inputs, dict):
            continue
        for name in ("ckpt_name", "model_name", "unet_name"):
            value = inputs.get(name)
            if isinstance(value, str) and value.lower() != "none":
                return value

    return ""


def _extract_size(graph, sampler_node):
    latent = sampler_node.get("inputs", {}).get("latent_image") if sampler_node else None
    if isinstance(latent, list) and latent:
        latent_node = graph.get(str(latent[0]))
        inputs = latent_node.get("inputs", {}) if isinstance(latent_node, dict) else {}
        if isinstance(inputs, dict):
            width = _resolve_api_value(inputs.get("width"), graph, "width")
            height = _resolve_api_value(inputs.get("height"), graph, "height")
            if width and height:
                return f"{width}x{height}"

    for node in graph.values():
        inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
        if isinstance(inputs, dict) and "width" in inputs and "height" in inputs:
            width = _resolve_api_value(inputs.get("width"), graph, "width")
            height = _resolve_api_value(inputs.get("height"), graph, "height")
            if width and height:
                return f"{width}x{height}"

    return ""


def _extract_api_settings(prompt):
    graph = _api_prompt_graph(prompt)
    sampler_id, sampler_node = _first_ksampler_node(graph)
    settings = {"loras": _extract_loras(graph), "sampler_node": sampler_id or ""}

    if sampler_node:
        inputs = sampler_node.get("inputs", {})
        for source, target in (
            ("seed", "seed"),
            ("steps", "steps"),
            ("cfg", "cfg_scale"),
            ("sampler_name", "sampler"),
            ("scheduler", "scheduler"),
            ("denoise", "denoise"),
        ):
            if source in inputs:
                settings[target] = _resolve_api_value(inputs[source], graph, source)

    model_name = _extract_model_name(graph, sampler_node)
    if model_name:
        settings["model"] = model_name

    size = _extract_size(graph, sampler_node)
    if size:
        settings["size"] = size

    return settings


def _literalize_api_prompt(prompt, generation_data):
    if not isinstance(prompt, dict):
        return None

    sanitized = copy.deepcopy(prompt)
    graph = _api_prompt_graph(sanitized)

    for node_id, node in graph.items():
        class_type = str(node.get("class_type", ""))
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue

        if "cliptextencode" in class_type.lower():
            for sampler in graph.values():
                sampler_inputs = sampler.get("inputs", {}) if isinstance(sampler, dict) else {}
                if not isinstance(sampler_inputs, dict):
                    continue
                if sampler_inputs.get("positive", [None])[0] == node_id and generation_data.get("positive"):
                    inputs["text"] = generation_data["positive"]
                if sampler_inputs.get("negative", [None])[0] == node_id and generation_data.get("negative"):
                    inputs["text"] = generation_data["negative"]

        if "ksampler" in class_type.lower():
            for scalar_name in ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"):
                if scalar_name in inputs:
                    inputs[scalar_name] = _resolve_api_value(inputs[scalar_name], graph, scalar_name)

        if class_type in {"Save Image w/Metadata", "SaveImageWithMetadata"} or "save image" in class_type.lower():
            if generation_data.get("positive") and "positive" in inputs:
                inputs["positive"] = generation_data["positive"]
            if generation_data.get("negative") and "negative" in inputs:
                inputs["negative"] = generation_data["negative"]
            if "seed_value" in inputs:
                inputs["seed_value"] = _resolve_api_value(inputs["seed_value"], graph, "seed")

    return sanitized


def _merge_prompt_data(workflow_data, api_prompt_data, api_settings=None):
    merged = {"positive": "", "negative": "", "sources": {}, "settings": api_settings or {}}
    for label in ("positive", "negative"):
        merged[label] = workflow_data.get(label) or api_prompt_data.get(label) or ""
        merged["sources"][label] = workflow_data.get("sources", {}).get(label) or api_prompt_data.get("sources", {}).get(label) or ""
    return merged


def _extract_generation_data(metadata_image):
    workflow = _loads_json(metadata_image.info.get("workflow"))
    prompt = _loads_json(metadata_image.info.get("prompt"))
    workflow_data = _extract_workflow_prompts(workflow)
    api_prompt_data = _extract_api_prompt_prompts(prompt)
    api_settings = _extract_api_settings(prompt)
    return _merge_prompt_data(workflow_data, api_prompt_data, api_settings)


def _format_setting(value):
    if value is None or value == "":
        return ""
    return str(value)


def _a1111_sampler(settings):
    sampler = _format_setting(settings.get("sampler"))
    scheduler = _format_setting(settings.get("scheduler"))
    if scheduler and scheduler.lower() not in {"normal", "simple"}:
        return f"{sampler} {scheduler}".strip()
    return sampler


def _civitai_resources(settings):
    resources = []
    model = settings.get("model")
    if model:
        resources.append(
            {
                "type": "checkpoint",
                "modelName": _basename_without_extension(model),
                "modelVersionName": Path(str(model).replace("\\", "/")).name,
            }
        )

    for lora in settings.get("loras") or []:
        resources.append(
            {
                "type": "lora",
                "modelName": _basename_without_extension(lora.get("name", "")),
                "modelVersionName": Path(str(lora.get("name", "")).replace("\\", "/")).name,
                "weight": lora.get("strength", 1.0),
            }
        )

    return resources


def _build_parameters(existing, generation_data):
    positive = generation_data.get("positive", "").strip()
    negative = generation_data.get("negative", "").strip()
    settings = generation_data.get("settings", {})

    if not positive and not negative:
        return existing

    if existing and positive and existing.lstrip().startswith(positive):
        return existing

    lines = []
    if positive:
        lines.append(positive)
    if negative:
        lines.append(f"Negative prompt: {negative}")

    setting_pairs = [
        ("Steps", _format_setting(settings.get("steps"))),
        ("Sampler", _a1111_sampler(settings)),
        ("CFG scale", _format_setting(settings.get("cfg_scale"))),
        ("Seed", _format_setting(settings.get("seed"))),
        ("Size", _format_setting(settings.get("size"))),
        ("Model", _format_setting(settings.get("model"))),
        ("Denoising strength", _format_setting(settings.get("denoise"))),
    ]
    settings_line = ", ".join(f"{name}: {value}" for name, value in setting_pairs if value)
    if settings_line:
        lines.append(settings_line)

    loras = settings.get("loras") or []
    if loras:
        lines.append("Loras: " + ", ".join(f"{item['name']}:{item['strength']}" for item in loras))

    if existing:
        remaining = existing.strip()
        if remaining and "Steps:" not in settings_line and remaining not in lines:
            lines.append(remaining)
    return "\n".join(lines).strip()


def _remove_keys(remove_prompt_json):
    keys = set(WORKFLOW_METADATA_KEYS)
    if remove_prompt_json:
        keys.update(PROMPT_METADATA_KEYS)
    return keys


def _clean_png_info(image, remove_prompt_json=False, extracted_metadata=None):
    pnginfo = PngImagePlugin.PngInfo()
    removed = []
    keys_to_remove = _remove_keys(remove_prompt_json)
    extracted_metadata = extracted_metadata or {}

    for key, value in image.info.items():
        if key in keys_to_remove:
            removed.append(key)
            continue

        if key in extracted_metadata:
            continue

        if key in {"exif", "icc_profile", "transparency", "dpi"}:
            continue

        if isinstance(value, bytes):
            pnginfo.add_text(key, value.decode("utf-8", errors="replace"))
        else:
            pnginfo.add_text(key, str(value))

    for key, value in extracted_metadata.items():
        if value:
            pnginfo.add_text(key, str(value))

    return pnginfo, removed


def _validate_suffix(suffix):
    if any(part in suffix for part in ("/", "\\")) or Path(suffix).drive:
        raise ValueError(f"clean_suffix must be only a filename suffix, not a path: {suffix}")


def _save_clean_copy(
    image,
    source_path,
    output_directory,
    suffix,
    metadata_source=None,
    remove_prompt_json=False,
    extract_generation_metadata=True,
):
    output_root = Path(output_directory).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    source = Path(source_path)
    suffix = suffix or "_clean"
    _validate_suffix(suffix)
    target = output_root / f"{source.stem}{suffix}{source.suffix}"
    save_image = image.copy()
    metadata_image = metadata_source or image

    kwargs = {}
    if "icc_profile" in metadata_image.info:
        kwargs["icc_profile"] = metadata_image.info["icc_profile"]
    if "exif" in metadata_image.info:
        kwargs["exif"] = metadata_image.info["exif"]
    if "dpi" in metadata_image.info:
        kwargs["dpi"] = metadata_image.info["dpi"]

    removed = []
    extracted_metadata = {}
    generation_data = {}
    if extract_generation_metadata:
        generation_data = _extract_generation_data(metadata_image)
        existing_parameters = metadata_image.info.get("parameters", "")
        parameters = _build_parameters(existing_parameters, generation_data)
        if parameters:
            extracted_metadata["parameters"] = parameters
        if generation_data.get("positive"):
            extracted_metadata["positive_prompt"] = generation_data["positive"]
        if generation_data.get("negative"):
            extracted_metadata["negative_prompt"] = generation_data["negative"]
        resources = _civitai_resources(generation_data.get("settings", {}))
        if resources:
            extracted_metadata["Civitai resources"] = json.dumps(resources, ensure_ascii=False)
        sanitized_prompt = _literalize_api_prompt(_loads_json(metadata_image.info.get("prompt")), generation_data)
        if sanitized_prompt is not None and not remove_prompt_json:
            extracted_metadata["prompt"] = json.dumps(sanitized_prompt, ensure_ascii=False)

    if source.suffix.lower() == ".png":
        pnginfo, removed = _clean_png_info(metadata_image, remove_prompt_json, extracted_metadata)
        kwargs["pnginfo"] = pnginfo
    else:
        removed = [key for key in _remove_keys(remove_prompt_json) if key in metadata_image.info]

    if save_image.mode == "RGBA" and source.suffix.lower() in {".jpg", ".jpeg"}:
        save_image = save_image.convert("RGB")

    save_image.save(target, **kwargs)
    return str(target), removed, generation_data


def _image_to_tensor(image):
    image = ImageOps.exif_transpose(image)
    rgba = image.convert("RGBA")

    image_array = np.asarray(rgba.convert("RGB")).astype(np.float32) / 255.0

    image_tensor = torch.from_numpy(image_array)[None,]
    return image_tensor


class LoadCleanImageFromDirectory:
    _incremental_indexes = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "multiline": False}),
                "pattern": ("STRING", {"default": "*.png;*.jpg;*.jpeg;*.webp", "multiline": False}),
                "mode": (["incremental", "index", "random"], {"default": "incremental"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "recursive": ("BOOLEAN", {"default": False}),
                "save_clean_copy": ("BOOLEAN", {"default": True}),
                "output_directory": ("STRING", {"default": "", "multiline": False}),
                "clean_suffix": ("STRING", {"default": "_clean", "multiline": False}),
                "remove_prompt_json": ("BOOLEAN", {"default": False}),
                "extract_generation_metadata": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("image", "path", "metadata_json", "index", "total_count")
    FUNCTION = "load"
    CATEGORY = "image/metadata"

    @classmethod
    def IS_CHANGED(
        cls,
        directory,
        pattern,
        mode,
        index,
        batch_size,
        seed,
        recursive,
        save_clean_copy,
        output_directory,
        clean_suffix,
        remove_prompt_json=False,
        extract_generation_metadata=True,
    ):
        if mode == "incremental":
            return random.random()
        return (
            directory,
            pattern,
            mode,
            index,
            batch_size,
            seed,
            recursive,
            save_clean_copy,
            output_directory,
            clean_suffix,
            remove_prompt_json,
            extract_generation_metadata,
        )

    def load(
        self,
        directory,
        pattern,
        mode,
        index,
        batch_size,
        seed,
        recursive,
        save_clean_copy,
        output_directory,
        clean_suffix,
        remove_prompt_json=False,
        extract_generation_metadata=True,
    ):
        files = _list_images(directory, pattern, _bool(recursive))
        key = _counter_key(directory, pattern, recursive)
        source_path, current_index = _select_file(files, mode, index, seed, key, self._incremental_indexes)

        clean_root = output_directory.strip() or directory
        return self._load_path(
            source_path,
            current_index,
            len(files),
            clean_root,
            save_clean_copy,
            clean_suffix,
            remove_prompt_json,
            extract_generation_metadata,
        )

    def _load_path(
        self,
        source_path,
        current_index,
        total_count,
        clean_root,
        save_clean_copy,
        clean_suffix,
        remove_prompt_json,
        extract_generation_metadata,
    ):
        with Image.open(source_path) as opened:
            source_image = ImageOps.exif_transpose(opened)
            original_metadata = _metadata_dict(opened)
            cleaned_path = ""
            removed = []

            if _bool(save_clean_copy):
                cleaned_path, removed, generation_data = _save_clean_copy(
                    source_image,
                    source_path,
                    clean_root,
                    clean_suffix,
                    metadata_source=opened,
                    remove_prompt_json=_bool(remove_prompt_json),
                    extract_generation_metadata=_bool(extract_generation_metadata),
                )
            else:
                generation_data = _extract_generation_data(opened) if _bool(extract_generation_metadata) else {}

            image_tensor = _image_to_tensor(source_image)

        metadata_record = {
            "source_path": str(source_path),
            "cleaned_path": cleaned_path,
            "selected_index": current_index,
            "total_count": total_count,
            "removed_comfy_keys": removed,
            "extracted_generation_metadata": generation_data,
            "metadata": original_metadata,
        }

        return (
            image_tensor,
            cleaned_path or str(source_path),
            json.dumps(metadata_record, indent=2, ensure_ascii=False),
            current_index,
            total_count,
        )


class LegacyLoadCleanImageFromDirectory(LoadCleanImageFromDirectory):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "multiline": False}),
                "pattern": ("STRING", {"default": "*.png;*.jpg;*.jpeg;*.webp", "multiline": False}),
                "mode": (["incremental", "index", "random"], {"default": "incremental"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "recursive": ("BOOLEAN", {"default": False}),
                "fit": (["original", "pad", "crop", "stretch"], {"default": "original"}),
                "width": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "save_clean_copy": ("BOOLEAN", {"default": True}),
                "output_directory": ("STRING", {"default": "", "multiline": False}),
                "clean_suffix": ("STRING", {"default": "_clean", "multiline": False}),
                "remove_prompt_json": ("BOOLEAN", {"default": False}),
                "extract_generation_metadata": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("image", "path", "metadata_json", "index", "total_count")

    @classmethod
    def IS_CHANGED(
        cls,
        directory,
        pattern,
        mode,
        index,
        batch_size,
        seed,
        recursive,
        fit,
        width,
        height,
        save_clean_copy,
        output_directory,
        clean_suffix,
        remove_prompt_json=False,
        extract_generation_metadata=True,
    ):
        return super().IS_CHANGED(
            directory,
            pattern,
            mode,
            index,
            batch_size,
            seed,
            recursive,
            save_clean_copy,
            output_directory,
            clean_suffix,
            remove_prompt_json,
            extract_generation_metadata,
        )

    def load(
        self,
        directory,
        pattern,
        mode,
        index,
        batch_size,
        seed,
        recursive,
        fit,
        width,
        height,
        save_clean_copy,
        output_directory,
        clean_suffix,
        remove_prompt_json=False,
        extract_generation_metadata=True,
    ):
        return super().load(
            directory,
            pattern,
            mode,
            index,
            batch_size,
            seed,
            recursive,
            save_clean_copy,
            output_directory,
            clean_suffix,
            remove_prompt_json,
            extract_generation_metadata,
        )


class LoadCleanSingleImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (_comfy_input_images(), {"image_upload": True}),
                "save_clean_copy": ("BOOLEAN", {"default": True}),
                "output_directory": ("STRING", {"default": "", "multiline": False}),
                "clean_suffix": ("STRING", {"default": "_clean", "multiline": False}),
                "remove_prompt_json": ("BOOLEAN", {"default": False}),
                "extract_generation_metadata": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "path", "metadata_json")
    FUNCTION = "load"
    CATEGORY = "image/metadata"

    def load(
        self,
        image,
        save_clean_copy,
        output_directory,
        clean_suffix,
        remove_prompt_json=False,
        extract_generation_metadata=True,
    ):
        input_path = _resolve_input_image(image)
        if input_path is None:
            raise FileNotFoundError("No input image was selected.")

        loader = LoadCleanImageFromDirectory()
        image_tensor, path, metadata_json, _index, _total_count = loader._load_path(
            input_path,
            0,
            1,
            output_directory.strip() or str(input_path.parent),
            save_clean_copy,
            clean_suffix,
            remove_prompt_json,
            extract_generation_metadata,
        )

        return image_tensor, path, metadata_json


class StripWorkflowFromDirectory:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "multiline": False}),
                "pattern": ("STRING", {"default": "*.png", "multiline": False}),
                "recursive": ("BOOLEAN", {"default": False}),
                "output_directory": ("STRING", {"default": "", "multiline": False}),
                "clean_suffix": ("STRING", {"default": "_noworkflow", "multiline": False}),
                "remove_prompt_json": ("BOOLEAN", {"default": False}),
                "extract_generation_metadata": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("report_json", "processed_count", "workflow_removed_count")
    FUNCTION = "strip"
    CATEGORY = "image/metadata"

    def strip(
        self,
        directory,
        pattern,
        recursive,
        output_directory,
        clean_suffix,
        remove_prompt_json=False,
        extract_generation_metadata=True,
    ):
        files = _list_images(directory, pattern, _bool(recursive))
        clean_root = output_directory.strip() or directory
        records = []
        workflow_removed_count = 0

        for source_path in files:
            with Image.open(source_path) as opened:
                source_image = ImageOps.exif_transpose(opened)
                cleaned_path, removed, generation_data = _save_clean_copy(
                    source_image,
                    source_path,
                    clean_root,
                    clean_suffix,
                    metadata_source=opened,
                    remove_prompt_json=_bool(remove_prompt_json),
                    extract_generation_metadata=_bool(extract_generation_metadata),
                )

            if "workflow" in removed:
                workflow_removed_count += 1

            records.append(
                {
                    "source_path": str(source_path),
                    "cleaned_path": cleaned_path,
                    "removed_keys": removed,
                    "extracted_generation_metadata": generation_data,
                }
            )

        return (
            json.dumps(records, indent=2, ensure_ascii=False),
            len(files),
            workflow_removed_count,
        )


NODE_CLASS_MAPPINGS = {
    "CleanMetadataLoader": LoadCleanImageFromDirectory,
    "CleanMetadataImageLoader": LoadCleanSingleImage,
    "CleanMetadataDirectoryLoader": LegacyLoadCleanImageFromDirectory,
    "LoadCleanImageFromDirectory": LegacyLoadCleanImageFromDirectory,
    "StripWorkflowFromDirectory": StripWorkflowFromDirectory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CleanMetadataLoader": "Clean Metadata Loader",
    "CleanMetadataImageLoader": "Clean Metadata Image Loader",
    "CleanMetadataDirectoryLoader": "Clean Metadata Directory Loader (Legacy)",
    "LoadCleanImageFromDirectory": "Load Clean Image(s) From Directory (Legacy)",
    "StripWorkflowFromDirectory": "Strip ComfyUI Workflow From Directory",
}
