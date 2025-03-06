import torch
import copy, logging, folder_paths, collections
import comfy.sd
import comfy.utils
import comfy.model_management
import comfy.model_patcher
from .ops import GGMLTensor, GGMLOps, move_patch_to_device
from .dequant import is_quantized, is_torch_compatible
# from gguf_connector import reader as gr # call it from loader.py instead
from .loader import gguf_sd_loader, gguf_clip_loader

def update_folder_names_and_paths(key, targets=[]):
    # check for existing key
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    # find base key & add w/ fallback, sanity check + warning
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    if base and base != orig:
        logging.warning(f"Unknown file list already present on key {key}: {base}")

update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])
update_folder_names_and_paths("clip_gguf", ["text_encoders", "clip"])

# move to loader.py ### ############################################# ###
# IMG_ARCH_LIST = {"flux", "sd1", "sdxl", "sd3", "aura", "ltxv", "hyvid"}
# TXT_ARCH_LIST = {"t5", "t5encoder", "llama"}
# def get_orig_shape(reader, tensor_name):
#     field_key = f"comfy.gguf.orig_shape.{tensor_name}"
#     field = reader.get_field(field_key)
#     if field is None:
#         return None
#     if len(field.types) != 2 or field.types[0] != gr.GGUFValueType.ARRAY or field.types[1] != gr.GGUFValueType.INT32:
#         raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
#     return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))
# def gguf_sd_loader(path, handle_prefix="model.diffusion_model.", return_arch=False):
#     reader = gr.GGUFReader(path)
#     has_prefix = False
#     if handle_prefix is not None:
#         prefix_len = len(handle_prefix)
#         tensor_names = set(tensor.name for tensor in reader.tensors)
#         has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)
#     tensors = []
#     for tensor in reader.tensors:
#         sd_key = tensor_name = tensor.name
#         if has_prefix:
#             if not tensor_name.startswith(handle_prefix):
#                 continue
#             sd_key = tensor_name[prefix_len:]
#         tensors.append((sd_key, tensor))
#     compat = None
#     arch_str = None
#     arch_field = reader.get_field("general.architecture")
#     if arch_field is not None:
#         if len(arch_field.types) != 1 or arch_field.types[0] != gr.GGUFValueType.STRING:
#             raise TypeError(f"Bad type for GGUF general.architecture key: expected string, got {arch_field.types!r}")
#         arch_str = str(arch_field.parts[arch_field.data[-1]], encoding="utf-8")
#         if arch_str not in IMG_ARCH_LIST and arch_str not in TXT_ARCH_LIST:
#             raise ValueError(f"Unexpected architecture type in GGUF file, expected one of flux, sd1, sdxl, t5encoder but got {arch_str!r}")
#     else:
#         from .tools.convert import detect_arch
#         arch_str = detect_arch(set(val[0] for val in tensors)).arch
#         compat = "sd.cpp"
#     state_dict = {}
#     qtype_dict = {}
#     for sd_key, tensor in tensors:
#         tensor_name = tensor.name
#         tensor_type_str = str(tensor.tensor_type)
#         torch_tensor = torch.from_numpy(tensor.data)
#         shape = get_orig_shape(reader, tensor_name)
#         if shape is None:
#             shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
#             if compat == "sd.cpp" and arch_str == "sdxl":
#                 if any([tensor_name.endswith(x) for x in (".proj_in.weight", ".proj_out.weight")]):
#                     while len(shape) > 2 and shape[-1] == 1:
#                         shape = shape[:-1]
#         if tensor.tensor_type in {gr.GGMLQuantizationType.F32, gr.GGMLQuantizationType.F16}:
#             torch_tensor = torch_tensor.view(*shape)
#         state_dict[sd_key] = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)
#         qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1
#     qsd = {k:v for k,v in state_dict.items() if is_quantized(v)}
#     if len(qsd) > 0:
#         max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
#         state_dict[max_key].is_largest_weight = True
#     print("\nggml_sd_loader:")
#     for k,v in qtype_dict.items():
#         print(f" {k:30}{v:3}")
#     if return_arch:
#         return (state_dict, arch_str)
#     return state_dict
# T5_SD_MAP = {
#     "enc.": "encoder.",
#     ".blk.": ".block.",
#     "token_embd": "shared",
#     "output_norm": "final_layer_norm",
#     "attn_q": "layer.0.SelfAttention.q",
#     "attn_k": "layer.0.SelfAttention.k",
#     "attn_v": "layer.0.SelfAttention.v",
#     "attn_o": "layer.0.SelfAttention.o",
#     "attn_norm": "layer.0.layer_norm",
#     "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
#     "ffn_up": "layer.1.DenseReluDense.wi_1",
#     "ffn_down": "layer.1.DenseReluDense.wo",
#     "ffn_gate": "layer.1.DenseReluDense.wi_0",
#     "ffn_norm": "layer.1.layer_norm",
# }
# LLAMA_SD_MAP = {
#     "blk.": "model.layers.",
#     "attn_norm": "input_layernorm",
#     "attn_q": "self_attn.q_proj",
#     "attn_k": "self_attn.k_proj",
#     "attn_v": "self_attn.v_proj",
#     "attn_output": "self_attn.o_proj",
#     "ffn_up": "mlp.up_proj",
#     "ffn_down": "mlp.down_proj",
#     "ffn_gate": "mlp.gate_proj",
#     "ffn_norm": "post_attention_layernorm",
#     "token_embd": "model.embed_tokens",
#     "output_norm": "model.norm",
#     "output.weight": "lm_head.weight",
# }
# def sd_map_replace(raw_sd, key_map):
#     sd = {}
#     for k,v in raw_sd.items():
#         for s,d in key_map.items():
#             k = k.replace(s,d)
#         sd[k] = v
#     return sd

# def llama_permute(raw_sd, n_head, n_head_kv):
#     sd = {}
#     permute = lambda x,h: x.reshape(h, x.shape[0] // h // 2, 2, *x.shape[1:]).swapaxes(1, 2).reshape(x.shape)
#     for k,v in raw_sd.items():
#         if k.endswith(("q_proj.weight", "q_proj.bias")):
#             v.data = permute(v.data, n_head)
#         if k.endswith(("k_proj.weight", "k_proj.bias")):
#             v.data = permute(v.data, n_head_kv)
#         sd[k] = v
#     return sd
# def gguf_clip_loader(path):
#     sd, arch = gguf_sd_loader(path, return_arch=True)
#     if arch in {"t5", "t5encoder"}:
#         sd = sd_map_replace(sd, T5_SD_MAP)
#     elif arch in {"llama"}:
#         temb_key = "token_embd.weight"
#         if temb_key in sd and sd[temb_key].shape != (128320, 4096):
#             print("Warning! token_embd shape may be incorrect for llama 3 model!")
#         sd = sd_map_replace(sd, LLAMA_SD_MAP)
#         sd = llama_permute(sd, 32, 8)
#     else:
#         pass
#     return sd
# ################################################################### ###

class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    patch_on_device = False

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = comfy.utils.get_attr(self.model, key)

        try:
            from comfy.lora import calculate_weight
        except Exception:
            calculate_weight = self.calculate_weight

        patches = self.patches[key]
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = move_patch_to_device(patches, self.load_device if self.patch_on_device else self.offload_device)
            out_weight.patches = [(calculate_weight, patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(
                    weight.to(device=self.offload_device, copy=inplace_update), inplace_update
                )

            if device_to is not None:
                temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)

            out_weight = calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype)

        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, "patches", [])
                if len(patches) > 0:
                    p.patches = []
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)

    mmap_released = False
    def load(self, *args, force_patch_weights=False, **kwargs):
        super().load(*args, force_patch_weights=True, **kwargs)

        if not self.mmap_released:
            linked = []
            if kwargs.get("lowvram_model_memory", 0) > 0:
                for n, m in self.model.named_modules():
                    if hasattr(m, "weight"):
                        device = getattr(m.weight, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
                    if hasattr(m, "bias"):
                        device = getattr(m.bias, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
            if linked:
                print(f"Attempting to release mmap ({len(linked)})")
                for n, m in linked:
                    m.to(self.load_device).to(self.offload_device)
            self.mmap_released = True

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = GGUFModelPatcher
        self.__class__ = src_cls
        n.patch_on_device = getattr(self, "patch_on_device", False)
        return n

class UnetLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
        return {
            "required": {
                "unet_name": (unet_names,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "bootleg"
    TITLE = "Unet Loader (GGUF)"

    def load_unet(self, unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None):
        ops = GGMLOps()

        if dequant_dtype in ("default", None):
            ops.Linear.dequant_dtype = None
        elif dequant_dtype in ["target"]:
            ops.Linear.dequant_dtype = dequant_dtype
        else:
            ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)

        if patch_dtype in ("default", None):
            ops.Linear.patch_dtype = None
        elif patch_dtype in ["target"]:
            ops.Linear.patch_dtype = patch_dtype
        else:
            ops.Linear.patch_dtype = getattr(torch, patch_dtype)

        unet_path = folder_paths.get_full_path("unet", unet_name)
        sd = gguf_sd_loader(unet_path)
        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}
        )
        if model is None:
            logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
            raise RuntimeError("ERROR: Could not detect model type of: {}".format(unet_path))
        model = GGUFModelPatcher.clone(model)
        model.patch_on_device = patch_on_device
        return (model,)

class UnetLoaderGGUFAdvanced(UnetLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
        return {
            "required": {
                "unet_name": (unet_names,),
                "dequant_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_on_device": ("BOOLEAN", {"default": False}),
            }
        }
    TITLE = "Unet Loader (GGUF/Advanced)"

CLIP_ENUM_MAP = {
    "stable_diffusion": "STABLE_DIFFUSION",
    "stable_cascade":   "STABLE_CASCADE",
    "stable_audio":     "STABLE_AUDIO",
    "sdxl":             "STABLE_DIFFUSION",
    "sd3":              "SD3",
    "flux":             "FLUX",
    "mochi":            "MOCHI",
    "ltxv":             "LTXV",
    "hunyuan_video":    "HUNYUAN_VIDEO",
    "pixart":           "PIXART",
    "wan":              "WAN",
}

def get_clip_type(name):
    enum_name = CLIP_ENUM_MAP.get(name, None)
    if enum_name is None:
        raise ValueError(f"Unknown CLIP model type {name}") 
    clip_type = getattr(comfy.sd.CLIPType, CLIP_ENUM_MAP[name], None)
    if clip_type is None:
        raise ValueError(f"Unsupported CLIP model type {name} (Update ComfyUI)")
    return clip_type

class CLIPLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name": (s.get_filename_list(),),
                # "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart"],),
                "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "wan"],)
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "bootleg"
    TITLE = "CLIPLoader (GGUF)"

    @classmethod
    def get_filename_list(s):
        files = []
        files += folder_paths.get_filename_list("clip")
        files += folder_paths.get_filename_list("clip_gguf")
        return sorted(files)

    def load_data(self, ckpt_paths):
        clip_data = []
        for p in ckpt_paths:
            if p.endswith(".gguf"):
                sd = gguf_clip_loader(p)
            else:
                sd = comfy.utils.load_torch_file(p, safe_load=True)
            clip_data.append(sd)
        return clip_data

    def load_patcher(self, clip_paths, clip_type, clip_data):
        clip = comfy.sd.load_text_encoder_state_dicts(
            clip_type = clip_type,
            state_dicts = clip_data,
            model_options = {
                "custom_operations": GGMLOps,
                "initial_device": comfy.model_management.text_encoder_offload_device()
            },
            embedding_directory = folder_paths.get_folder_paths("embeddings"),
        )
        clip.patcher = GGUFModelPatcher.clone(clip.patcher)
        return clip

    def load_clip(self, clip_name, type="stable_diffusion"):
        clip_path = folder_paths.get_full_path("clip", clip_name)
        return (self.load_patcher([clip_path], get_clip_type(type), self.load_data([clip_path])),)

class DualCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = (s.get_filename_list(), )
        return {
            "required": {
                "clip_name1": file_options,
                "clip_name2": file_options,
                "type": (("sdxl", "sd3", "flux"), ),
            }
        }

    TITLE = "DualCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, type):
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        clip_paths = (clip_path1, clip_path2)
        return (self.load_patcher(clip_paths, get_clip_type(type), self.load_data(clip_paths)),)

class TripleCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = (s.get_filename_list(), )
        return {
            "required": {
                "clip_name1": file_options,
                "clip_name2": file_options,
                "clip_name3": file_options,
            }
        }

    TITLE = "TripleCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, clip_name3, type="sd3"):
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        clip_path3 = folder_paths.get_full_path("clip", clip_name3)
        clip_paths = (clip_path1, clip_path2, clip_path3)
        return (self.load_patcher(clip_paths, get_clip_type(type), self.load_data(clip_paths)),)

NODE_CLASS_MAPPINGS = {
    "UnetLoaderGGUF": UnetLoaderGGUF,
    "CLIPLoaderGGUF": CLIPLoaderGGUF,
    "DualCLIPLoaderGGUF": DualCLIPLoaderGGUF,
    "TripleCLIPLoaderGGUF": TripleCLIPLoaderGGUF,
    "UnetLoaderGGUFAdvanced": UnetLoaderGGUFAdvanced,
}
