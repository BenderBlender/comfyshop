"""NAG for the MiniMax H3 audio-video DiT.

H3 is single-stream: text, condition/reference rows, audio and video share one packed
sequence ([text | cond | audio | video]) and one attention per block. NAG runs a negative
text stream next to it, the way ComfyUI-NAG does for Flux single blocks: every block gets
the negative text rows appended to the packed sequence, the non-text rows attend once to the
positive text and once to the negative text, and their attention output is replaced by the
NAG mix of the two. The non-text rows are shared by both streams, so the cost is one extra
attention per block over the non-text rows plus the negative text.

The negative text sits on the RoPE time axis so that it ends where the positive text ends,
right before the cond/target rows. RoPE only sees relative positions, so this is exactly the
layout a native sequence built around the negative prompt would have, shifted as a whole.

Everything is installed through the model patcher (a DIFFUSION_MODEL wrapper plus
dit/double_block patches), so stock samplers and guiders keep handling the nested AV latent.
"""

import functools
import logging

import torch

import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension
import comfy.quant_ops
from comfy.ldm.minimax.model import MiniMaxH3Model, rope_rotation_table
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

from ..utils import nag_

NAG_WRAPPER_KEY = "nag_minimax_h3"
TEXT_TAG = 1  # modality tag of plain text rows, see MiniMaxH3Model._forward


def get_minimax_h3_model(model_patcher):
    model = model_patcher.get_model_object("diffusion_model")
    model = getattr(model, "_orig_mod", model)
    return model if isinstance(model, MiniMaxH3Model) else None


def _attend(attn, q, k, v, transformer_options):
    # [S, heads, head_dim] -> [S, heads * head_dim], as in comfy.ldm.minimax.model.Attention.forward
    q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
    k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
    v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
    out = optimized_attention(q, k, v, attn.heads, preferred_attention=attn.comfy_attention, mask=None,
                              skip_reshape=True, transformer_options=transformer_options)
    return out.squeeze(0)


def nag_attention(x, rope_freqs=None, transformer_options={}, *, attn, seq_len, text_len,
                  nag_scale, nag_tau, nag_alpha):
    """H3 block attention over [positive sequence | negative text] rows.

    Rows [0, seq_len) are the packed positive sequence, text first; the rest is the negative
    text. The positive sequence attends to itself, the non-text rows and the negative text
    attend to each other, and the non-text rows get the NAG mix of both outputs.
    """
    s = x.shape[0]
    q, k, v = attn.qkv_proj(x).split(attn.heads * attn.head_dim, dim=-1)
    v = v.view(s, attn.heads, attn.head_dim)
    if rope_freqs is not None:
        q = q.view(1, s, attn.heads, attn.head_dim)
        k = k.view(1, s, attn.heads, attn.head_dim)
        qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
        q = q[0]
        k = k[0]
    else:
        q = attn.q_norm(q.view(s, attn.heads, attn.head_dim))
        k = attn.k_norm(k.view(s, attn.heads, attn.head_dim))

    out = _attend(attn, q[:seq_len], k[:seq_len], v[:seq_len], transformer_options)
    out_negative = _attend(attn, q[text_len:], k[text_len:], v[text_len:], transformer_options)
    del q, k, v

    # out_negative rows are [non-text | negative text]: the guidance overwrites its non-text part
    nag_(out[text_len:], out_negative[:seq_len - text_len], nag_scale, nag_tau, nag_alpha)
    out = torch.cat([out[:text_len], out_negative])
    return attn.out_proj(out)


def _text_mod_segments(start, length, row_base, tags):
    # same tag runs MiniMaxH3Model._forward builds for the positive text span
    if tags is None:
        return [(start, start + length, row_base + TEXT_TAG)]
    segments = []
    run_start = 0
    for i in range(1, length + 1):
        if i == length or tags[i] != tags[run_start]:
            segments.append((start + run_start, start + i, row_base + tags[run_start]))
            run_start = i
    return segments


class NAGMiniMaxH3State:
    """Negative text stream of one diffusion model call, carried from block to block."""

    def __init__(self, negative):
        self.negative = negative  # [L_neg, hidden] negative text rows entering the next block
        self.packed = None  # last [positive sequence | negative text] block output
        self.mod_segments_src = None
        self.mod_segments = None
        self.rope_freqs_src = None
        self.rope_freqs = None

    def pack(self, h):
        packed = self.packed
        if (packed is not None and packed.shape[0] == h.shape[0] + self.negative.shape[0]
                and packed.data_ptr() == h.data_ptr() and packed.stride() == h.stride()):
            # h is the positive view of the previous block output, which the blocks update in place
            return packed
        return torch.cat([h, self.negative.to(h.dtype)])

    def unpack(self, out, seq_len):
        self.packed = out
        self.negative = out[seq_len:]

    def get_mod_segments(self, mod_segments, seq_len, tags):
        if self.mod_segments_src is not mod_segments:
            # text rows run at the video timestep: their mod row is t_row * 3 + modality tag
            row_base = mod_segments[0][2] // 3 * 3
            self.mod_segments = list(mod_segments) + _text_mod_segments(
                seq_len, self.negative.shape[0], row_base, tags)
            self.mod_segments_src = mod_segments
        return self.mod_segments

    def get_rope_freqs(self, rope_freqs, model, text_len):
        if rope_freqs is None:
            return None
        if self.rope_freqs_src is not rope_freqs:
            n = self.negative.shape[0]
            position_ids = torch.zeros(n, 3, dtype=torch.float64)
            position_ids[:, 0] = torch.arange(text_len - n, text_len, dtype=torch.float64)
            negative = rope_rotation_table(model.rope_freqs(position_ids, rope_freqs.device), rope_freqs.dtype)
            self.rope_freqs = torch.cat([rope_freqs, negative], dim=1)
            self.rope_freqs_src = rope_freqs
        return self.rope_freqs


class NAGMiniMaxH3Patch:
    def __init__(self, nag_negative, nag_scale, nag_tau, nag_alpha, nag_sigma_end):
        self.negative_context = nag_negative[0][0][:1]
        tags = nag_negative[0][1].get("minimax_token_tags", None)
        self.negative_tags = None
        if tags is not None and tags.numel() == self.negative_context.shape[1]:
            self.negative_tags = [int(t) for t in tags.view(-1).tolist()]
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        self.nag_sigma_end = nag_sigma_end
        self.model = None
        self.negative_embeds = {}
        self.state = None
        self.warned_attention_override = False
        self.warned_no_blocks = False

    def register(self, model_patcher):
        self.model = get_minimax_h3_model(model_patcher)
        if self.model is None:
            raise ValueError("NAG (MiniMax H3) only supports MiniMax H3 models")
        # applying NAG again replaces the earlier settings: its block patches stay but go idle
        model_patcher.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, NAG_WRAPPER_KEY)
        model_patcher.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, NAG_WRAPPER_KEY,
                                           self.diffusion_model_wrapper)
        blocks_replace = model_patcher.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
        for i in range(len(self.model.blocks)):
            previous = blocks_replace.get(("double_block", i))
            model_patcher.set_model_patch_replace(NAGMiniMaxH3BlockPatch(self, i, previous), "dit", "double_block", i)

    def is_active(self, x, transformer_options):
        if self.nag_scale <= 1.0 or x[0].shape[0] != 1:
            return False
        sigmas = transformer_options.get("sigmas", None)
        if sigmas is None or not bool(torch.all(sigmas >= self.nag_sigma_end)):
            return False
        return 0 in transformer_options.get("cond_or_uncond", [])

    def get_negative_embeds(self, context):
        key = (context.device, context.dtype)
        if key not in self.negative_embeds:
            # condition_proj + token refiner, as MiniMaxH3.extra_conds does for the positive prompt
            negative = self.negative_context.to(device=context.device, dtype=context.dtype)
            self.negative_embeds[key] = self.model.preprocess_text_embeds(negative)[0]
        return self.negative_embeds[key]

    def diffusion_model_wrapper(self, executor, x, timestep, context, transformer_options={}, **kwargs):
        self.state = None
        if self.is_active(x, transformer_options):
            # the negative stream outlives single blocks: keep it out of the block allocation scope
            with comfy.model_prefetch.pause_malloc_graph():
                self.state = NAGMiniMaxH3State(self.get_negative_embeds(context))
        try:
            out = executor(x, timestep, context, transformer_options, **kwargs)
            if self.state is not None and self.state.packed is None and not self.warned_no_blocks:
                logging.warning("NAG (MiniMax H3): no DiT block ran with NAG; a node applied after NAG "
                                "(e.g. Model Sparse Attention) probably replaced its block patches")
                self.warned_no_blocks = True
            return out
        finally:
            self.state = None

    def run_block(self, block_index, args, original_block):
        state = self.state
        h = args["img"]
        layout = args["layout"]
        seq_len = h.shape[0]
        _, text_len, kind = layout.segments[0]
        if state is None or kind != "text" or layout.seq_len != seq_len:
            return original_block(args)
        if args.get("attention", None) is not None and not self.warned_attention_override:
            logging.warning("NAG (MiniMax H3): replacing another patch's block attention (e.g. Model Sparse Attention) while NAG is active")
            self.warned_attention_override = True

        with comfy.model_prefetch.pause_malloc_graph():
            packed = state.pack(h)
            mod_segments = state.get_mod_segments(args["mod_segments"], seq_len, self.negative_tags)
            rope_freqs = state.get_rope_freqs(args["rope_freqs"], self.model, text_len)
        attention = functools.partial(
            nag_attention, attn=self.model.blocks[block_index].attn, seq_len=seq_len, text_len=text_len,
            nag_scale=self.nag_scale, nag_tau=self.nag_tau, nag_alpha=self.nag_alpha)

        out = original_block({**args, "img": packed, "mod_segments": mod_segments,
                              "rope_freqs": rope_freqs, "attention": attention})
        state.unpack(out["img"], seq_len)
        return {**out, "img": out["img"][:seq_len]}

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            self.negative_embeds = {}
        return self

    def cleanup(self):
        self.negative_embeds = {}
        self.state = None


class NAGMiniMaxH3BlockPatch:
    def __init__(self, nag_patch, block_index, previous):
        self.nag_patch = nag_patch
        self.block_index = block_index
        self.previous = previous

    def __call__(self, args, extra_args):
        if self.nag_patch.state is not None:
            # NAG goes innermost, right around the block, so other patches keep seeing the positive sequence only
            original_block = extra_args["original_block"]
            extra_args = {**extra_args, "original_block": functools.partial(
                self.nag_patch.run_block, self.block_index, original_block=original_block)}
        if self.previous is None:
            return extra_args["original_block"](args)
        return self.previous(args, extra_args)

    def to(self, device_or_dtype):
        self.nag_patch.to(device_or_dtype)
        if hasattr(self.previous, "to"):
            self.previous = self.previous.to(device_or_dtype)
        return self

    def cleanup(self):
        self.nag_patch.cleanup()
        if hasattr(self.previous, "cleanup"):
            self.previous.cleanup()

    def models(self):
        if hasattr(self.previous, "models"):
            return self.previous.models()
        return []


def apply_nag(model, nag_negative, nag_scale, nag_tau, nag_alpha, nag_sigma_end):
    m = model.clone()
    NAGMiniMaxH3Patch(nag_negative, nag_scale, nag_tau, nag_alpha, nag_sigma_end).register(m)
    return m
