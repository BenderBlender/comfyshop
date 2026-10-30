# ComfyUI-NAG-MiniMaxH3

[Normalized Attention Guidance](https://chendaryen.github.io/NAG.github.io/) (NAG) for the MiniMax H3 audio-video model in ComfyUI. The nodes work like the ones in [ComfyUI-NAG](https://github.com/ChenDarYen/ComfyUI-NAG), which does not support H3.

Needs a ComfyUI build with native MiniMax H3 support (`comfy/ldm/minimax`).

## Install

copy the `ComfyUI-NAG-MiniMaxH3` folder into `ComfyUI/custom_nodes` and restart ComfyUI. It works with or without ComfyUI-NAG installed, the node names do not clash.

## Nodes

Replace the stock node with its NAG version, then plug a negative prompt (`CLIPTextEncode` with the H3 text encoder) into `nag_negative`:

| stock node | NAG node |
| --- | --- |
| `KSampler` | `KSamplerWithNAG (MiniMax H3)` |
| `KSampler (Advanced)` | `KSamplerWithNAG (Advanced, MiniMax H3)` |
| `SamplerCustom` | `SamplerCustomWithNAG (MiniMax H3)` |
| `BasicGuider` | `NAGGuider (MiniMax H3)` |
| `CFGGuider` | `NAGCFGGuider (MiniMax H3)` |

`Apply NAG (MiniMax H3)` patches the `MODEL` instead, so any sampler or guider after it uses NAG. Applying NAG a second time replaces the earlier settings.

The NAG inputs are the same as in ComfyUI-NAG (the guiders have no `latent_image` input, H3 samples one video at a time): `nag_scale`, `nag_tau`, `nag_alpha` and `nag_sigma_end` (NAG is active only while sigma >= `nag_sigma_end`). The defaults are ComfyUI-NAG's; tune `nag_tau` and `nag_alpha` first, then only `nag_scale`. For image-to-video, reference-to-video and keyframe guides, use lower `nag_tau` and `nag_alpha` to keep the references faithful.

## How it works

H3 is a single-stream DiT: text, guide/reference rows, audio and video tokens are packed into one sequence with one attention per block. NAG runs a second, negative-prompt text stream next to it, like ComfyUI-NAG does for Flux single blocks:

- the negative text rows are appended to the packed sequence of every block;
- the video, audio and guide rows attend once with the positive text and once with the negative text, and their attention output is replaced with the NAG mix of the two;
- the negative text sits right where the positive text ends on the RoPE time axis, which is the layout a sequence built around the negative prompt would have.

NAG is installed through ComfyUI's model patches (a diffusion-model wrapper plus `dit` block patches), so the stock samplers keep handling the nested video+audio latent. It sits right around each block, inside other block patches such as `Apply MiniMax H3 Fun ControlNet`.

Cost: while NAG is active every block runs a second attention over the non-text rows, about 2x the attention time of the positive pass. Use `nag_sigma_end` to stop it early.

`Model Sparse Attention` replaces the block patches of every node before it, so put it before NAG. On the steps where NAG is active, the negative pass runs dense, and the positive pass goes through the sparse attention's generic kernel (dense with VSA).
