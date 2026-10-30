import comfy.samplers
import nodes
from comfy_extras.nodes_custom_sampler import Guider_Basic, SamplerCustom

from .minimax_h3.model import apply_nag

NAG_INPUTS = {
    "nag_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                            "tooltip": "The scale for attention feature extrapolation. Higher values result in stronger negative guidance."}),
    "nag_tau": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1, "round": 0.01,
                          "tooltip": "The normalisation threshold. Higher values result in stronger negative guidance."}),
    "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01,
                            "tooltip": "Blending factor between original and extrapolated attention. Higher values result in stronger negative guidance."}),
    "nag_sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01, "round": 0.01,
                                "tooltip": "NAG will be active only until nag_sigma_end."}),
}
NAG_NEGATIVE = ("CONDITIONING", {
    "tooltip": "The conditioning describing the attributes you want to exclude from the video and audio for NAG."})


class ApplyNAGMiniMaxH3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "nag_negative": NAG_NEGATIVE,
            **NAG_INPUTS,
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/patch/minimax"
    DESCRIPTION = "Patches a MiniMax H3 model with NAG, so any sampler or guider downstream samples with it."

    def patch(self, model, nag_negative, nag_scale, nag_tau, nag_alpha, nag_sigma_end):
        return (apply_nag(model, nag_negative, nag_scale, nag_tau, nag_alpha, nag_sigma_end),)


class NAGGuiderMiniMaxH3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "conditioning": ("CONDITIONING",),
            "nag_negative": NAG_NEGATIVE,
            **NAG_INPUTS,
        }}

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, conditioning, nag_negative, nag_scale, nag_tau, nag_alpha, nag_sigma_end):
        guider = Guider_Basic(apply_nag(model, nag_negative, nag_scale, nag_tau, nag_alpha, nag_sigma_end))
        guider.set_conds(conditioning)
        return (guider,)


class NAGCFGGuiderMiniMaxH3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "nag_negative": NAG_NEGATIVE,
            "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            **NAG_INPUTS,
        }}

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, nag_negative, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end):
        guider = comfy.samplers.CFGGuider(apply_nag(model, nag_negative, nag_scale, nag_tau, nag_alpha, nag_sigma_end))
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        return (guider,)


class KSamplerWithNAGMiniMaxH3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                             "tooltip": "The random seed used for creating the noise."}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000,
                              "tooltip": "The number of steps used in the denoising process."}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                              "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
            **NAG_INPUTS,
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                "tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                "tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
            "positive": ("CONDITIONING", {
                "tooltip": "The conditioning describing the attributes you want to include in the image."}),
            "negative": ("CONDITIONING", {
                "tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
            "nag_negative": NAG_NEGATIVE,
            "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                  "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
        }}

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image, with NAG on MiniMax H3."

    def sample(self, model, seed, steps, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end, sampler_name, scheduler,
               positive, negative, nag_negative, latent_image, denoise=1.0):
        model = apply_nag(model, nag_negative, nag_scale, nag_tau, nag_alpha, nag_sigma_end)
        return nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                     denoise=denoise)


class KSamplerAdvancedWithNAGMiniMaxH3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "add_noise": (["enable", "disable"],),
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            **NAG_INPUTS,
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "nag_negative": NAG_NEGATIVE,
            "latent_image": ("LATENT",),
            "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
            "return_with_leftover_noise": (["disable", "enable"],),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, model, add_noise, noise_seed, steps, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end,
               sampler_name, scheduler, positive, negative, nag_negative,
               latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        model = apply_nag(model, nag_negative, nag_scale, nag_tau, nag_alpha, nag_sigma_end)
        return nodes.common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                     latent_image, denoise=denoise, disable_noise=add_noise == "disable",
                                     start_step=start_at_step, last_step=end_at_step,
                                     force_full_denoise=return_with_leftover_noise != "enable")


class SamplerCustomWithNAGMiniMaxH3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "add_noise": ("BOOLEAN", {"default": True}),
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            **NAG_INPUTS,
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "nag_negative": NAG_NEGATIVE,
            "sampler": ("SAMPLER",),
            "sigmas": ("SIGMAS",),
            "latent_image": ("LATENT",),
        }}

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"

    def sample(self, model, add_noise, noise_seed, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end,
               positive, negative, nag_negative, sampler, sigmas, latent_image):
        model = apply_nag(model, nag_negative, nag_scale, nag_tau, nag_alpha, nag_sigma_end)
        return SamplerCustom.execute(model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas,
                                     latent_image).result


NODE_CLASS_MAPPINGS = {
    "ApplyNAGMiniMaxH3": ApplyNAGMiniMaxH3,
    "NAGGuiderMiniMaxH3": NAGGuiderMiniMaxH3,
    "NAGCFGGuiderMiniMaxH3": NAGCFGGuiderMiniMaxH3,
    "KSamplerWithNAGMiniMaxH3": KSamplerWithNAGMiniMaxH3,
    "KSamplerWithNAGMiniMaxH3 (Advanced)": KSamplerAdvancedWithNAGMiniMaxH3,
    "SamplerCustomWithNAGMiniMaxH3": SamplerCustomWithNAGMiniMaxH3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyNAGMiniMaxH3": "Apply NAG (MiniMax H3)",
    "NAGGuiderMiniMaxH3": "NAGGuider (MiniMax H3)",
    "NAGCFGGuiderMiniMaxH3": "NAGCFGGuider (MiniMax H3)",
    "KSamplerWithNAGMiniMaxH3": "KSamplerWithNAG (MiniMax H3)",
    "KSamplerWithNAGMiniMaxH3 (Advanced)": "KSamplerWithNAG (Advanced, MiniMax H3)",
    "SamplerCustomWithNAGMiniMaxH3": "SamplerCustomWithNAG (MiniMax H3)",
}
