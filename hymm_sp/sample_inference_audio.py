import math
import time
import torch
import random
from loguru import logger
from einops import rearrange
from hymm_sp.diffusion import load_diffusion_pipeline
from hymm_sp.helpers import get_nd_rotary_pos_embed_new
from hymm_sp.inference import Inference
from hymm_sp.diffusion.schedulers import FlowMatchDiscreteScheduler
from hymm_sp.data_kits.audio_preprocessor import encode_audio, get_facemask

def align_to(value, alignment):
    return int(math.ceil(value / alignment) * alignment)

class HunyuanVideoSampler(Inference):
    def __init__(self, args, vae, vae_kwargs, text_encoder, model, text_encoder_2=None, pipeline=None,
                 device=0, logger=None):
        super().__init__(args, vae, vae_kwargs, text_encoder, model, text_encoder_2=text_encoder_2,
                         pipeline=pipeline,  device=device, logger=logger)
        
        self.args = args
        self.pipeline = load_diffusion_pipeline(
            args, 0, self.vae, self.text_encoder, self.text_encoder_2, self.model,
            device=self.device)
        print('load hunyuan model successful... ')

    def get_rotary_pos_embed(self, video_length, height, width, concat_dict={}):
        target_ndim = 3
        ndim = 5 - 2
        if '884' in self.args.vae:
            latents_size = [(video_length-1)//4+1 , height//8, width//8]
        else:
            latents_size = [video_length , height//8, width//8]

        if isinstance(self.model.patch_size, int):
            assert all(s % self.model.patch_size == 0 for s in latents_size), \
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), " \
                f"but got {latents_size}."
            rope_sizes = [s // self.model.patch_size for s in latents_size]
        elif isinstance(self.model.patch_size, list):
            assert all(s % self.model.patch_size[idx] == 0 for idx, s in enumerate(latents_size)), \
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), " \
                f"but got {latents_size}."
            rope_sizes = [s // self.model.patch_size[idx] for idx, s in enumerate(latents_size)]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        head_dim = self.model.hidden_size // self.model.num_heads
        rope_dim_list = self.model.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed_new(rope_dim_list, 
                                                    rope_sizes, 
                                                    theta=self.args.rope_theta, 
                                                    use_real=True,
                                                    theta_rescale_factor=1,
                                                    concat_dict=concat_dict)
        return freqs_cos, freqs_sin

    @torch.no_grad()
    def predict(self, 
                args, batch, wav2vec, feature_extractor, align_instance,
                **kwargs):
        """
        Predict the image from the given text.

        Args:
            prompt (str or List[str]): The input text.
            kwargs:
                size (int): The (height, width) of the output image/video. Default is (256, 256).
                video_length (int): The frame number of the output video. Default is 1.
                seed (int or List[str]): The random seed for the generation. Default is a random integer.
                negative_prompt (str or List[str]): The negative text prompt. Default is an empty string.
                infer_steps (int): The number of inference steps. Default is 100.
                guidance_scale (float): The guidance scale for the generation. Default is 6.0.
                num_videos_per_prompt (int): The number of videos per prompt. Default is 1.    
                verbose (int): 0 for no log, 1 for all log, 2 for fewer log. Default is 1.
                output_type (str): The output type of the image, can be one of `pil`, `np`, `pt`, `latent`.
                    Default is 'pil'.
        """
        
        out_dict = dict()

        prompt = batch['text_prompt'][0]
        image_path = str(batch["image_path"][0])
        audio_path = str(batch["audio_path"][0])
        neg_prompt = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, Lens changes"
        # videoid = batch['videoid'][0]
        fps = batch["fps"].to(self.device)
        audio_prompts = batch["audio_prompts"].to(self.device)
        weight_dtype = audio_prompts.dtype

        audio_prompts = [encode_audio(wav2vec, audio_feat.to(dtype=wav2vec.dtype), fps.item(), num_frames=batch["audio_len"][0]) for audio_feat in audio_prompts]
        audio_prompts = torch.cat(audio_prompts, dim=0).to(device=self.device, dtype=weight_dtype)
        if audio_prompts.shape[1] <= 129:
            audio_prompts = torch.cat([audio_prompts, torch.zeros_like(audio_prompts[:, :1]).repeat(1,129-audio_prompts.shape[1], 1, 1, 1)], dim=1)
        else:
            audio_prompts = torch.cat([audio_prompts, torch.zeros_like(audio_prompts[:, :1]).repeat(1, 5, 1, 1, 1)], dim=1)
        
        wav2vec.to("cpu")
        torch.cuda.empty_cache()

        uncond_audio_prompts = torch.zeros_like(audio_prompts[:,:129])
        motion_exp = batch["motion_bucket_id_exps"].to(self.device)
        motion_pose = batch["motion_bucket_id_heads"].to(self.device)
        
        pixel_value_ref = batch['pixel_value_ref'].to(self.device)  # (b f c h w) 取值范围[0,255]
        face_masks = get_facemask(pixel_value_ref.clone(), align_instance, area=3.0) 

        pixel_value_ref = pixel_value_ref.clone().repeat(1,129,1,1,1)
        uncond_pixel_value_ref = torch.zeros_like(pixel_value_ref)
        pixel_value_ref = pixel_value_ref / 127.5 - 1.             
        uncond_pixel_value_ref = uncond_pixel_value_ref * 2 - 1    
        
        pixel_value_ref_for_vae = rearrange(pixel_value_ref, "b f c h w -> b c f h w")
        uncond_uncond_pixel_value_ref = rearrange(uncond_pixel_value_ref, "b f c h w -> b c f h w")

        pixel_value_llava = batch["pixel_value_ref_llava"].to(self.device)
        pixel_value_llava = rearrange(pixel_value_llava, "b f c h w -> (b f) c h w")
        uncond_pixel_value_llava = pixel_value_llava.clone()
    
        # ========== Encode reference latents ==========
        vae_dtype = self.vae.dtype
        with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):

            self.vae.enable_tiling()
            ref_latents = self.vae.encode(pixel_value_ref_for_vae.clone()).latent_dist.sample()
            uncond_ref_latents = self.vae.encode(uncond_uncond_pixel_value_ref).latent_dist.sample()
            self.vae.disable_tiling()
            if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
                ref_latents.sub_(self.vae.config.shift_factor).mul_(self.vae.config.scaling_factor)
                uncond_ref_latents.sub_(self.vae.config.shift_factor).mul_(self.vae.config.scaling_factor)
            else:
                ref_latents.mul_(self.vae.config.scaling_factor)
                uncond_ref_latents.mul_(self.vae.config.scaling_factor)
            
                
        face_masks = torch.nn.functional.interpolate(face_masks.float().squeeze(2), 
                                                (ref_latents.shape[-2], 
                                                ref_latents.shape[-1]), 
                                                mode="bilinear").unsqueeze(2).to(dtype=ref_latents.dtype)


        size = (batch['pixel_value_ref'].shape[-2], batch['pixel_value_ref'].shape[-1])
        target_length = 129
        target_height = align_to(size[0], 16)
        target_width = align_to(size[1], 16)
        concat_dict = {'mode': 'timecat', 'bias': -1} 
        # concat_dict = {}
        freqs_cos, freqs_sin = self.get_rotary_pos_embed(
            target_length, 
            target_height, 
            target_width, 
            concat_dict)  
        n_tokens = freqs_cos.shape[0]

        generator = torch.Generator(device=self.device).manual_seed(args.seed)

        debug_str = f"""
                    prompt: {prompt}
                image_path: {image_path}
                audio_path: {audio_path}
           negative_prompt: {neg_prompt}
                      seed: {args.seed}
                       fps: {fps.item()}
               infer_steps: {args.infer_steps}
             target_height: {target_height}
              target_width: {target_width}
             target_length: {target_length}
            guidance_scale: {args.cfg_scale}
            """
        self.logger.info(debug_str)
        pipeline_kwargs = {
            "cpu_offload": False
        }
        start_time = time.time()
        samples = self.pipeline(prompt=prompt,                                
                                height=target_height,
                                width=target_width,
                                frame=target_length,
                                num_inference_steps=args.infer_steps,
                                guidance_scale=args.cfg_scale,                      # cfg scale
                         
                                negative_prompt=neg_prompt,
                                num_images_per_prompt=args.num_images,
                                generator=generator,
                                prompt_embeds=None,

                                ref_latents=ref_latents,                            # [1, 16, 1, h//8, w//8]
                                uncond_ref_latents=uncond_ref_latents,
                                pixel_value_llava=pixel_value_llava,                # [1, 3, 336, 336]
                                uncond_pixel_value_llava=uncond_pixel_value_llava,
                                face_masks=face_masks,                              # [b f h w]
                                audio_prompts=audio_prompts, 
                                uncond_audio_prompts=uncond_audio_prompts, 
                                motion_exp=motion_exp, 
                                motion_pose=motion_pose, 
                                fps=fps, 
                                
                                num_videos_per_prompt=1,
                                attention_mask=None,
                                negative_prompt_embeds=None,
                                negative_attention_mask=None,
                                output_type="pil",
                                freqs_cis=(freqs_cos, freqs_sin),
                                n_tokens=n_tokens,
                                data_type='video',
                                is_progress_bar=True,
                                vae_ver=self.args.vae,
                                enable_tiling=self.args.vae_tiling,
                                **pipeline_kwargs
                                )[0]
        if samples is None:
            return None
        out_dict['samples'] = samples
        gen_time = time.time() - start_time
        logger.info(f"Success, time: {gen_time}")
        
        wav2vec.to(self.device)
        
        return out_dict
    
