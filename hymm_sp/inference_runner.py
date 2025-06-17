import os
import numpy as np
from pathlib import Path
from loguru import logger
import imageio
import torch
# import torch.distributed # Not strictly needed for single inference, can be removed if it causes issues without full distributed setup
# from torch.utils.data.distributed import DistributedSampler # Not needed for single inference
# from torch.utils.data import DataLoader # Not needed for single inference
from argparse import Namespace # To create args object

from hymm_sp.sample_inference_audio import HunyuanVideoSampler
# from hymm_sp.data_kits.audio_dataset import VideoAudioTextLoaderVal # Will replace this logic
from hymm_sp.data_kits.face_align import AlignImage
from hymm_sp.data_kits.audio_preprocessor import AudioPreprocessor # For audio duration

from transformers import WhisperModel, AutoFeatureExtractor
from einops import rearrange

# Attempt to get MODEL_BASE from environment, default to './weights' if not set
MODEL_BASE = os.environ.get('MODEL_BASE', './weights')
CPU_OFFLOAD_ENV = os.environ.get('CPU_OFFLOAD', '1')

# Global cache for models to avoid reloading on every call in a serverless environment
# However, for RunPod, it's often better to load models inside the handler or a dedicated init function
# For now, let's keep them inside process_video_avatar_job and optimize later if needed.
# hunyuan_video_sampler_global = None
# wav2vec_global = None
# align_instance_global = None
# feature_extractor_global = None

def process_video_avatar_job(image_path: str, audio_path: str, output_base_path: str, job_id: str):
    global MODEL_BASE, CPU_OFFLOAD_ENV
    # global hunyuan_video_sampler_global, wav2vec_global, align_instance_global, feature_extractor_global

    logger.info(f"Starting video generation for job_id: {job_id}")
    logger.info(f"Using MODEL_BASE: {MODEL_BASE}")
    logger.info(f"Image path: {image_path}, Audio path: {audio_path}")
    logger.info(f"Output base path: {output_base_path}")

    # --- 1. Configure Arguments ---
    args = Namespace(
        ckpt=f"{MODEL_BASE}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt",
        input=None, # Was CSV path, now handled by direct image_path, audio_path
        sample_n_frames=129, # Max frames, will be adjusted by audio length
        seed=128,
        image_size=704, # This is the target video resolution height, width will be adjusted
        cfg_scale=7.5,
        infer_steps=50,
        use_deepcache=1,
        flow_shift_eval_video=5.0,
        save_path=output_base_path, # Base path for saving
        use_fp8=True,
        cpu_offload=(CPU_OFFLOAD_ENV == '1'),
        infer_min=True, # This might mean use minimal necessary frames based on audio
        save_path_suffix="", # Not needed as we construct unique names with job_id
        world_size=1,
        local_rank=0,
        torch_dtype="bf16", # from original parse_args default
        pipeline_type="hunyuan_video_audio", # from original parse_args default
        text_encoder_name=f"{MODEL_BASE}/ckpts/hunyuan-video-t2v-base/mt5-xl", # from original
        text_encoder_2_name=f"{MODEL_BASE}/ckpts/hunyuan-video-t2v-base/clip-vit-large-patch14-336", # from original
        vae_name=f"{MODEL_BASE}/ckpts/hunyuan-video-t2v-base/sd-vae-ft-ema", # from original
        transformer_pretrain_path=f"{MODEL_BASE}/ckpts/hunyuan-video-t2v-base/transformer", # from original
        max_frames=129, # from original parse_args default
        target_fps=25, # from original parse_args default, but sample_gpu_poor uses 16
        sample_fps=16, # As used in sample_gpu_poor for output, but VideoAudioTextLoaderVal might use target_fps
        frame_stride=1, # from original parse_args default
        audio_sample_rate=16000, # from original parse_args default
        audio_n_fft=None, # from original parse_args default
        audio_hop_length=None, # from original parse_args default
        max_audio_len=129, # from original parse_args default, relates to sample_n_frames
        image_align_method="face_align", # from original parse_args default
        num_frames_condition=None, # from original parse_args default
        num_inference_steps=50, # from original parse_args default (same as infer_steps)
        guidance_scale=7.5, # from original parse_args default (same as cfg_scale)
        video_latent_height=None, # from original parse_args default
        video_latent_width=None, # from original parse_args default
        use_temporal_conv=True # from original parse_args default
    )

    models_root_path = Path(args.ckpt)
    if not models_root_path.exists():
        logger.error(f"Checkpoint path does not exist: {models_root_path}")
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    output_dir = Path(output_base_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Load Models (Consider loading these once if script is imported) ---
    # For now, loading per call to match original structure closely.
    # This might be slow in a serverless context.
    rank = 0 # Assuming single process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # if hunyuan_video_sampler_global is None:
    logger.info("Loading HunyuanVideoSampler...")
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(args.ckpt, args=args, device=device)
    args = hunyuan_video_sampler.args # Get updated args from sampler
    if args.cpu_offload:
        from diffusers.hooks import apply_group_offloading
        onload_device = torch.device("cuda") # Should be same as device if cuda is available
        apply_group_offloading(hunyuan_video_sampler.pipeline.transformer, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=1)
    #     hunyuan_video_sampler_global = hunyuan_video_sampler
    # else:
    #     hunyuan_video_sampler = hunyuan_video_sampler_global
    #     args = hunyuan_video_sampler.args # Ensure args are consistent

    whisper_path = Path(f"{MODEL_BASE}/ckpts/whisper-tiny/")
    # if wav2vec_global is None:
    if not whisper_path.exists():
        raise FileNotFoundError(f"Whisper model path not found: {whisper_path}")
    logger.info(f"Loading WhisperModel from: {whisper_path}")
    wav2vec = WhisperModel.from_pretrained(str(whisper_path)).to(device=device, dtype=torch.float32) # Original uses float32
    wav2vec.requires_grad_(False)
    #     wav2vec_global = wav2vec
    # else:
    #     wav2vec = wav2vec_global

    # if align_instance_global is None:
    det_align_base_dir = Path(f'{MODEL_BASE}/ckpts/det_align/')
    if not det_align_base_dir.exists():
        raise FileNotFoundError(f"Detection and alignment model base directory not found: {det_align_base_dir}")
    det_path = det_align_base_dir / 'detface.pt'
    if not det_path.exists():
        raise FileNotFoundError(f"Face detection model not found: {det_path}")
    logger.info("Loading AlignImage...")
    align_instance = AlignImage(str(device), det_path=str(det_path))
    #     align_instance_global = align_instance
    # else:
    #     align_instance = align_instance_global

    # if feature_extractor_global is None:
    logger.info(f"Loading AutoFeatureExtractor from: {whisper_path}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(str(whisper_path))
    #     feature_extractor_global = feature_extractor
    # else:
    #     feature_extractor = feature_extractor_global

    # --- 3. Prepare Data Batch (Replaces VideoAudioTextLoaderVal for single item) ---
    logger.info("Preparing data batch...")

    # Get audio duration to determine number of frames
    # This logic might be part of VideoAudioTextLoaderVal or AudioPreprocessor
    # For now, let's simulate getting audio length.
    # AudioPreprocessor from the original repo might be useful here.
    audio_processor = AudioPreprocessor(sr=args.audio_sample_rate, n_fft=args.audio_n_fft, hop_length=args.audio_hop_length, target_fps=args.sample_fps) # Use sample_fps

    try:
        # This is a simplified way to get num_frames. Original loader has more complex logic.
        # It might involve whisper to get features and then calculate length.
        # For now, let's assume audio_len is derived based on sample_n_frames or actual audio length.
        # The `predict` function seems to expect `batch["audio_len"][0]` to be set.
        # If `args.infer_min` is True, sample_gpu_poor.py sets `batch["audio_len"][0] = 129`
        # This needs to be the number of latent frames for audio.
        # audio_features_len = audio_processor.get_audio_features_len(audio_path, None, args.max_audio_len * hunyuan_video_sampler.pipeline.transformer.config.temporal_compression_ratio)
        # audio_latent_len = min(args.max_frames, audio_features_len)

        # The original script does:
        # If args.infer_min: batch["audio_len"][0] = 129
        # This seems to be the number of *video* frames to generate, up to sample_n_frames.
        # Let's try to calculate it based on actual audio duration.
        # The `HunyuanVideoSampler.predict` expects `audio_feat` which is processed by Whisper.
        # The `VideoAudioTextLoaderVal` seems to do this:
        #   audio_feat, _, audio_len_sec = self.audio_preprocessor.process(audio_path, text="", sr=self.sr, target_fps=self.target_fps)
        #   audio_len = min(self.max_audio_len, int(audio_len_sec * self.target_fps)) # Number of video frames
        #   audio_latent_len = audio_len // self.temporal_compression_ratio # Number of latent frames for audio

        # Let's try to replicate parts of VideoAudioTextLoaderVal logic for a single item
        audio_data = audio_processor.read_audio(audio_path, args.audio_sample_rate)
        duration_sec = len(audio_data) / args.audio_sample_rate

        # audio_len here is number of video frames based on audio duration and target FPS for generation
        # This should match how `sample_n_frames` is used.
        # `args.sample_n_frames` is the max.
        num_video_frames = min(args.sample_n_frames, int(duration_sec * args.sample_fps)) # Use sample_fps as it's used for output
        if num_video_frames == 0: # Ensure at least some frames if audio is very short
             num_video_frames = 16 # A small default
        logger.info(f"Calculated num_video_frames based on audio duration ({duration_sec:.2f}s) and sample_fps ({args.sample_fps}): {num_video_frames}")

        if args.infer_min: # Override if infer_min is set (as in original script)
            num_video_frames = args.max_frames # max_frames is 129 by default
            logger.info(f"args.infer_min is True, setting num_video_frames to args.max_frames: {num_video_frames}")


        # The batch needs to be on the device
        batch = {
            "fps": torch.tensor([args.sample_fps], device=device), # FPS for the output video
            "videoid": [job_id], # Use job_id as videoid
            "audio_path": [audio_path],
            "image_path": [image_path],
            "text": [""], # Assuming no text prompt is used for this specific task
            "text_neg": [""], # Negative prompt
            "audio_len": torch.tensor([num_video_frames], device=device, dtype=torch.long), # Number of video frames to generate
            "target_fps": torch.tensor([args.sample_fps], device=device), # FPS of the original audio/video source (used for feature extraction rate)
            "image_cond": None, # Will be prepared by align_instance in predict
            "audio_feat": None, # Will be prepared by wav2vec + feature_extractor in predict
            # These might be needed by the pipeline if not handled internally
            "height": args.image_size, # Target height
            "width": args.image_size, # Target width (will be adjusted from image)
            "num_frames": num_video_frames, # num_frames for pipeline
            "guidance_scale": args.cfg_scale,
            "num_inference_steps": args.infer_steps,
            "seed": args.seed,
            # The following are expected by HunyuanVideoSampler.predict based on its usage of args
            "use_fp8": args.use_fp8,
            "cpu_offload": args.cpu_offload,
            "use_deepcache": args.use_deepcache,
            "flow_shift_eval_video": args.flow_shift_eval_video,
        }
        logger.info(f"Prepared batch: { {k: v.shape if isinstance(v, torch.Tensor) else v for k,v in batch.items()} }")

    except Exception as e:
        logger.error(f"Error preparing batch or audio features for {job_id}: {e}")
        raise

    # --- 4. Run Inference ---
    logger.info(f"Running inference for {job_id}...")
    try:
        # `predict` will handle image alignment and audio feature extraction internally
        samples = hunyuan_video_sampler.predict(args, batch, wav2vec, feature_extractor, align_instance)

        # samples['samples'] is the denoised latent, (bs, channels, num_latent_frames, h_latent, w_latent)
        # For video, channels=16 (from original code comment, but could be 4 for vae)
        # num_latent_frames = num_video_frames // temporal_compression_ratio
        # Example: sample = samples['samples'][0].unsqueeze(0) # (1, 16, t//4, h//8, w//8)
        # The pipeline should return decoded frames directly if possible, or VAE decode here.
        # The `HunyuanVideoSampler.predict` already does VAE decoding and returns pixel space video.

        # samples['samples'] should be (bs, num_video_frames, H, W, C) in pixel space
        video_tensor = samples['samples'][0] # Get first item from batch, shape (num_video_frames, H, W, C)
        logger.info(f"Output video tensor shape: {video_tensor.shape}")

        # Ensure it's on CPU and converted to numpy uint8
        video_numpy = (video_tensor.data.cpu().numpy() * 255.).astype(np.uint8)

    except Exception as e:
        logger.error(f"Error during inference for {job_id}: {e}")
        # Perform cleanup of models from GPU memory if possible
        del hunyuan_video_sampler, wav2vec, align_instance, feature_extractor, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    # --- 5. Save Output ---
    output_video_filename = f"{job_id}.mp4"
    output_video_path = output_dir / output_video_filename

    # Temporary path for video without audio (if ffmpeg step is separate)
    temp_raw_video_path = output_dir / f"{job_id}_raw.mp4"

    logger.info(f"Saving video to {temp_raw_video_path} with FPS: {batch['fps'][0].item()}")
    try:
        imageio.mimsave(str(temp_raw_video_path), video_numpy, fps=batch['fps'][0].item(), quality=8) # quality can be adjusted
    except Exception as e:
        logger.error(f"Error saving raw video with imageio: {e}")
        raise

    # Combine with audio using ffmpeg
    logger.info(f"Combining video with audio. Video: {temp_raw_video_path}, Audio: {audio_path}, Output: {output_video_path}")
    ffmpeg_cmd = (
        f"ffmpeg -i '{str(temp_raw_video_path)}' -i '{audio_path}' "
        f"-c:v libx264 -c:a aac -shortest " # Re-encode video to ensure compatibility, use aac for audio
        f"-y '{str(output_video_path)}' -loglevel error" # Overwrite and suppress verbose logs
    )
    try:
        os.system(ffmpeg_cmd)
        if os.path.exists(temp_raw_video_path):
            os.remove(temp_raw_video_path) # Clean up raw video file
        logger.info(f"Successfully muxed video and audio to {output_video_path}")
    except Exception as e:
        logger.error(f"Error during ffmpeg muxing: {e}")
        # If ffmpeg fails, the raw video might still be useful for debugging
        # Depending on requirements, may want to return temp_raw_video_path or raise
        raise RuntimeError(f"ffmpeg command failed: {ffmpeg_cmd}") from e

    if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) == 0:
        logger.error(f"Output video file not created or is empty after ffmpeg: {output_video_path}")
        # Fallback or further error handling
        if os.path.exists(temp_raw_video_path): # If raw video is there, maybe it's an audio issue
             logger.warning(f"FFMPEG failed, but raw video exists: {temp_raw_video_path}. Consider returning this or handling error.")
             # For now, let's assume this is an error state for the job if muxing fails.
        raise RuntimeError(f"Output video {output_video_path} was not created or is empty.")

    # --- 6. Cleanup (explicitly delete large objects) ---
    logger.info(f"Cleaning up resources for job {job_id}")
    del hunyuan_video_sampler, wav2vec, align_instance, feature_extractor, samples, video_tensor, video_numpy, batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Video generation for {job_id} completed. Output: {output_video_path}")
    return str(output_video_path)
```
