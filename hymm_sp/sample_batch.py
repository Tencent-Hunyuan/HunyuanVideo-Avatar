import os
import torch
import numpy as np
import imageio
from pathlib import Path
from loguru import logger
from einops import rearrange
import torch.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from hymm_sp.config import parse_args
from hymm_sp.sample_inference_audio import HunyuanVideoSampler
from hymm_sp.data_kits.audio_dataset import VideoAudioTextLoaderVal
from hymm_sp.data_kits.data_tools import save_videos_grid
from hymm_sp.data_kits.face_align import AlignImage
from hymm_sp.modules.parallel_states import (
    initialize_distributed,
    nccl_info,
)

from transformers import WhisperModel
from transformers import AutoFeatureExtractor

MODEL_OUTPUT_PATH = os.environ.get('MODEL_BASE')


def main():
    args = parse_args()
    models_root_path = Path(args.ckpt)
    print("*"*20) 
    initialize_distributed(args.seed)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    print("+"*20)
    # Create save folder to save the samples
    save_path = args.save_path 
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    rank = 0
    vae_dtype = torch.float16
    device = torch.device("cuda")
    if nccl_info.sp_size > 1:
        device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        rank = torch.distributed.get_rank()

    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(args.ckpt, args=args, device=device)
    # Get the updated args
    args = hunyuan_video_sampler.args

    wav2vec = WhisperModel.from_pretrained(f"{MODEL_OUTPUT_PATH}/ckpts/whisper-tiny/").to(device=device, dtype=torch.float32)
    wav2vec.requires_grad_(False)
    
    BASE_DIR = f'{MODEL_OUTPUT_PATH}/ckpts/det_align/'
    det_path = os.path.join(BASE_DIR, 'detface.pt')    
    align_instance = AlignImage("cuda", det_path=det_path)
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(f"{MODEL_OUTPUT_PATH}/ckpts/whisper-tiny/")

    kwargs = {
            "text_encoder": hunyuan_video_sampler.text_encoder, 
            "text_encoder_2": hunyuan_video_sampler.text_encoder_2, 
            "feature_extractor": feature_extractor, 
        }
    video_dataset = VideoAudioTextLoaderVal(
            image_size=args.image_size,
            meta_file=args.input, 
            **kwargs,
        )

    sampler = DistributedSampler(video_dataset, num_replicas=1, rank=0, shuffle=False, drop_last=False)
    json_loader = DataLoader(video_dataset, batch_size=1, shuffle=False, sampler=sampler, drop_last=False)

    for batch_index, batch in enumerate(json_loader, start=1):

        fps = batch["fps"]
        videoid = batch['videoid'][0]
        audio_path = str(batch["audio_path"][0])
        save_path = args.save_path 
        output_path = f"{save_path}/{videoid}.mp4"
        output_audio_path = f"{save_path}/{videoid}_audio.mp4"

        samples = hunyuan_video_sampler.predict(args, batch, wav2vec, feature_extractor, align_instance)
        
        sample = samples['samples'][0].unsqueeze(0)                    # denoised latent, (bs, 16, t//4, h//8, w//8)
        sample = sample[:, :, :batch["audio_len"][0]]
        
        video = rearrange(sample[0], "c f h w -> f h w c")
        video = (video * 255.).data.cpu().numpy().astype(np.uint8)  # （f h w c)
        
        torch.cuda.empty_cache()

        final_frames = []
        for frame in video:
            final_frames.append(frame)
        final_frames = np.stack(final_frames, axis=0)
        
        if rank == 0:
            imageio.mimsave(output_path, final_frames, fps=fps.item())
            os.system(f"ffmpeg -i '{output_path}' -i '{audio_path}' -shortest '{output_audio_path}' -y -loglevel quiet; rm '{output_path}'")



    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
