import os
import io
import math
import uuid
import base64
import imageio
import torch
import torchvision
from PIL import Image
import numpy as np
from copy import deepcopy
from einops import rearrange
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from hymm_sp.data_kits.audio_dataset import get_audio_feature

TEMP_DIR = "./temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR, exist_ok=True)


def data_preprocess_server(args, image_path, audio_path, prompts, feature_extractor):
    llava_transform = transforms.Compose(
            [
                transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BILINEAR), 
                transforms.ToTensor(), 
                transforms.Normalize((0.48145466, 0.4578275, 0.4082107), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
    
    """ 生成prompt """
    if prompts is None:
        prompts = "Authentic, Realistic, Natural, High-quality, Lens-Fixed." 
    else:
        prompts = "Authentic, Realistic, Natural, High-quality, Lens-Fixed, " + prompts

    fps = 25
    
    img_size = args.image_size
    ref_image = Image.open(image_path).convert('RGB')
    
    # Resize reference image
    w, h = ref_image.size
    scale = img_size / min(w, h)
    new_w = round(w * scale / 64) * 64
    new_h = round(h * scale / 64) * 64

    if img_size == 704:
        img_size_long = 1216
    if new_w * new_h > img_size * img_size_long:
        scale = math.sqrt(img_size * img_size_long / w / h)
        new_w = round(w * scale / 64) * 64
        new_h = round(h * scale / 64) * 64

    ref_image = ref_image.resize((new_w, new_h), Image.LANCZOS)
    
    ref_image = np.array(ref_image)
    ref_image = torch.from_numpy(ref_image)
        
    audio_input, audio_len = get_audio_feature(feature_extractor, audio_path)
    audio_prompts = audio_input[0]
    
    motion_bucket_id_heads = np.array([25] * 4)
    motion_bucket_id_exps = np.array([30] * 4)
    motion_bucket_id_heads = torch.from_numpy(motion_bucket_id_heads)
    motion_bucket_id_exps = torch.from_numpy(motion_bucket_id_exps)
    fps = torch.from_numpy(np.array(fps))
    
    to_pil = ToPILImage()
    pixel_value_ref = rearrange(ref_image.clone().unsqueeze(0), "b h w c -> b c h w")   # (b c h w)
    
    pixel_value_ref_llava = [llava_transform(to_pil(image)) for image in pixel_value_ref]
    pixel_value_ref_llava = torch.stack(pixel_value_ref_llava, dim=0)

    batch = {
        "text_prompt": [prompts],
        "audio_path": [audio_path],
        "image_path": [image_path],
        "fps": fps.unsqueeze(0).to(dtype=torch.float16),
        "audio_prompts": audio_prompts.unsqueeze(0).to(dtype=torch.float16),
        "audio_len": [audio_len],
        "motion_bucket_id_exps": motion_bucket_id_exps.unsqueeze(0),
        "motion_bucket_id_heads": motion_bucket_id_heads.unsqueeze(0),
        "pixel_value_ref": pixel_value_ref.unsqueeze(0).to(dtype=torch.float16),
        "pixel_value_ref_llava": pixel_value_ref_llava.unsqueeze(0).to(dtype=torch.float16)
    }

    return batch

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8, quality=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x,0,1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps, quality=quality)

def encode_image_to_base64(image_path):
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
        encoded_data = base64.b64encode(image_data).decode('utf-8')
        print(f"Image file '{image_path}' has been successfully encoded to Base64.")
        return encoded_data
    
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def encode_video_to_base64(video_path):
    try:
        with open(video_path, 'rb') as video_file:
            video_data = video_file.read()
        encoded_data = base64.b64encode(video_data).decode('utf-8')
        print(f"Video file '{video_path}' has been successfully encoded to Base64.")
        return encoded_data
    
    except Exception as e:
        print(f"Error encoding video: {e}")
        return None
    
def encode_wav_to_base64(wav_path):
    try:
        with open(wav_path, 'rb') as audio_file:
            audio_data = audio_file.read()
        encoded_data = base64.b64encode(audio_data).decode('utf-8')
        print(f"Audio file '{wav_path}' has been successfully encoded to Base64.")
        return encoded_data
    
    except Exception as e:
        print(f"Error encoding audio: {e}")
        return None
    
def encode_pkl_to_base64(pkl_path):
    try:
        with open(pkl_path, 'rb') as pkl_file:
            pkl_data = pkl_file.read()
        
        encoded_data = base64.b64encode(pkl_data).decode('utf-8')
        
        print(f"Pickle file '{pkl_path}' has been successfully encoded to Base64.")
        return encoded_data

    except Exception as e:
        print(f"Error encoding pickle: {e}")
        return None
      
def decode_base64_to_image(base64_buffer_str):
    try:
        image_data = base64.b64decode(base64_buffer_str)
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        print(f"Image Base64 string has beed succesfully decoded to image.")
        return image_array
    except Exception as e:
        print(f"Error encdecodingoding image: {e}")
        return None
    
def decode_base64_to_video(base64_buffer_str):
    try:
        video_data = base64.b64decode(base64_buffer_str)
        video_bytes = io.BytesIO(video_data)
        video_bytes.seek(0)
        video_reader = imageio.get_reader(video_bytes, 'ffmpeg')
        video_frames = [frame for frame in video_reader]
        return video_frames
    except Exception as e:
        print(f"Error decoding video: {e}")
        return None

    
def save_video_base64_to_local(video_path=None, base64_buffer=None, output_video_path=None):
    if video_path is not None and base64_buffer is None:
        video_buffer_base64 = encode_video_to_base64(video_path)
    elif video_path is None and base64_buffer is not None:
        video_buffer_base64 = deepcopy(base64_buffer)
    else:
        print("Please pass either 'video_path' or 'base64_buffer'")
        return None
    
    if video_buffer_base64 is not None:
        video_data = base64.b64decode(video_buffer_base64)
        if output_video_path is None:
            uuid_string = str(uuid.uuid4())
            temp_video_path = f'{TEMP_DIR}/{uuid_string}.mp4'
        else:
            temp_video_path = output_video_path
        with open(temp_video_path, 'wb') as video_file:
            video_file.write(video_data)
        return temp_video_path
    else:
        return None
    
def save_audio_base64_to_local(audio_path=None, base64_buffer=None):
    if audio_path is not None and base64_buffer is None:
        audio_buffer_base64 = encode_wav_to_base64(audio_path)
    elif audio_path is None and base64_buffer is not None:
        audio_buffer_base64 = deepcopy(base64_buffer)
    else:
        print("Please pass either 'audio_path' or 'base64_buffer'")
        return None
    
    if audio_buffer_base64 is not None:
        audio_data = base64.b64decode(audio_buffer_base64)
        uuid_string = str(uuid.uuid4())
        temp_audio_path = f'{TEMP_DIR}/{uuid_string}.wav'
        with open(temp_audio_path, 'wb') as audio_file:
            audio_file.write(audio_data)
        return temp_audio_path
    else:
        return None
    
def save_pkl_base64_to_local(pkl_path=None, base64_buffer=None):
    if pkl_path is not None and base64_buffer is None:
        pkl_buffer_base64 = encode_pkl_to_base64(pkl_path)
    elif pkl_path is None and base64_buffer is not None:
        pkl_buffer_base64 = deepcopy(base64_buffer)
    else:
        print("Please pass either 'pkl_path' or 'base64_buffer'")
        return None
    
    if pkl_buffer_base64 is not None:
        pkl_data = base64.b64decode(pkl_buffer_base64)
        uuid_string = str(uuid.uuid4())
        temp_pkl_path = f'{TEMP_DIR}/{uuid_string}.pkl'
        with open(temp_pkl_path, 'wb') as pkl_file:
            pkl_file.write(pkl_data)
        return temp_pkl_path
    else:
        return None
    
def remove_temp_fles(input_dict):
    for key, val in input_dict.items():
        if "_path" in key and val is not None and os.path.exists(val):
            os.remove(val)
            print(f"Remove temporary {key} from {val}")

def process_output_dict(output_dict):

    uuid_string = str(uuid.uuid4())
    temp_video_path = f'{TEMP_DIR}/{uuid_string}.mp4'
    imageio.mimsave(temp_video_path, output_dict["video"], fps=output_dict.get("save_fps", 25))

    # Add audio
    if output_dict["audio"] is not None and os.path.exists(output_dict["audio"]):
        output_path = temp_video_path
        audio_path = output_dict["audio"]
        save_path = temp_video_path.replace(".mp4", "_audio.mp4")
        print('='*100)
        print(f"output_path = {output_path}\n audio_path = {audio_path}\n save_path = {save_path}")
        os.system(f"ffmpeg -i '{output_path}' -i '{audio_path}' -shortest '{save_path}' -y -loglevel quiet; rm '{output_path}'")
    else:
        save_path = temp_video_path

    video_base64_buffer = encode_video_to_base64(save_path)

    encoded_output_dict = {
        "errCode": output_dict["err_code"], 
        "content": [
                    {
                        "buffer": video_base64_buffer
                    },
                ],
        "info":output_dict["err_msg"],
    }
    
    

    return encoded_output_dict


def save_image_base64_to_local(image_path=None, base64_buffer=None):
    # Encode image to base64 buffer
    if image_path is not None and base64_buffer is None:
        image_buffer_base64 = encode_image_to_base64(image_path)
    elif image_path is None and base64_buffer is not None:
        image_buffer_base64 = deepcopy(base64_buffer)
    else:
        print("Please pass either 'image_path' or 'base64_buffer'")
        return None
    
    # Decode base64 buffer and save to local disk
    if image_buffer_base64 is not None:
        image_data = base64.b64decode(image_buffer_base64)
        uuid_string = str(uuid.uuid4())
        temp_image_path = f'{TEMP_DIR}/{uuid_string}.png'
        with open(temp_image_path, 'wb') as image_file:
            image_file.write(image_data)
        return temp_image_path
    else:
        return None
    
def process_input_dict(input_dict):
    
    decoded_input_dict = {}
   
    decoded_input_dict["save_fps"] = input_dict.get("save_fps", 25)

    image_base64_buffer = input_dict.get("image_buffer", None)
    if image_base64_buffer is not None:
        decoded_input_dict["image_path"] = save_image_base64_to_local(
            image_path=None, 
            base64_buffer=image_base64_buffer)
    else:
        decoded_input_dict["image_path"] = None
    
    audio_base64_buffer = input_dict.get("audio_buffer", None)
    if audio_base64_buffer is not None:
        decoded_input_dict["audio_path"] = save_audio_base64_to_local(
            audio_path=None, 
            base64_buffer=audio_base64_buffer)
    else:
        decoded_input_dict["audio_path"] = None
    
    decoded_input_dict["prompt"] = input_dict.get("text", None)
        
    return decoded_input_dict