import os
import cv2
import torch
import datetime
import torchaudio
from audiocraft.models.musicgen import MusicGen, MusicGenCLAP, MusicGenCLIP
from audiocraft.data.audio import audio_write
from audiocraft.data.music_dataset import fetch_frames
from audiocraft.data.audio_utils import convert_audio
from torchvision.transforms import Compose, Resize, InterpolationMode, CenterCrop


def frames_write(path, frames):
    os.makedirs(path, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(
            os.path.join(path, f"{i+1}.png"),
            cv2.cvtColor(frame.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
        )


model = MusicGenCLIP.get_pretrained('checkpoints/clipemb(v25)')

frame_transform = Compose([
    Resize((380, 380), interpolation=InterpolationMode.BICUBIC, antialias=True),
])

model.set_generation_params(duration=30, cfg_coef=3, top_k=250)

video_files = [
    "dataset/muvi/train/065.mp4",
    "dataset/muvi/train/774.mp4",
    "dataset/muvi/train/070.mp4",
    "dataset/muvi/test/390.mp4",
    "dataset/muvi/test/007.mp4",
    "dataset/muvi/test/028.mp4",
]

clips = []
for file in video_files:
    frames = fetch_frames(video_path=file, duration=30, offset=90, num_frames=30, transform=frame_transform)
    clips.append(frames)

_clips = []
for i in range(len(clips) - 1):
    _clips.append(torch.cat([clips[i][:15], clips[i + 1][15:]]))

clips = torch.stack(_clips)
wav = model.generate_with_clip_embed(clips)  # generates 3 samples.

folder_name = f"samples/{datetime.datetime.now().strftime('%m%d_%H%M%S')}"

for idx, one_wav, one_frames, src in zip(range(len(wav)), wav, clips, video_files):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(
        f'{folder_name}/{idx+1}',
        one_wav.cpu(),
        model.sample_rate,
        strategy="loudness",
        loudness_compressor=True
    )
    frames_write(
        f'{folder_name}/{idx+1}',
        one_frames
    )
    # origin clip
    os.system(
        f"ffmpeg  -hide_banner -loglevel error -ss 00:01:30 -t 30  -i '{src}' '{folder_name}/o_{idx+1}.mp4'"
    )
    # generate clip
    os.system(
        f"ffmpeg -hide_banner -loglevel error  -i '{folder_name}/o_{idx+1}.mp4' -i '{folder_name}/{idx+1}.wav' -c:v copy -map 0:v:0 -map 1:a:0 '{folder_name}/g_{idx+1}.mp4'"
    )
