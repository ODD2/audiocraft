import os
import cv2
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


model = MusicGenCLIP.get_pretrained('checkpoints/clipemb(v4)')

frame_transform = Compose([
    Resize((380, 380), interpolation=InterpolationMode.BICUBIC, antialias=True),
])

model.set_generation_params(duration=15)  # generate 8 seconds.

video_files = [
    "dataset/muvi/train/065.mp4",
    "dataset/muvi/train/774.mp4",
    "dataset/muvi/train/070.mp4",
    "dataset/muvi/train/746.mp4",
    "dataset/muvi/test/390.mp4"
]

clips = []
for file in video_files:
    frames = fetch_frames(video_path=file, duration=30, offset=90, num_frames=10, transform=frame_transform)
    clips.append(frames)

wav = model.generate_with_clip_embed(clips)  # generates 3 samples.

folder_name = f"samples/{datetime.datetime.now().strftime('%m%d_%H%M%S')}"

for idx, one_wav, one_frames in zip(range(len(wav)), wav, clips):
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
