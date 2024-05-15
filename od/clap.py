import torchaudio
from audiocraft.models.musicgen import MusicGen, MusicGenCLAP
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio

model = MusicGenCLAP.get_pretrained('checkpoints/clapemb(v2_full)')
model.set_generation_params(duration=8)  # generate 8 seconds.
wav_files = [
    "./samples/v2/references/1.mp3",
    "./samples/v2/references/2.mp3",
    "./samples/v2/references/3.mp3",
    "./samples/v2/references/4.mp3",
    "./samples/v2/references/5.mp3"
]

wavs = []
for file in wav_files:
    wav, sr = torchaudio.load(file)
    wav = convert_audio(wav, sr, model.sample_rate, model.audio_channels)
    wavs.append(wav)

wav = model.generate_with_clap_embed(wavs)  # generates 3 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx+1}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
