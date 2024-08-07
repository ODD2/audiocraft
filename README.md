# AudioCraft Varient For Image to Music Generation
We extend the MusicGen project, a text-to-music generation framework, for extra modality controlling like music and video.
In this page, we'll provide several helpful instructions/tools to speed up the preparation to work under this framework, if further instructions are required (e.g. installation and data preparation) please refer to the original documents located in the [docs](./docs/) folder, starting with this [page](./docs/AUDIOCRAFT.md).

## Major Configuration Files
```yaml
config/config.yaml: gpu and vram configuration (?).
config/teams/default.yaml: global folder localization.
config/solver/musicgen/default.yaml: setup for evalution metric environments, and dataset sample size.
config/solver/musicgen/musicgen_base_32khz.yaml: major training setup file, configures the batch size and workers of the dataloader,the generation phase interval, the evaluation phase interval and metrics to use, the optimizer, the logging period, the checkpoint setup...
config/conditioner/clipemb2music.yaml: musicgen conditioner configuration, also controls the cross attention positional encoding and the classifier free guidance.
config/dset/audio/ytcharts.yaml: music dataset configuration.
```
## Major Source Files
```yaml
audiocraft/data/music_dataset.py: modify code for the video music dataloader 
/models/builders.py: modify code to use specific conditioner.
/models/musicgen.py: configure inference interface for the new video/music conditioner.
/modules/conditioners.py: create new conditioners here.
/modules/transformers.py: transformer modules here, currently remains unaltered.
/solvers/musicgen.py:  training specific model layers after initialization.
/utils/samples/manager.py: control the namings of the generated samples here.   
```
## FAD Evaluation
The FAD evaluation is processed by google's FAD project under a **seperated environment**, which requires extra setup to make this function work properly.

### Environment Installation 
Please refer and ***run only the dependency installation step (step 2)*** of the [guidelines](https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/metrics/fad.py), the first step to modify the source code has been completed and included in this project.

```bash
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
pip install apache-beam numpy scipy tf_slim
```

Then, [download the model checkpoint]() and [run the test](https://github.com/google-research/google-research/tree/master/frechet_audio_distance#create-test-files-and-file-lists) from the original repo to validate FAD's functionality, would need to specify the *--model_ckpt* option to the local vgg checkpoint. 

### Environment Configuration for Evaluation 
configure the following environment variables in the **audiocraft environment** and re-activate the environment to enable evaluation during training (remember to move vgg checkpoint to the default location 'log/fad/' folder).
```bash
# refer to: conda env config vars set XXX=XXX
conda env config vars set TF_PYTHON_EXE=/home/od/miniconda3/envs/fad/bin/python `# use 'which python' in the fad's environment to determine`
conda env config vars set TF_LIBRARY_PATH=/home/od/miniconda3/envs/fad/lib/python3.9/site-packages/nvidia/cudnn/lib `# similarly locate the nvidia cudnn library after located in the conda environment.`
conda env config vars set TF_FORCE_GPU_ALLOW_GROWTH=true
```

## Main Workflow
Basically, we repeat the following pipeline to train and evaluate the generated samples.
```bash
# run 
dora run -d \
solver=musicgen/musicgen_base_32khz `# base config file`\
model/lm/model_scale=small `# the scale of musicgen model, in [small,medium,large]`\
continue_from=/scratch4/users/od/repos/audiocraft/small_syno_L14_30F.pt `# the checkpoint modified for custom conditioner`\
conditioner=clipemb2music `# the conditioner config file`\
dset=audio/ytcharts  `# the dataset to train & evaluate on`

# Now, after training, the model checkpoints should locate at 'logs/xps/2e5708a4/', we need to transform the checkpoint into an dedicated model weight for inference. Configure and run the od/export.py script to export the model. The exported weights will be saved at './checkpoints'. 
python -m od.export

# Setup the generation script (e.g. od/clip.py) to create videos with generated music.
pyhton -m od.clip `# The samples are located at 'samples/0601_120000/*.mp4'`
```

## Several Helpful Instructions
```bash
# If the conditioner has different feature size than the default musicgen model, we need to re-create a checkpoint for the new feature size. Use od/make_ckpt.py to do the work.
python -m od.make_ckpt `# This will create a checkpoint in the root folder`
```
```bash
# The structure of the dataset folder is:
./dataset - ytcharts/ - {test/, train/} - {ytid.mp3, ytid.mp4, ytid.json}
```
```bash
# To create the egs for dataset config file.
python -m audiocraft.data.audio_dataset dataset/ytcharts/train egs/ytcharts/train.jsonl
python -m audiocraft.data.audio_dataset dataset/ytcharts/test egs/ytcharts/test.jsonl
```
```bash
# Run tensorboard to visualize the training process
tensorboard --logdir=logs/xps/2e5708a4/tensorboard
```
```bash
# Locate the generated samples in the samples folder
cd logs/xps/2e5708a4/samples
```
```bash
# Download VGGISH checkpoint to logs/fad/
curl -o data/vggish_model.ckpt https://storage.googleapis.com/audioset/vggish_model.ckpt
```
```bash
# Download CLAP checkpoint to logs/clap/
curl -o https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt
```