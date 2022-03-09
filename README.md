# 🗣 SpeakUnique: Efficient speaker personalization
```shell
# TODO: Fix the Arxiv button
```
[![arXiv](https://img.shields.io/badge/arXiv-year.todo-todo.svg)](todo)

> **Abstract**
>
> The abstract remains to be written

📢 [Demo website](https://polvanrijn.github.io/speakunique/)


## Preprocessing
### Create face styles
````shell
git clone https://github.com/polvanrijn/SpeakUnique.git
cd SpeakUnique
main_dir=$PWD
````

The first step is to set up the [Cartoon-StyleGan2](https://github.com/happy-jihye/Cartoon-StyleGAN) repository.

```shell
# Setup StyleGAN repo
preprocessing_env="$main_dir/preprocessing"
conda create --prefix $preprocessing_env python=3.7
conda activate $preprocessing_env

git clone https://github.com/happy-jihye/Cartoon-StyleGan2.git cartoon_stylegan2
cd cartoon_stylegan2
pip install ftfy regex tqdm ninja gdown
pip install git+https://github.com/openai/CLIP.git
pip install IPython
pip install matplotlib
pip install scikit-image==0.16.1

# Install SpeakerNet embeddings
pip install Cython
pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[all]

# Install checkpoints
python -c "exec(\"from utils import download_pretrained_model\ndownload_pretrained_model()\")"
```

The next step is to project reference images into the StyleGAN space. We use the same sentence ("Kids are talking by the
door", neutral recording) from the RAVDESS corpus from all 24 speakers. You can download all videos by running
`sh videos/download_RAVDESS.sh`. However, the stills used in this paper are also part of the repository (`stills`). You
can project the stills into the StyleGAN space and obtain the different styles by running:

```shell
sh create_latent_for_all_speakers.sh
```

### Obtain the PCA space
The model used in the paper was trained on SpeakerNet embeddings, so we to extract the embeddings from a dataset. Here
we use the commonvoice data. To download it, run:
````shell
python3 preprocess_commonvoice.py --language en
````

To extract the principal components, run `compute_pca.py`.

## Synthesis
### Setup
We'll assume, you'll setup a remote instance for synthesis.  The first step is to make sure you cloned our repository:
````shell
git clone https://github.com/polvanrijn/SpeakUnique.git
cd SpeakUnique
main_dir=$PWD
````

Now, we'll set up a remote synthesis API and install the virtual environment.
```shell
synthesis_env="$main_dir/synthesis"
conda create --prefix $synthesis_env python=3.7
conda activate $synthesis_env

##############
# Setup Wav2Lip
##############
git clone https://github.com/Rudrabha/Wav2Lip.git
cd Wav2Lip

# Install Requirements
pip install -r requirements.txt
pip install opencv-python-headless==4.1.2.30
wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "face_detection/detection/sfd/s3fd.pth"  --no-check-certificate

# Install as package
mv ../setup_wav2lip.py setup.py
pip install -e .
cd ..


##############
# Setup VITS
##############
git clone https://github.com/jaywalnut310/vits
cd vits

# Install Requirements
pip install -r requirements.txt

# Install monotonic_align
cd monotonic_align/
python setup.py build_ext --inplace
cd ..

# Download the checkpoint
pip install gdown
gdown 'https://drive.google.com/uc?id=1q86w74Ygw2hNzYP9cWkeClGT5X25PvBT'

# Install as package
mv ../setup_vits.py setup.py
pip install -e .

pip install flask
pip install wget

```

You'll need to do the last step manually (let me know if you know an automatic way). Download the
checkpoint `wav2lip_gan.pth` from [here](https://github.com/Rudrabha/Wav2Lip) and put it in `Wav2Lip/checkpoints`. Make 
sure you have `espeak` installed and it is in your path.

### Running
Start the remote service (I used port 31337)
```shell
python server.py --port 31337
```

You can send an example request locally, by running (don't forget to change host and port accordingly):
```shell
python request_demo.py
```

We also made a small 'playground' so you can see how slider values will influence the voice. First setup a local HTTP 
server.
```shell
 python3 -m http.server
```

Now open the following url: http://localhost:8000/client.html
