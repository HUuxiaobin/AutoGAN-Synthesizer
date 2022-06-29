# AutoGAN-Synthesizer

This is the official code for the paper **AutoGAN-Synthesizer: Neural Architecture Searchfor Cross-Modality MRI Synthesis**, which is accepted by MICCAI 2022.

## Usage

1. Search model structure:

```
python main_train_search.py
```

2. Use the found model structure to train:

```
python main_train_exploring_time_model_brats_gan_v2_perception.py --ckt_path path

```
