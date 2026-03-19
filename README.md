# Zaebuc
This repo contains code and pretrained models to reproduce the results in our paper [A Bilingual Bimodal Benchmark for Arabic-English NLP Across Grammatical Correction, Essay Scoring, Morphological Tagging, and Speech Recognition]().

# Requirements:
The code was written for python>=3.10. You will need a few additional packages. Here's how you can set up the environment using conda (assuming you have conda and cuda installed):

```bash
git clone https://github.com/CAMeL-Lab/zaebuc.git
cd zaebuc

conda create -n zaebuc python=3.10
conda activate zaebuc

pip install -r requirements.txt
```
Note that to run the Arabic and English grammatical error correction (GEC) experiments using the models developed by [Alhafni et al., ]() and [Omelianchuk et al., ](), you'd need to follow their instructions on installing the necessary packages.

# Experiments and Reproducibility:
This repo is organized as follows:
1. [aes](aes/): includes all code and scripts used to train and evaluate automated essay scoring (AES) models in the paper.
2. [gec](gec/): includes all code and scripts used to evaluate the GEC models we report in the paper.
3. [morph](morph/):
4. [asr](asr/):

# License:
This repo is available under the MIT license. See the [LICENSE](LICENSE) for more info.

# Citation:
```bibtex
@inproceedings{alhafni-etal-2026-bilingual,
    title = "A Bilingual Bimodal Benchmark for Arabic-English NLP Across Grammatical Correction, Essay Scoring, Morphological Tagging, and Speech Recognition",
    author = "Alhafni, Bashar and
         Hamed, Injy and
         Eryani, Fadhl and
         Palfreyman, David and
         Habash, Nizar",
    booktitle = "Proceedings of the Sixteenth Language Resources and Evaluation Conference",
    year = "2026",
    abbr= "LREC",
    address = "Mallorca, Spain",
    publisher = "European Language Resources Association"
}


