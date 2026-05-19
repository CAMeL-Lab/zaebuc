# Zaebuc
This repo contains code and pretrained models to reproduce the results in our paper [A Bilingual Bimodal Benchmark for Arabic-English NLP Across Grammatical Correction, Essay Scoring, Morphological Tagging, and Speech Recognition](http://www.lrec-conf.org/proceedings/lrec2026/pdf/2026.lrec2026-1.137.pdf).

# Requirements:
The code was written for python>=3.10. You will need a few additional packages. Here's how you can set up the environment using conda (assuming you have conda and cuda installed):

```bash
git clone https://github.com/CAMeL-Lab/zaebuc.git
cd zaebuc

conda create -n zaebuc python=3.10
conda activate zaebuc

pip install -r requirements.txt
```
Note that to run the Arabic and English grammatical error correction (GEC) experiments using the models developed by [Alhafni et al., 2023](https://github.com/CAMeL-Lab/arabic-gec/) and [Omelianchuk et al., 2020](https://github.com/grammarly/gector), you'd need to follow their instructions on installing the necessary packages.

# Experiments and Reproducibility:
For full reproducibility, all model outputs and pretrained models used in this work are available [here](https://drive.google.com/drive/folders/1gKfe8shgoE2o48vV_JeE1kSt5pIU94WZ?usp=sharing). This repo is organized as follows:
1. [aes](aes/): includes all code and scripts used to train and evaluate automated essay scoring (AES) models in the paper.
2. [gec](gec/): includes all code and scripts used to evaluate the GEC models we report in the paper.
3. [morph](morph/): includes all code and scripts used to evaluate the morphosyntactic models we report in the paper.
4. [asr](asr/): includes the scripts used to evaluate the whisper model we report in the paper.

# License:
This repo is available under the MIT license. See the [LICENSE](LICENSE) for more info.

# Citation:
```bibtex
@inproceedings{alhafni-etal-2026-bilingual,
  title = {A Bilingual Bimodal Benchmark for Arabic-English NLP across Grammatical Correction, Essay Scoring, Morphological Tagging, and Speech Recognition},
  author = {Alhafni, Bashar and Hamed, Injy and Eryani, Fadhl and Palfreyman, David and Habash, Nizar},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {1732--1749},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  editor = {Piperidis, Stelios and Bel, Núria and van den Heuvel, Henk and Ide, Nancy and Krek, Simon and Toral, Antonio},
  doi = {10.63317/489vftd6umyh},
  abstract = {Building comprehensive datasets that support a variety of NLP tasks and cover a diversity of languages and domains is vital for NLP evaluation purposes. In this paper, we present ZAEBUC*, a dataset that builds upon and enriches prior corpora with new annotations and benchmarking experiments. ZAEBUC* serves as a benchmark for a range of NLP tasks, including grammatical error correction, automated essay scoring, automatic speech recognition, and morphological tagging, which includes tokenization, part-of-speech tagging, and lemmatization. The dataset covers Arabic and English in both written and spoken forms, offering a bilingual and bimodal resource. Furthermore, the corpus brings together a collection of resources gathered from a similar population, enabling cross-linguistic and cross-modal comparisons. We provide benchmarking results, demonstrating the performance of NLP models, including LLMs, across various tasks, languages, and modalities.}
}

