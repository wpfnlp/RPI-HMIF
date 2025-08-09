# Source Code of RPI-HMIF Model for Multimodal Named Entity Recognition
Implementation of Our Paper "**How People Read? Reading Preference-Inspired Multimodal NER with Heterogeneous Mining and Iterative Fusion Engine**" in Information Fusion 2025. This Implementation is based on the [DeepKE](https://github.com/zjunlp/DeepKE).

## Model Architecture

<img width="1116" height="658" alt="image" src="https://github.com/user-attachments/assets/c6eb7304-ac5d-4ba2-bc00-faf3b3dee6a1" />

The Overall Framework of Our Proposed **RPI-HMIF** Model. We propose <ins>Reading Preference-Inspired Multimodal NER with Heterogeneous Mining and Iterative Fusion Engine (RPI-HMIF)</ins>. Inspired by people reading preference of reading with questions, we introduce an **explicit-question based Heterogeneous Information Mining Module (HIM)** to guide the model to accurately mine the task-guided text core semantics and task-guided images core features to mitigate textual and visual bias. In addition, we build an **Iterative Feature Fusion Engine (IFFE)** with improved dynamic routing as the core algorithm. It provides a deeper level interaction, continuously optimizes and deepens the interaction between modalities through an iterative process to adjust weights and update features. We conduct experiments on the Twitter-2015 and Twitter-2017 datasets, and the results show that our approach outperforms the SOTA model, achieving F1-scores of 76.77\% and 88.35\%, respectively.

## Experiment
The overall experimental results on RPI-HMIF for Multi-Modal NER task can be seen as follows:
<img width="881" height="673" alt="image" src="https://github.com/user-attachments/assets/44b5b005-2129-405f-8fec-26e5e5bd3cd3" />


## Installation
Clone the newest repository:

```bash
git clone https://github.com/wpfnlp/RPI-HMIF
cd RPI-HMIF/example/ner/multimodal
```

Install with Pip

```bash
pip install -r requirements.txt
python setup.py install
```

## Usage
#### Dataset - Twitter2015 & Twitter2017
- Download the dataset to this directory.
    
  The text data follows the conll format.
  The acquisition of Twitter15 and Twitter17 data refer to the code from [UMT](https://github.com/jefferyYu/UMT/), many thanks.
    
  You can download the Twitter2015 and Twitter2017 dataset with detected visual objects using folloing command:
    
  ```bash
  wget 120.27.214.45/Data/ner/multimodal/data.tar.gz
  tar -xzvf data.tar.gz
  ```
- The twitter15 dataset with detected visual objects is stored in `data`:
    
  - `twitter15_detect`：Detected objects using RCNN
  - `twitter2015_aux_images`：Detected objects using visual grouding
  - `twitter2015_images`： Original images
  - `train.txt`: Train set
  - `...`
    
#### Training
- Parameters, model paths and configuration for training are in the `conf` folder and users can modify them before training.
- Download the [PLM](https://huggingface.co/openai/clip-vit-base-patch32/tree/main) and set `vit_name` in `train.yaml` and `predict.yaml` as the directory of the PLM.
- Run
    
  ```bash
  python run.py
  ```
- The trained model is stored in the `checkpoint` directory by default and you can change it by modifying "save_path" in `train.yaml`.
- Start to train from last-trained model
    
  modify `load_path` in `train.yaml` as the path of the last-trained model
- Logs for training are stored in the current directory by default and the path can be configured by modifying `log_dir` in `.yaml`
  
#### Prediction
Modify "load_path" in `predict.yaml` to the trained model path. 
<!-- **In addition, we provide [the model trained on Twitter2017 dataset](https://drive.google.com/drive/folders/1ZGbX9IiNU3cLZtt4U8oc45zt0BHyElAQ?usp=sharing) for users to predict directly.** -->
  
```bash
python predict.py

```
## Cite
If you use or extend our work, please cite the following paper:
```bibtex
```
