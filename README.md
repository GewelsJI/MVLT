# Masked Vision-Language Transformer in Fashion

- Authors: Ge-Peng Ji^, Mingcheng Zhuge^, Dehong Gao, Deng-Ping Fan#, Christos Sakaridis, and Luc Van Gool
- Link: [arXiv Paper]()
- This project is still work in progress, and we invite all to contribute in making it more acessible and useful. If you have any questions, please feel free to drop us an e-mail (gepengai.ji@gmail.com & mczhuge@gmail.com & dengpfan@gmail.com) or directly report it in the issue or push a PR. Your star is our motivation, let's enjoy it!

# Dataset Preparation

This project conducts several experiments on the public dataset, Fashion-Gen, which contains 260,480 training text-image pairs for training and 35,528 text-image pairs for inference. The M-ViLT model can directly process the original image and text without any feature engineering pre-processing of the data. However, it is necessary to sort out the storage form of the data to facilitate the dataloader of torch:

Please download the reorganized dataset via runing `wget xxx` in your terminal.

# Prelimilaries

- Installing the basic libararies python3.6, pytorch1.8, cuda10.1 on UBUNTU18.04. I did validate the flexibilty on other versions of libraries and systems, but I think it is easy to adaptation with minor changes. 
- Installing the auxiliary libraries via running `pip install -r requirements.txt`

# Inference

- Downstream retrieval tasks
  - We provide the zero-shot retrieval performance without any finetuning process, and thus, the well-trained weight could be directly used in the retrieval tasks
  - Please download the pre-trained weight via runing `wget xxx` and move it into `./checkpoints/mvlt/`
  - Just run `bash downstream_retrieval.sh` and then get the prediction result like this:
    - Image-Text Retrieval (ITR): `>>> retrieval ITR: acc@1: 0.331, acc@5: 0.772, acc@10: 0.911`
    - Text-Image Retrieval (TIR): `>>> retrieval TIR: acc@1: 0.346, acc@5: 0.78, acc@10: 0.895`

- Downstream recognition tasks
  - This task needs the fine-tuning process because our pre-trained model is not equipped with the classification head.
  - Please download the pre-trained weight via runing `wget xxx` and move it into `./checkpoints/mvlt/`
  - Just run `bash downstream_recognition.sh` and then get the prediction result like this:
    - Main-Category Recognition (M-CR): `> logging-sup: accuracy (0.9825996064928677) macro_f1 (0.8954719842489123) micro_f1 (0.9825996064928677) weighted_f1 (0.9824654977888717)`
    - Sub-Category Recognition (S-CR): `> logging-sub: accuracy (0.9356554353172651) macro_f1 (0.8285927576055913) micro_f1 (0.9356554353172651) weighted_f1 (0.9351514388782373)`

# Citation

    @article{ji2022masked,
      title={Masked Vision-Language Transformer in Fashion},
      author={Ji, Ge-Peng and Zhuge, Mingchen and Gao, Dehong and Fan, Deng-Ping and Van Gool, Luc},
      journal={arXiv preprint arXiv:1906.02691},
      year={2022}
    }

# Acknowlegement

Thanks Alibaba ICBU Search Team and Wenhai Wang ([PVT](https://github.com/whai362/PVT)) for technical support.