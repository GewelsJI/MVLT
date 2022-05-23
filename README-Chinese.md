# M-ViLT: Masked Vision-Language Transformer in Fashion

- Authors: Dehong Gao, Daniel-Ji
- Department: Alibaba ICBU
- Version Control:
    - [March 13, 2022] 更新README，对模型训练中的一些注释，清理一些不必要的代码段
    - [February 25, 2022] 更新README的文件树说明、若干小项等
    - [February 23-24, 2022] 上传项目文件，清理与注释无关代码；验证本地DWS和远程PAI训练功能；整理数据预处理的工具

# 文件树结构

为了方便了解代码，首先对重要核心文件进行注释说明

    .
    ├── bak  # 备份文件夹（暂无用处）
    │   ├── scripts_ft_dws
    │   └── scripts_ft_pai
    ├── checkpoints     # 在local-dws或remote-pai训练后，将checkpoint.pth文件会存放在这里
    │   ├── dws_pvlt_exp1
    │   └── pai_pvlt_exp21
    ├── datasets.py     # 用于调用FashionGen数据集的dataloader（见Liness-101）
    ├── downstream_recognition.sh
    ├── downstream_retrieval.sh
    ├── engine_grid_masking.py  # 训练引擎、测试引擎
    ├── evaluation.sh   # 一键测试脚本
    ├── hubconf.py
    ├── libs (##)
    │   ├── backup  # 备份文件夹（暂无用处）
    │   ├── __init__.py
    │   ├── pvlt.py     ## 模型主文件
    │   ├── utils.py    # 定义了训练过程中的一些小工具
    │   ├── vl_heads.py ## 定义了一些下游任务分类头
    │   ├── vl_scores.py    # 定义了一些指标计算方式
    │   └── vl_tools.py
    ├── logging
    │   ├── MakeDataset-full_train_info.txt
    │   └── MakeDataset-full_valid_info.log
    ├── losses.py
    ├── main_vl.py      # 训练主文件，配合调用engine.py里面的函数进行训练、测试等功能
    ├── mcloader (##)
    │   ├── backup
    │   ├── classification.py
    │   ├── data_prefetcher.py
    │   ├── fashion_gen.py  ## dataloader的定义（针对Fashion-Gen数据集）
    │   ├── image_list.py
    │   ├── imagenet.py
    │   ├── __init__.py
    │   └── mcloader.py
    ├── mytools (##)
    │   ├── extract_text_info-train.py ## 组织测试过程所使用的文本（分离）
    │   ├── extract_text_info-val.py ## 组织训练过程所使用的文本（仅用于对FashionGen数据集中的info.txt文件进行分离）
    │   ├── generate_class_templete.py  # recognition任务中的index对应（用于方便识别任务的可视化）
    │   ├── generate_MaskingData.py ## 用于生成测试过程所使用的Mask模板（当然也可以随机化生成，这里仅为保持测试的公正性）
    │   ├── generate_retrieval_data.py  ## 用于生成1 positive和100 negative的ITR和TIR检索样本
    │   └── others  # 存放了一些小的工具（暂无用处）
    ├── preweights
    │   ├── bert-base-uncased-vocab.txt # BERT的vocab.txt（查词用的词汇表）
    │   ├── cls_templete_train.txt  # recognition的子/父类别（训练集合）
    │   ├── cls_templete_valid.txt  # recognition的子/父类别（测试集合）
    │   └── pvt_v1  # PVT-Tiny的ImageNet预训练模型
    ├── README.md
    ├── requirements.txt    ## 环境配置文件
    ├── samplers.py # FAIR提出的MultiGrain策略（https://arxiv.org/abs/1902.05509），用于对mini-batch中行同一个样本进行重复抽样
    ├── scripts_pt_dws  # 一键训练脚本（local-DWS）
    │   ├── configs
    │   └── dws_pvlt_exp1.sh
    ├── scripts_pt_pai  # 一键训练脚本（remote-PAI）
    │   ├── configs # PAI训练所需要的模型配置文件存放位置
    │   ├── odps_config.ini # PAI配置文件
    │   └── pai_pvlt_exp21.sh   # PAI训练脚本


数据集存放说明（文件在: `oss://internshipalgo/jigepeng.jigepeng/PVLT-Data/Fashion-Gen/`中）

    oss://internshipalgo/jigepeng.jigepeng/PVLT-Data/
    ├── Fashion-Gen   #训练集合和测试集合存放位置
    │   ├── extracted_train_images  #训练数据集（图像）
    │   ├── extracted_valid_images  #测试数据集（图像）
    │   ├── full_train_info_PAI     #训练数据集（文本）
    │   ├── full_valid_info_PAI     #测试数据集（文本）
    │   ├── generated_valid_masking0.50_size16_images   #测试中的Random Mask（也可以实时生成）

# 项目使用说明

## 1. 环境配置

本代码目前仅在 `pytorch:1.8PAI-gpu-py36-cu101-ubuntu18.04`环境上进行了测试，不保证其他版本能够成功运行。除了自带的基础torch环境外，运行代码前，还需要安装其他的一些支持库:  `pip install -r requirements.txt`

## 2. 数据预处理

本项目在公开数据集Fashion-Gen上进行相关的验证实验，该数据集中包含260,480个训练文本图像对和35,528个测试文本图像对，M-ViLT模型可以直接处理原始图像和文本，无需任何数据的特征工程前处理。但需要对数据的存放形式进行整理，方便torch.dataloader进行文件的读取: 

- 将文本描述文件（包括`full_train_info.txt`和`full_val_info.txt`）解析为单独的pkl文件保存下来，与图像文件逐一组成文本图像对。运行脚本文件: `python ./mytools/extract_text_info-train.py`和`python ./mytools/extract_text_info-val.py`
- 生成测试中图像端所使用的Random Mask文件，以保证每次
- 生成下游检索任务（即: ITR和TIR）所需要的数据（按照1个正样本，100个负样本的比例进行样本抽取，具体规则可以参考Kaleido-BERT原文。运行脚本文件: `python ./mytools/generate_retrieval_data.py`

## 3. PAI 集群运行（用于 pre-training）

- pre-training
    - 进入对应目录: `cd ./scripts_pt_pai/`
    - 配置文件放置在: `./scripts_pt_pai/configs/pai_pvlt_exp21.py`
    - 开启训练程序: `bash pai_pvlt_exp21.sh.sh`即可
    - 训练完成后，checkpoints文件会保存在`oss://internshipalgo/jigepeng.jigepeng/PVLT-Data/pai_checkpoints/pai_pvlt_exp21/`中，对应的txt文件也可以在Dataworks LogView Portal中找到

## 4. DWS 本地运行（用于 debug & evaluation）

- local debugging

    - 进入对应目录: `cd ./scripts_pt_dws/`
    - 配置文件放置在: `scripts_pt_dws/configs/dws_pvlt_exp1.py`
    - 开启训练程序: `bash dws_pvlt_exp1.sh`即可

- 一键化inference和evaluation

    - 完成在remote-PAI上的训练后，相关文件会被存放于`oss://internshipalgo/jigepeng.jigepeng/PVLT-Data/pai_checkpoints/pai_pvlt_exp21/`，赋予下载权限后，放置于`checkpoints/pai_pvlt_exp21/checkpoint.pth`中
    - 当获取到上述`checkpoint.pth`文件后，则需转到本地DWS开发机器上进行inference和evaluation等后续操作
    - 为确保本地路径匹配，需要将配置文件`scripts_pt_pai/configs/pai_pvlt_exp21.py`中的对应的远程oss_bucket路径改为local_dev路径，即: 注释第3行、第7行，然后解注释第4行、第8行
    - 在`./`目录下，开启evaluation脚本程序: `bash evaluation.sh`
        - 得到MLM和ITM预训练任务的测评结果: `>>> accuracy of the network on the 32528 test image-text pairs: mlm_acc=0.89805% itm_acc=0.97430%`

- downstream retrieval tasks -> Image-Text Retrieval (ITR) & Text-Image Retrieval (TIR）

    - 注意: 本文衡量的是zero-shot retrieval任务性能，即无需fine-tune过程，可以直接拿pre-training的模型用来推理下游检索任务
    - 完成上述第二大步骤后（确保从远程oss_bucket路径改为local_dev路径）
    - 在`./`目录下，运行脚本程序: `downstream_retrieval.sh`，用于测试downstream的检索指标（TIR和ITR）
    - 得到ITR下游任务的测评结果: `>>> retrieval ITR: acc@1: 0.331, acc@5: 0.772, acc@10: 0.911`
    - 得到TIR下游任务的测评结果: `>>> retrieval TIR: acc@1: 0.346, acc@5: 0.78, acc@10: 0.895`

- downstream recognition tasks -> Main-Category Recognition (M-CR) & Sub-Category Recognition (S-CR)

    - 注意: 本文衡量的是Few-shot recognition任务的性能，即需要fine-tune过程，才能用于推理下游识别任务。但是，由于端到端框架的设计，这里的fine-tune过程是可以并入预训练过程之中的，可以做成多任务的框架（只需要调整损失函数的比重，然后调参就可以逼近最优点）
    - 完成上述第二大步骤后（确保从远程oss_bucket路径改为local_dev路径）