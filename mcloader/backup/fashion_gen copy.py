import os
import random
import warnings
import numpy as np
import torch.utils.data as data
import pickle
import torch
import cv2
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from PIL import ImageEnhance
from transformers import BertTokenizer
import argparse
import copy

ONLY_FOR_DEBUG = False # @Daniel: use True for debugging


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right. (used for BART task)
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # print(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -1 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -1, pad_token_id)

    return shifted_input_ids


class FashionGenDatasetPreTrain(data.Dataset):
    """ loading fashion-gen dataset (train/val).
    @self.text_dicts: 
        'product_id', 'img_name', 'super_cls_name', 'super_cls_id', sub_cls_name', 'sub_cls_id', 
        'captions', 'input_ids', 'token_type_ids', 'attention_mask'
    """
    # def __init__(self, root, trainsize, data_type, if_itm, if_itg, mask_ratio, mask_strategy, max_token_length=128, word_mask_rate=0.15, is_train=True):
    def __init__(self, root, data_type, is_train=True, args=None):
        # pre-defination
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.trainsize = args.input_size
        self.is_train = is_train
        self.max_token_length = args.num_text_tokens
        self.word_mask_rate = args.word_mask_rate
        # whether use image-text matching task
        self.if_itm = True if args.loss_type['itm'] == 1 else False    
        # whether use image-text generation task
        # self.if_itg = True if args.loss_type['itg'] == 1 else False

        self.mask_ratio = args.mask_ratio
        self.mask_strategy = args.mask_strategy
        self.mask_patch_size = args.mask_patch_size

        self.dataset_type = data_type

        # get filenames
        if self.dataset_type == 'train':
            print('>>> load FashionGenDataset < Pre-Training > train phase')
            image_root = os.path.join(root, 'extracted_train_images')
            text_info_root = os.path.join(root, 'full_train_info_PAI')
            # sort files
            if ONLY_FOR_DEBUG:
                self.images = sorted([image_root + '/' + f for f in os.listdir(image_root)])[:5000]
                self.text_dicts = sorted([text_info_root + '/' + f for f in os.listdir(text_info_root)])[:5000]
            else:
                self.images = sorted([image_root + '/' + f for f in os.listdir(image_root)])
                self.text_dicts = sorted([text_info_root + '/' + f for f in os.listdir(text_info_root)])
        elif self.dataset_type == 'valid':
            print('>>> load FashionGenDataset < Pre-Training > valid phase')
            image_root = os.path.join(root, 'extracted_valid_images')
            text_info_root = os.path.join(root, 'full_valid_info_PAI')
            noisy_image_root = os.path.join(root, 'generated_valid_noise_images')
            noisy_text_root = os.path.join(root, 'generated_valid_noise_texts')
            if self.mask_strategy == 'random_grid':
                if self.trainsize == 256:
                    masking_image_root = os.path.join(root, 'generated_valid_masking{:.2f}_size{}_images'.format(self.mask_ratio, self.mask_patch_size))
                else:
                    masking_image_root = os.path.join(root, 'generated_valid_masking{:.2f}_size{}_img{}_images'.format(self.mask_ratio, self.mask_patch_size, self.trainsize))
            else:
                masking_image_root = os.path.join(root, 'generated_valid_noise_images')

            print('>>> routine checkpoint (for masking_image_root): {}'.format(masking_image_root))
            
            # sort files (only select all entries for fast evaluation during training)
            if ONLY_FOR_DEBUG:
                self.images = sorted([image_root + '/' + f for f in os.listdir(image_root)])[:1000]
                self.text_dicts = sorted([text_info_root + '/' + f for f in os.listdir(text_info_root)])[:1000]
                self.noisy_images = sorted([noisy_image_root + '/' + f for f in os.listdir(noisy_image_root)])[:1000]
                self.noisy_texts = sorted([noisy_text_root + '/' + f for f in os.listdir(noisy_text_root)])[:1000]
                self.grid_masking_images = sorted([masking_image_root + '/' + f for f in os.listdir(masking_image_root)])[:1000]
            else:
                self.images = sorted([image_root + '/' + f for f in os.listdir(image_root)])
                self.text_dicts = sorted([text_info_root + '/' + f for f in os.listdir(text_info_root)])
                self.noisy_images = sorted([noisy_image_root + '/' + f for f in os.listdir(noisy_image_root)])
                self.noisy_texts = sorted([noisy_text_root + '/' + f for f in os.listdir(noisy_text_root)])
                self.grid_masking_images = sorted([masking_image_root + '/' + f for f in os.listdir(masking_image_root)])
        else:
            raise Exception('No type named {}'.format(self.dataset_type))

        # raise assertion if the total image & text pairs are not matching
        assert len(self.images) == len(self.text_dicts)

        # define data transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    # TODO: check whether need a normalization of all data
            ])
        
        # get size of dataset
        self.size = len(self.images)

        print('>>> FashionGenDataset Configuration:\n\troot: {},\n\ttrainsize: {},\n\tdata_type: {},\n\tif_itm: {},\n\tmask_ratio: {},\n\tmask_strategy: {},\n\tmax_token_length: {},\n\tword_mask_rate: {},\n\tis_train: {}\n\timage_root: {},\n\ttext_info_root: {},\n\tlength_of_dataset: {}'.format(root, self.trainsize, self.dataset_type, self.if_itm, self.mask_ratio, self.mask_strategy, self.max_token_length, self.word_mask_rate, self.is_train, image_root, text_info_root, self.size))

    def __getitem__(self, index):
        # data convertion in PyTorch: https://blog.csdn.net/moshiyaofei/article/details/90519430
        # get itm_label for pre-training
        if self.if_itm:
            # itm_probability = 0.5
            if random.random() <= 0.5:
                # not random the token_index
                text_dict = self.pkl_loader(self.text_dicts[index])

                # print(self.images[index].split('/')[-1], text_dict['img_name'])
                assert self.images[index].split('/')[-1] == text_dict['img_name']   # check whether match

                # create itm label
                itm_labels = torch.tensor([1], dtype=torch.long)
            else:
                # random the token_index
                increment_index = index + random.randint(50, self.size // 2)  # Daniel
                # increment_index = index + 1
                # if the value overflows
                if increment_index > self.size - 1:
                    increment_index -= self.size
                # load text using `pkl_loader` function
                text_dict = self.pkl_loader(self.text_dicts[increment_index])

                # print(self.images[index].split('/')[-1].split('_')[0], text_dict['img_name'].split('_')[0])
                assert self.images[index].split('/')[-1].split('_')[0] != text_dict['img_name'].split('_')[0]   # check if not match  # Daniel

                # create itm label
                itm_labels = torch.tensor([0], dtype=torch.long)
        else:
            text_dict = self.pkl_loader(self.text_dicts[index])
            # print(self.images[index].split('/')[-1], text_dict['img_name'])
            assert self.images[index].split('/')[-1] == text_dict['img_name']   # check if match

            # create itm label
            itm_labels = 0  # plz do not use <class 'NoneType'>
        
        # load image and transform it
        image = self.rgb_loader(self.images[index])
        image = self.img_transform(image)

        # get image mask for masking strategy
        if self.dataset_type == 'train':
            if self.mask_strategy == 'square':
                img_mask = self.generate_square_mask(im_size=self.trainsize, mask_size=self.trainsize//self.mask_ratio)
            elif self.mask_strategy == 'stroke':
                img_mask = self.generate_stroke_mask(im_size=self.trainsize)
            elif self.mask_strategy == 'random_grid':
                img_mask = self.generate_grid_mask(input_size=(self.trainsize, self.trainsize), mask_ratio=self.mask_ratio, patch_size=16)
            else:
                raise NameError('>>> invalid parameter: {}'.format(self.mask_strategy))
        elif self.dataset_type == 'valid':
            # print(self.grid_masking_images[index])
            img_mask = self.pkl_loader(self.grid_masking_images[index])
        else:
            raise Exception('No type named {}'.format(self.dataset_type))
        
        masked_images = image.clone().masked_fill_(torch.Tensor(img_mask).byte().bool(), value=torch.tensor(1e-6))
        t2i_labels = torch.Tensor(img_mask)

        # process text data
        input_ids, attention_mask, mlm_labels, segment_ids, ori_input_ids, i2t_labels, bartMSS_input_dict = self.text_process(prod_caption=text_dict['captions'], max_token_length=self.max_token_length)

        # get noisy data for generation
        if self.dataset_type == 'train':
            img_arr = np.random.randint(0, 256, (self.trainsize, self.trainsize, 3)).astype('uint8')
            n_image = Image.fromarray(img_arr).convert('RGB')
            n_image = self.img_transform(n_image)

            text_arr = np.random.randint(0, 30522, (self.max_token_length, 1))
            n_input_ids = torch.from_numpy(text_arr).long().squeeze()
        elif self.dataset_type == 'valid':
            n_image = self.rgb_loader(self.noisy_images[index])
            n_image = self.img_transform(n_image)

            text_arr = self.pkl_loader(self.noisy_texts[index])
            n_input_ids = torch.from_numpy(text_arr).long().squeeze()   # Daniel
            # n_input_ids = None
        else:
            raise Exception('No type named {}'.format(self.dataset_type))
        # get some data infos for debugging
        # print('>>> Debug-L129:', input_ids, attention_mask, mlm_labels, segment_ids)
        data_info = dict()
        data_info.update(img_name=self.images[index].split('/')[-1])
        # data_info.update(itg_mask=img_mask)

        # Daniel
        sup_cls_labels = torch.tensor([int(text_dict['super_cls_id'])], dtype=torch.long)
        sub_cls_labels = torch.tensor([int(text_dict['sub_cls_id'])], dtype=torch.long)

        item_dict = {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'masked_images': masked_images,
            'mlm_labels': mlm_labels,
            'segment_ids': segment_ids,
            'itm_labels': itm_labels,
            'i2t_labels': i2t_labels,
            'bartMSS_input_dict': bartMSS_input_dict,
            'data_info': data_info,
            'n_image': n_image,
            'n_input_ids': n_input_ids,
            'ori_input_ids': ori_input_ids,
            'sup_cls_labels': sup_cls_labels,
            'sub_cls_labels': sub_cls_labels,
            't2i_labels': t2i_labels
        }

        return item_dict

    def __len__(self):
        return self.size

    @staticmethod
    def tensor2pil(masked_img, image):
        """only for debug"""
        masked_img = transforms.ToPILImage()(masked_img)
        masked_img.save('./img_masked.png')

        image = transforms.ToPILImage()(image)
        image.save('./image.png')

    def generate_grid_mask(self, input_size=(352, 352), mask_ratio=0.75, patch_size=16):
        # ensure input_width and input_height are divisible by patch_size
        assert input_size[0] % patch_size == 0
        assert input_size[1] % patch_size == 0

        num_width = input_size[0] // patch_size
        num_height = input_size[1] // patch_size

        num_patches = num_width * num_height
        num_mask = int(mask_ratio * num_patches)

        mask = np.concatenate([
            np.zeros((num_patches - num_mask, patch_size, patch_size)),     # the number of unmasked patches
            np.ones((num_mask, patch_size, patch_size)),    # the number of masked patches
        ], axis=0)

        mask_split = np.split(mask, num_patches, axis=0)
        np.random.shuffle(mask_split)   # random

        h_list = list()
        for i in range(num_height):
            cur_list = mask_split[0 + i: i + num_width]
            np.random.shuffle(cur_list)
            h_list.append(np.transpose(np.hstack(cur_list), (2, 1, 0)))

        final_mask = np.vstack(h_list)

        # final_mask = np.uint8(final_mask.squeeze()) * 255
        # cv2.imwrite(save_name, final_mask)
        return np.transpose(final_mask, (2, 0, 1))

    def generate_square_mask(self, im_size, mask_size):
        # print('debug271', im_size, mask_size)
        # print('Debug272', mask_size//2, im_size-mask_size//2, mask_size//2, im_size-mask_size//2)
        mask_center = (np.random.randint(mask_size//2, im_size-mask_size//2), np.random.randint(mask_size//2, im_size-mask_size//2))
        min_x, max_x, min_y, max_y = \
            mask_center[0] - mask_size // 2, mask_center[0] + mask_size // 2, \
            mask_center[1] - mask_size // 2, mask_center[1] + mask_size // 2

        mask = np.zeros((1, im_size, im_size))
        mask[:, int(min_x):int(max_x), int(min_y):int(max_y)] = 1
        return mask
        # return torch.Tensor(mask).byte()    # convert to byte format (blog: https://zhuanlan.zhihu.com/p/151783950)
    
    def generate_stroke_mask(self, im_size, maxAngle=360, mask_scale=1):
        maxLength = im_size
        maxVertex = im_size // (70/mask_scale)
        maxBrushWidth = im_size // (25/mask_scale)

        mask = np.zeros((im_size, im_size, 1), dtype=np.float32)
        parts = random.randint(5, 13)
        # print(parts)
        for _ in range(parts):
            # print(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size, im_size)
            mask = mask + self.np_free_form_mask(maxVertex, maxLength,
                                            maxBrushWidth, maxAngle, im_size, im_size)
        mask = np.minimum(mask, 1.0)
        mask = np.transpose(mask, (2, 0, 1))
        # mask = np.expand_dims(mask, axis=0)
        return mask
        # return torch.Tensor(mask).byte()
    
    @staticmethod
    def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
        """
        :param maxVertex: 数目
        :param maxLength: 长度
        :param maxBrushWidth: 宽度
        :param maxAngle:
        :param h:
        :param w:
        :return:
        """
        mask = np.zeros((h, w, 1), np.float32)
        numVertex = np.random.randint(maxVertex + 1)
        startY = np.random.randint(h)
        startX = np.random.randint(w)
        brushWidth = 0
        for i in range(numVertex):
            angle = np.random.randint(maxAngle + 1)
            angle = angle / 360.0 * 2 * np.pi
            if i % 2 == 0:
                angle = 2 * np.pi - angle
            length = np.random.randint(maxLength + 1)
            brushWidth = np.random.randint(5, maxBrushWidth + 1) // 2 * 2
            nextY = startY + length * np.cos(angle)
            nextX = startX + length * np.sin(angle)
            nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
            nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
            cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
            cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
            startY, startX = nextY, nextX
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)

        return mask

    def text_process(
        self, prod_caption, max_token_length, 
        cls_token_segment_id=0, sequence_segment_id=0, pad_token_segment_id=0, pad_token_id=0, decoder_start_token_id=2):
        ori_tokens_tmp = self.tokenizer.tokenize(prod_caption)  # list()

        if len(ori_tokens_tmp) > max_token_length - 2:
            # drop tails and save the position of [CLS], [SEP].
            ori_tokens_tmp = ori_tokens_tmp[:(max_token_length - 2)]

        # for Masking Seq-to-Seq
        ori_tokens = [self.tokenizer.cls_token] + ori_tokens_tmp + [self.tokenizer.sep_token]   

        # get the tokenizations and corresponding lm_label -> [mask] with -1 
        # print('>>> before mlm: ', tokens)
        tokens, mlm_labels = self.random_masking_features(ori_tokens_tmp)
        # generate tokens
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]

        # print('Debug-L292', tokens == ori_tokens) # judge two variants are equal
        
        segment_ids = [cls_token_segment_id] + [sequence_segment_id] * (len(tokens)-1)
        token_len = len(tokens)

        # pad on the right of text tokens
        seq_padding_len = max_token_length - token_len
        tokens += [self.tokenizer.pad_token] * seq_padding_len
        ori_tokens += [self.tokenizer.pad_token] * seq_padding_len
        # get the segment_ids
        segment_ids += [pad_token_segment_id] * seq_padding_len
        # pad on the right of mlm_labels and then pad on the start/end point of it (due to [cls_token]/[sep_token])
        mlm_labels += [-1] * seq_padding_len
        mlm_labels = [-1] + mlm_labels + [-1]         
        # tokens to ids
        # print('>>> after mlm: ', tokens, mlm_labels)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ori_input_ids = self.tokenizer.convert_tokens_to_ids(ori_tokens)
        # get attention mask
        attention_mask = [1] * token_len + [0] * seq_padding_len
        
        # convert to pytorch-tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        ori_input_ids = torch.tensor(ori_input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        mlm_labels = torch.tensor(mlm_labels, dtype=torch.long)    # mark (-1) which token is masked
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        # print(prod_caption, input_ids, attention_mask, mlm_labels, segment_ids)
        decoder_input_ids = shift_tokens_right(ori_input_ids.unsqueeze(0), pad_token_id=0, decoder_start_token_id=2)
        i2t_labels = ori_input_ids.unsqueeze(0).clone()
        i2t_labels[i2t_labels[:, :] == pad_token_id] = -1

        # print('Debug-L324', input_ids == ori_input_ids)

        bartMSS_input_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids.squeeze(),
            'labels': mlm_labels
        }

        return input_ids, attention_mask, mlm_labels, segment_ids, ori_input_ids, i2t_labels.squeeze(), bartMSS_input_dict

    def text_process_bak(self, prod_caption, max_token_length, cls_token_segment_id=0, sequence_segment_id=0, pad_token_segment_id=0):
        tokens = self.tokenizer.tokenize(prod_caption)  # list()

        if len(tokens) > max_token_length - 2:
            # drop tails and save the position of [CLS], [SEP].
            tokens = tokens[:(max_token_length - 2)]

        # get the tokenizations and corresponding lm_label -> [mask] with -1 
        # print('>>> before mlm: ', tokens)
        tokens, mlm_labels = self.random_masking_features(tokens)

        # generate tokens
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_segment_id] * (len(tokens)-1)
        token_len = len(tokens)

        # pad on the right of text tokens
        seq_padding_len = max_token_length - token_len
        tokens += [self.tokenizer.pad_token] * seq_padding_len
        # get the segment_ids
        segment_ids += [pad_token_segment_id] * seq_padding_len
        # pad on the right of mlm_labels and then pad on the start/end point of it (due to [cls_token]/[sep_token])
        mlm_labels += [-1] * seq_padding_len
        mlm_labels = [-1] + mlm_labels + [-1]         
        # tokens to ids
        # print('>>> after mlm: ', tokens, mlm_labels)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # print(type(input_ids), len(input_ids), input_ids)
        # get attention mask
        attention_mask = [1] * token_len + [0] * seq_padding_len
        
        # convert to pytorch-tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        mlm_labels = torch.tensor(mlm_labels, dtype=torch.long)    # mark (-1) which token is masked
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        # print(prod_caption, input_ids, attention_mask, mlm_labels, segment_ids)
        decoder_input_ids = shift_tokens_right(input_ids.unsqueeze(0), pad_token_id=0, decoder_start_token_id=2)

        bartMSS_input_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids.squeeze(),
            'labels': mlm_labels
        }

        return input_ids, attention_mask, mlm_labels, segment_ids, bartMSS_input_dict

    def random_masking_features(self, tokens):
        """ random masking text features """
        lm_label = []
        # masking text-side
        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with probability
            ratio = self.word_mask_rate
            if prob < ratio:
                prob /= ratio 
                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = "[MASK]"
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.choice(list(self.tokenizer.vocab.items()))[0]
                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                try:
                    lm_label.append(self.tokenizer.vocab[token])
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    lm_label.append(self.tokenizer.vocab["[UNK]"])
            else:
                # no masking token (will be ignored by loss function later)
                lm_label.append(-1)
        return tokens, lm_label 

    def rgb_loader(self, img_path, if_crop=True):
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            if if_crop:
                # keep the 
                img_npy = np.array(img.convert('1'))
                # img.save('./ori.png')
                # gray_img.save('./gray.png')
                coord = (img_npy == False).nonzero() 
                # print(coord)
                w_top, w_bottom = coord[1].min(), coord[1].max()
                h_top, h_bottom = coord[0].min(), coord[0].max()
                # print('h_top={}, h_bottom={}, w_top={}, w_bottom={}'.format(h_top, h_bottom, w_top, w_bottom))
                crop_img = img.crop((w_top, h_top, w_bottom, h_bottom))
                # crop_img.save('./crop_img.png')
                return crop_img.convert('RGB')
            else:
                return img.convert('RGB')
    
    def pkl_loader(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            info_dict = pickle.load(f)
            return info_dict


class FashionGenDatasetDownstream_Retrieval(data.Dataset):
    """ loading fashion-gen dataset (downstream).
    @self.text_dicts: 
        'product_id', 'img_name', 'super_cls_name', 'super_cls_id', sub_cls_name', 'sub_cls_id', 
        'captions', 'input_ids', 'token_type_ids', 'attention_mask'
    """
    def __init__(self, root, args):
        # pre-define
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.trainsize = args.input_size
        self.max_token_length = args.num_text_tokens
        self.word_mask_rate = args.word_mask_rate
        self.image_root = os.path.join(root, 'extracted_valid_images')

        # get filenames
        self.eval_retrieval_itr = args.eval_retrieval_itr
        self.eval_retrieval_tir = args.eval_retrieval_tir

        if self.eval_retrieval_itr:
            itr_root = os.path.join(root, 'retrieve_ITR')
            self.pkls_itr = sorted([itr_root + '/' + f for f in os.listdir(itr_root)])
            self.size = len(self.pkls_itr)
        elif self.eval_retrieval_tir:
            tir_root = os.path.join(root, 'retrieve_TIR')
            self.pkls_tir = sorted([tir_root + '/' + f for f in os.listdir(tir_root)])
            self.size = len(self.pkls_tir)
        else:
            raise Exception('No types implemented')

        # define data transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
            ])

    def __getitem__(self, index):
        if self.eval_retrieval_itr:
            dict_101 = self.pkl_loader(self.pkls_itr[index])
        elif self.eval_retrieval_tir:
            dict_101 = self.pkl_loader(self.pkls_tir[index])

        text_list, img_list, info_list = list(), list(), list()

        for key, value in dict_101.items():
            _data_info = dict()
            _prod_caption = value['captions']
            _img_name = value['img_name']

            # load and process text
            ori_input_ids = self.text_process(prod_caption=_prod_caption, max_token_length=self.max_token_length)[4]

            # load and process image
            image = self.img_transform(self.rgb_loader(os.path.join(self.image_root, _img_name)))

            # load data info
            _data_info.update(prod_caption=_prod_caption, img_name=_img_name)

            # save
            text_list.append(ori_input_ids)
            img_list.append(image)
            info_list.append(copy.deepcopy(_data_info))

        # concatenation
        ori_input_ids_101 = torch.stack(text_list, dim=0)
        images_101 = torch.stack(img_list, dim=0)

        item_dict = dict()
        item_dict.update(
            ori_input_ids_101=ori_input_ids_101,
            images_101=images_101,
            info_list=info_list
        )
        return item_dict

    def __len__(self):
        return self.size

    def text_process(
        self, prod_caption, max_token_length, 
        cls_token_segment_id=0, sequence_segment_id=0, pad_token_segment_id=0, pad_token_id=0, decoder_start_token_id=2):
        ori_tokens_tmp = self.tokenizer.tokenize(prod_caption)  # list()

        if len(ori_tokens_tmp) > max_token_length - 2:
            # drop tails and save the position of [CLS], [SEP].
            ori_tokens_tmp = ori_tokens_tmp[:(max_token_length - 2)]

        # for Masking Seq-to-Seq
        ori_tokens = [self.tokenizer.cls_token] + ori_tokens_tmp + [self.tokenizer.sep_token]   

        # get the tokenizations and corresponding lm_label -> [mask] with -1 
        # print('>>> before mlm: ', tokens)
        tokens, mlm_labels = self.random_masking_features(ori_tokens_tmp)
        # generate tokens
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]

        # print('Debug-L292', tokens == ori_tokens) # judge two variants are equal
        
        segment_ids = [cls_token_segment_id] + [sequence_segment_id] * (len(tokens)-1)
        token_len = len(tokens)

        # pad on the right of text tokens
        seq_padding_len = max_token_length - token_len
        tokens += [self.tokenizer.pad_token] * seq_padding_len
        ori_tokens += [self.tokenizer.pad_token] * seq_padding_len
        # get the segment_ids
        segment_ids += [pad_token_segment_id] * seq_padding_len
        # pad on the right of mlm_labels and then pad on the start/end point of it (due to [cls_token]/[sep_token])
        mlm_labels += [-1] * seq_padding_len
        mlm_labels = [-1] + mlm_labels + [-1]         
        # tokens to ids
        # print('>>> after mlm: ', tokens, mlm_labels)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ori_input_ids = self.tokenizer.convert_tokens_to_ids(ori_tokens)
        # get attention mask
        attention_mask = [1] * token_len + [0] * seq_padding_len
        
        # convert to pytorch-tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        ori_input_ids = torch.tensor(ori_input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        mlm_labels = torch.tensor(mlm_labels, dtype=torch.long)    # mark (-1) which token is masked
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        # print(prod_caption, input_ids, attention_mask, mlm_labels, segment_ids)
        decoder_input_ids = shift_tokens_right(ori_input_ids.unsqueeze(0), pad_token_id=0, decoder_start_token_id=2)
        i2t_labels = ori_input_ids.unsqueeze(0).clone()
        i2t_labels[i2t_labels[:, :] == pad_token_id] = -1

        # print('Debug-L324', input_ids == ori_input_ids)

        bartMSS_input_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids.squeeze(),
            'labels': mlm_labels
        }

        return input_ids, attention_mask, mlm_labels, segment_ids, ori_input_ids, i2t_labels.squeeze(), bartMSS_input_dict

    def random_masking_features(self, tokens):
        """ random masking text features """
        lm_label = []
        # masking text-side
        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with probability
            ratio = self.word_mask_rate
            if prob < ratio:
                prob /= ratio 
                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = "[MASK]"
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.choice(list(self.tokenizer.vocab.items()))[0]
                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                try:
                    lm_label.append(self.tokenizer.vocab[token])
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    lm_label.append(self.tokenizer.vocab["[UNK]"])
            else:
                # no masking token (will be ignored by loss function later)
                lm_label.append(-1)
        return tokens, lm_label 

    def rgb_loader(self, img_path, if_crop=True):
        """ load image from *.png """
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            if if_crop:
                # keep the 
                img_npy = np.array(img.convert('1'))
                # img.save('./ori.png')
                # gray_img.save('./gray.png')
                coord = (img_npy == False).nonzero() 
                # print(coord)
                w_top, w_bottom = coord[1].min(), coord[1].max()
                h_top, h_bottom = coord[0].min(), coord[0].max()
                # print('h_top={}, h_bottom={}, w_top={}, w_bottom={}'.format(h_top, h_bottom, w_top, w_bottom))
                crop_img = img.crop((w_top, h_top, w_bottom, h_bottom))
                # crop_img.save('./crop_img.png')
                return crop_img.convert('RGB')
            else:
                return img.convert('RGB')
    
    def pkl_loader(self, pkl_path):
        """ load text from *.pkl """
        with open(pkl_path, 'rb') as f:
            info_dict = pickle.load(f)
            return info_dict


class FashionGenDatasetDownstream_Recognition(data.Dataset):
    """ loading fashion-gen dataset (downstream).
    @self.text_dicts: 
        'product_id', 'img_name', 'super_cls_name', 'super_cls_id', sub_cls_name', 'sub_cls_id', 
        'captions', 'input_ids', 'token_type_ids', 'attention_mask'
    """
    def __init__(self, root, args):
        # pre-define
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.trainsize = args.input_size
        self.max_token_length = args.num_text_tokens
        self.word_mask_rate = args.word_mask_rate

        image_root = os.path.join(root, 'extracted_valid_images')
        text_info_root = os.path.join(root, 'full_valid_info_PAI')

        self.images = sorted([image_root + '/' + f for f in os.listdir(image_root)])
        self.text_dicts = sorted([text_info_root + '/' + f for f in os.listdir(text_info_root)])

        # raise assertion if the total image & text pairs are not matching
        assert len(self.images) == len(self.text_dicts)

        # define data transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
            ])

        # get size of dataset
        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        image = self.img_transform(image)
        text_dict = self.pkl_loader(self.text_dicts[index])
        # process text data
        ori_input_ids = self.text_process(prod_caption=text_dict['captions'], max_token_length=self.max_token_length)[4]

        sup_cls_labels = torch.tensor([int(text_dict['super_cls_id'])], dtype=torch.long)
        sub_cls_labels = torch.tensor([int(text_dict['sub_cls_id'])], dtype=torch.long)
        # print(text_dict)

        item_dict = dict()
        item_dict.update(
            ori_input_ids=ori_input_ids,
            images=image,
            sup_cls_labels=sup_cls_labels,
            sub_cls_labels=sub_cls_labels,
            info_list=text_dict['img_name']
        )
        return item_dict

    def __len__(self):
        return self.size

    def text_process(
        self, prod_caption, max_token_length, 
        cls_token_segment_id=0, sequence_segment_id=0, pad_token_segment_id=0, pad_token_id=0, decoder_start_token_id=2):
        ori_tokens_tmp = self.tokenizer.tokenize(prod_caption)  # list()

        if len(ori_tokens_tmp) > max_token_length - 2:
            # drop tails and save the position of [CLS], [SEP].
            ori_tokens_tmp = ori_tokens_tmp[:(max_token_length - 2)]

        # for Masking Seq-to-Seq
        ori_tokens = [self.tokenizer.cls_token] + ori_tokens_tmp + [self.tokenizer.sep_token]   

        # get the tokenizations and corresponding lm_label -> [mask] with -1 
        # print('>>> before mlm: ', tokens)
        tokens, mlm_labels = self.random_masking_features(ori_tokens_tmp)
        # generate tokens
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]

        # print('Debug-L292', tokens == ori_tokens) # judge two variants are equal
        
        segment_ids = [cls_token_segment_id] + [sequence_segment_id] * (len(tokens)-1)
        token_len = len(tokens)

        # pad on the right of text tokens
        seq_padding_len = max_token_length - token_len
        tokens += [self.tokenizer.pad_token] * seq_padding_len
        ori_tokens += [self.tokenizer.pad_token] * seq_padding_len
        # get the segment_ids
        segment_ids += [pad_token_segment_id] * seq_padding_len
        # pad on the right of mlm_labels and then pad on the start/end point of it (due to [cls_token]/[sep_token])
        mlm_labels += [-1] * seq_padding_len
        mlm_labels = [-1] + mlm_labels + [-1]         
        # tokens to ids
        # print('>>> after mlm: ', tokens, mlm_labels)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ori_input_ids = self.tokenizer.convert_tokens_to_ids(ori_tokens)
        # get attention mask
        attention_mask = [1] * token_len + [0] * seq_padding_len
        
        # convert to pytorch-tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        ori_input_ids = torch.tensor(ori_input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        mlm_labels = torch.tensor(mlm_labels, dtype=torch.long)    # mark (-1) which token is masked
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        # print(prod_caption, input_ids, attention_mask, mlm_labels, segment_ids)
        decoder_input_ids = shift_tokens_right(ori_input_ids.unsqueeze(0), pad_token_id=0, decoder_start_token_id=2)
        i2t_labels = ori_input_ids.unsqueeze(0).clone()
        i2t_labels[i2t_labels[:, :] == pad_token_id] = -1

        # print('Debug-L324', input_ids == ori_input_ids)

        bartMSS_input_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids.squeeze(),
            'labels': mlm_labels
        }

        return input_ids, attention_mask, mlm_labels, segment_ids, ori_input_ids, i2t_labels.squeeze(), bartMSS_input_dict

    def random_masking_features(self, tokens):
        """ random masking text features """
        lm_label = []
        # masking text-side
        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with probability
            ratio = self.word_mask_rate
            if prob < ratio:
                prob /= ratio 
                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = "[MASK]"
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.choice(list(self.tokenizer.vocab.items()))[0]
                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                try:
                    lm_label.append(self.tokenizer.vocab[token])
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    lm_label.append(self.tokenizer.vocab["[UNK]"])
            else:
                # no masking token (will be ignored by loss function later)
                lm_label.append(-1)
        return tokens, lm_label 

    def rgb_loader(self, img_path, if_crop=True):
        """ load image from *.png """
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            if if_crop:
                # keep the 
                img_npy = np.array(img.convert('1'))
                # img.save('./ori.png')
                # gray_img.save('./gray.png')
                coord = (img_npy == False).nonzero() 
                # print(coord)
                w_top, w_bottom = coord[1].min(), coord[1].max()
                h_top, h_bottom = coord[0].min(), coord[0].max()
                # print('h_top={}, h_bottom={}, w_top={}, w_bottom={}'.format(h_top, h_bottom, w_top, w_bottom))
                crop_img = img.crop((w_top, h_top, w_bottom, h_bottom))
                # crop_img.save('./crop_img.png')
                return crop_img.convert('RGB')
            else:
                return img.convert('RGB')
    
    def pkl_loader(self, pkl_path):
        """ load text from *.pkl """
        with open(pkl_path, 'rb') as f:
            info_dict = pickle.load(f)
            return info_dict


def get_args_parser():
    parser = argparse.ArgumentParser('PVT training and evaluation script', add_help=False)

    # # VL parameters
    parser.add_argument('--num-text-tokens', default=128, type=int, metavar='VL', help='number of text tokens')
    parser.add_argument('--token-hidden-size', default=768, type=int, metavar='VL', help='token hidden size')
    parser.add_argument('--word-mask-rate', default=0.15, type=float, metavar='VL', help='word_mask_rate in masking strategy')
    parser.add_argument('--loss-type', default={'itm':1, 'mlm':1, 'itg':0, 'i2t':0, 't2i':1, 'cls':1, 'rtd':0, 'bartNSG': 0, 'bartMSS':0}, type=dict, metavar='VL', help='please indicate the loss type')
    parser.add_argument('--mask-ratio', default=0.50, type=int, metavar='VL', help='mask ratio in itg task')
    parser.add_argument('--mask-strategy', default='random_grid', type=str, metavar='VL', help='choice: square or stroke or random_grid')
    parser.add_argument('--pretrain-pth', default='./preweights/pvt_v1/pvt_tiny.pth', type=str, metavar='VL', help='please indicate the loss type')
    parser.add_argument('--mask-patch-size', default=8, type=int, metavar='VL', help='choice: square or stroke or random_grid')
    parser.add_argument('--eval_retrieval_itr', default=True, type=bool, metavar='VL', help='whether evaluate retrieval_itr')
    parser.add_argument('--eval_retrieval_tir', default=False, type=bool, metavar='VL', help='whether evaluate retrieval_tir')
    
    parser.add_argument('--input-size', default=256, type=int, help='images input size')

    # # Dataset parameters
    parser.add_argument('--data-path', default='/home/admin/workspace/daniel_ji/dataset/Fashion-Gen', type=str,
                        help='dataset path')
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    # train_loader = data.DataLoader(
    #     dataset=FashionGenDatasetPreTrain(
    #         root=args.data_path, 
    #         data_type='valid', # 'train'
    #         is_train=False, # True
    #         args=args),
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=1,
    #     pin_memory=True,
    #     drop_last=False)

        
    # for i, item_dict in enumerate(train_loader, start=1):
    #     # @Danil: watch the shape of each variables
    #     print(
    #         i, item_dict['image'].shape, item_dict['input_ids'].shape, item_dict['attention_mask'].shape, item_dict['mlm_labels'].shape, 
    #         item_dict['bartMSS_input_dict']['input_ids'].shape, item_dict['bartMSS_input_dict']['attention_mask'].shape, item_dict['bartMSS_input_dict']['decoder_input_ids'].shape,
    #         item_dict['bartMSS_input_dict']['labels'].shape
    #     )
        
    #     # @Danil: watch the exact value of each variables 
    #     print(
    #         '>>> image\n', item_dict['image'].shape, '\n>>>', item_dict['image'], '\n>>>', 
    #     '>>> input_ids\n', item_dict['input_ids'].shape, '\n>>>', item_dict['input_ids'], '\n>>>',
    #     '>>> n_input_ids\n', item_dict['n_input_ids'].shape, '\n>>>', item_dict['n_input_ids'], '\n>>>',
    #     '>>> ori_input_ids\n', item_dict['ori_input_ids'].shape, '\n>>>', item_dict['ori_input_ids'], '\n>>>', 
    #     '>>> attention_mask\n', item_dict['attention_mask'].shape, '\n>>>', item_dict['attention_mask'],
    #     '>>> mlm_labels\n', item_dict['mlm_labels'].shape, '\n>>>', item_dict['mlm_labels'],
    #     '>>> i2t_labels\n', item_dict['i2t_labels'].shape, '\n>>>', item_dict['i2t_labels'])
    #     print('>>> bartMSS_input_dict-input_ids\n', item_dict['bartMSS_input_dict']['input_ids'].shape, '\n>>>', item_dict['bartMSS_input_dict']['input_ids'], '\n>>>', 
    #     '>>> bartMSS_input_dict-attention_mask\n', item_dict['bartMSS_input_dict']['attention_mask'].shape, '\n>>>', item_dict['bartMSS_input_dict']['attention_mask'], '\n>>>', 
    #     '>>> bartMSS_input_dict-decoder_input_ids\n', item_dict['bartMSS_input_dict']['decoder_input_ids'].shape, '\n>>>', item_dict['bartMSS_input_dict']['decoder_input_ids'], '\n>>>',
    #     '>>> bartMSS_input_dict-labels\n', item_dict['bartMSS_input_dict']['labels'].shape, '\n>>>', item_dict['bartMSS_input_dict']['labels'], '\n>>>')
    #     print('#'*30)

        # import torchvision.transforms as transforms
        # img = transforms.ToPILImage()(item_dict['masked_images'].squeeze())
        # save_pth = './bak_for_debug_1207/'
        # os.makedirs(save_pth, exist_ok=True)
        # img.save(save_pth+item_dict['data_info']['img_name'][0])
        # pass

    # retrieval_loader = data.DataLoader(
    #     dataset=FashionGenDatasetDownstream_Retrieval(root=args.data_path, args=args),
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=1,
    #     pin_memory=True,
    #     drop_last=False)

    # for i, item_dict in enumerate(retrieval_loader, start=1):
    #     # @Danil: watch the exact value of each variables (retrieval-itr)
    #     print(
    #         '>>> ori_input_ids_101\n', item_dict['ori_input_ids_101'].shape, '\n>>>', item_dict['ori_input_ids_101'][0],
    #         '>>> images_101\n', item_dict['images_101'].shape, '\n>>>', item_dict['images_101'][0],
    #         # '>>> info_list\n', item_dict['info_list'], '\n>>>', item_dict['info_list'][0],
    #     )

    recognition_loader = data.DataLoader(
        dataset=FashionGenDatasetDownstream_Recognition(root=args.data_path, args=args),
        batch_size=10,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False)

    for i, item_dict in enumerate(recognition_loader, start=1):
        # @Danil: watch the exact value of each variables (retrieval-itr)
        print(
            '>>> ori_input_ids\n', item_dict['ori_input_ids'].shape, '\n>>>', item_dict['ori_input_ids'][0],
            '>>> images\n', item_dict['images'].shape, '\n>>>', item_dict['images'][0],
            '>>> info_list\n', item_dict['info_list'], '\n>>>', item_dict['info_list']
        )