import torch
import math
from torch import nn
from functools import reduce


class GELU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertHeadTransform, self).__init__()

        self.ActivePool = {'gelu': GELU(), 'relu': torch.nn.functional.relu, 'tanh': nn.Tanh}

        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        if isinstance(config['hidden_act'], str):
            self.transform_act_fn = self.ActivePool[config['hidden_act']]
        else:
            self.transform_act_fn = config['hidden_act']
        self.LayerNorm = nn.LayerNorm(config['hidden_size'])

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class MLMHead(nn.Module):
    '''
    Masked Language Modeling (MLM)
    '''
    def __init__(self, config, bert_model_embedding_weights):
        super(MLMHead, self).__init__()
        self.transform = BertHeadTransform(config)
        self.hidden_size = config['hidden_size']
        self.vocab_size = config['vocab_size']

        # throwing exceptions
        assert self.hidden_size == bert_model_embedding_weights.size(1), \
              '>>> hidden size: {} is not equal to bert embedding setting: {}'.format(
               self.hidden_size, bert_model_embedding_weights.size(1))

        assert self.vocab_size == bert_model_embedding_weights.size(0), \
               '>>> vocab size: {} is not equal to bert embedding setting: {}'.format(
               self.hidden_size, bert_model_embedding_weights.size(0))
        
        # 768, 30522
        self.mlm_decoder = nn.Linear(bert_model_embedding_weights.size(1), bert_model_embedding_weights.size(0), bias=False)

        # share weight with Bert-TextEmbeddings.
        print('using bert_model_embedding_weights')
        self.mlm_decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(30522))

    def forward(self, input_):
        hidden_states = self.transform(input_)
        hidden_states = self.mlm_decoder(hidden_states)
        mlm_logits = hidden_states + self.bias

        return mlm_logits   # bs, 128, 30522


class ITMHead(nn.Module):
    ''' < Pre-Training >
    Image Text Matching (ITM)
    '''
    def __init__(self, config):
        super(ITMHead, self).__init__()
        self.dim = config['hidden_size']

        self.linear = nn.Linear(self.dim, 2)
        self.linear_bias = nn.Parameter(torch.zeros(2))

    def forward(self, input_):
        itm_logits = self.linear(input_) + self.linear_bias
        
        return itm_logits


class CLSHead(nn.Module):
    ''' < Pre-Training >
    sup-/sub- cls
    '''
    def __init__(self, config, cls_num):
        super(CLSHead, self).__init__()
        self.dim = config['hidden_size']

        self.linear = nn.Linear(self.dim, cls_num)
        self.linear_bias = nn.Parameter(torch.zeros(cls_num))

    def forward(self, input_):
        cls_logits = self.linear(input_) + self.linear_bias
        
        return cls_logits


class ITGHead(nn.Module):
    """ < Pre-Training >
    Image-Text Generation (ITG)
    """
    def __init__(self, embed_dims, channel=64):
        super().__init__()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.reduction1 = self.ConvBN(embed_dims[1], channel, 3, padding=1)
        self.reduction2 = self.ConvBN(embed_dims[2], channel, 3, padding=1)
        self.reduction3 = self.ConvBN(embed_dims[3], channel, 3, padding=1)

        self.conv_upsample1 = self.ConvBN(channel, channel, 3, padding=1)
        self.conv_upsample2 = self.ConvBN(channel, channel, 3, padding=1)
        self.conv_upsample3 = self.ConvBN(channel, channel, 3, padding=1)
        self.conv_upsample4 = self.ConvBN(channel, channel, 3, padding=1)
        self.conv_upsample5 = self.ConvBN(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = self.ConvBN(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = self.ConvBN(3 * channel, 3 * channel, 3, padding=1)

        self.conv4 = self.ConvBN(3 * channel, 3 * channel, 3, padding=1)

        # score layer
        self.score = nn.Sequential(
            nn.Conv2d(3 * channel, 3, 1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True))

    def forward(self, low_feat, mid_feat, high_feat):

        refine_feature = self.combine_feature(
            self.reduction1(low_feat),
            self.reduction2(mid_feat),
            self.reduction3(high_feat))

        out = self.score(refine_feature)

        return out

    @staticmethod
    def ConvBN(in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        # Initialize Basic Convolutional Block and BN layer
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                                       stride=stride, padding=padding, dilation=dilation, bias=False),
                             nn.BatchNorm2d(out_planes))

    def combine_feature(self, low_feat, mid_feat, high_feat):  # [64, 16, 56, 56]
        x1_1 = high_feat
        x2_1 = self.conv_upsample1(self.upsample2(x1_1)) * mid_feat
        x3_1 = self.conv_upsample2(self.upsample2(mid_feat)) * self.conv_upsample3(self.upsample2(x2_1)) * low_feat

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample2(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample2(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        return self.conv4(x3_2)


class ELECTRA_Generator(nn.Module):
    def __init__(self, embed_dims, channel=64):
        super().__init__()

    def forward(self, input_ids):
        replace_prob = self.prob_mask_like(input_ids, self.replace_prob)

        # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
        # also do not include these special tokens in the tokens chosen at random
        no_mask = self.mask_with_tokens(input_ids, self.mask_ignore_token_ids)
        mask = self.get_mask_subset_with_prob(~no_mask, self.mask_prob)

        # get mask indices
        mask_indices = torch.nonzero(mask, as_tuple=True)

        # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)
        masked_input_ids = input_ids.clone().detach()

        # if random token probability > 0 for mlm
        if self.random_token_prob > 0:
            assert self.num_tokens is not None, 'Number of tokens (num_tokens) must be passed to Electra for randomizing tokens during masked language modeling'

            random_token_prob = self.prob_mask_like(input_ids, self.random_token_prob)
            random_tokens = torch.randint(0, self.num_tokens, input_ids.shape, device=input_ids.device)
            random_no_mask = self.mask_with_tokens(random_tokens, self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            random_indices = torch.nonzero(random_token_prob, as_tuple=True)
            masked_input_ids[random_indices] = random_tokens[random_indices]

        # [mask] input
        masked_input_ids = masked_input_ids.masked_fill(mask * replace_prob, self.mask_token_id)

        # set inverse of mask to padding tokens for labels
        rtd_gen_labels = input_ids.masked_fill(~mask, self.pad_token_id)

        # get generator output and get mlm loss
        rtd_logtis = self.generator(masked_input_ids)  # (1, 1024, 20000)

        return rtd_logtis, rtd_gen_labels

    @staticmethod
    def gumbel_sample(t, temperature=1.):
        # TODO: I am not sure this function is work well here. Please re-check it.
        def log(t, eps=1e-9):
            return torch.log(t + eps)

        def gumbel_noise(t):
            noise = torch.zeros_like(t).uniform_(0, 1)
            return -log(-log(noise))
        return ((t / temperature) + gumbel_noise(t)).argmax(dim=-1)

    @staticmethod
    def prob_mask_like(t, prob):
        return torch.zeros_like(t).float().uniform_(0, 1) < prob

    @staticmethod
    def mask_with_tokens(t, token_ids):
        init_no_mask = torch.full_like(t, False, dtype=torch.bool)
        mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
        return mask

    @staticmethod
    def get_mask_subset_with_prob(mask, prob):
        batch, seq_len, device = *mask.shape, mask.device
        max_masked = math.ceil(prob * seq_len)

        num_tokens = mask.sum(dim=-1, keepdim=True)
        mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
        mask_excess = mask_excess[:, :max_masked]

        rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
        _, sampled_indices = rand.topk(max_masked, dim=-1)
        sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

        new_mask = torch.zeros((batch, seq_len + 1), device=device)
        new_mask.scatter_(-1, sampled_indices, 1)

        return new_mask[:, 1:].bool()
