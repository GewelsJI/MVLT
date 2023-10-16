import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertModel
# from transformers.models.reformer.modeling_reformer import ReformerConfig, ReformerModel

from transformers import (
                BartConfig,
                BartForConditionalGeneration,
                BartModel
                )

from libs.vl_heads import MLMHead, ITMHead, ITGHead, CLSHead


__all__ = [
    'pvlt_tiny', 'pvlt_small', 'pvlt_medium', 'pvlt_large'
]


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -1, pad_token_id)

    return shifted_input_ids


def log(t, eps=1e-9):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=-1)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W, T_num):
        # print('debug97', x.shape)
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)    # bs, 1, 7744+40, 64

        if self.sr_ratio > 1:
            # process visual tokens and texture tokens individually
            x_, t_ = torch.split(x, [H*W, T_num], dim=1)
            x_ = x_.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            # print('debug106', x_.shape, t_.shape)
            x_ = torch.cat((x_, t_), dim=1) # bs, 121+40, 64
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # bs, 1, 121+40, 64

        attn = (q @ k.transpose(-2, -1)) * self.scale   # bs, 1, 7744+40, 121+40 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # bs, 7744+40, 64
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W, T_num):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, T_num))   # # bs, 7744+40, 64
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape # torch.Size([5, 3, 352, 352])

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PyramidVisionLanguageTransformer(nn.Module):
    """
    Pyramid Vision Language Transformer (PVLT) by Daniel@Alibaba Group
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512], 
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=4, F4=False,
                 # add new model configurations (VL part)
                 token_hidden_size=768, num_text_tokens=128, loss_type={'itm':1, 'mlm':1, 'itg':1, 'rtd':1}
                 ):
        super().__init__()

        print('>>> model configuration (VL part):\n\ttoken_hidden_size: {},\n\tnum_text_tokens: {},\n\tloss_type: {}'.format(token_hidden_size, num_text_tokens, loss_type))

        self.num_classes = num_classes
        self.depths = depths
        self.F4 = F4
        self.num_stages = num_stages
        self.T_num = num_text_tokens
        self.loss_type = loss_type

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                     patch_size=patch_size if i == 0 else 2,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            text_embed = nn.Sequential(
                nn.Linear(in_features=token_hidden_size if i==0 else embed_dims[i-1], 
                          out_features=embed_dims[i]),
                nn.LayerNorm(embed_dims[i]))
            num_patches = patch_embed.num_patches if i != num_stages - 1 else patch_embed.num_patches + 1
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            text_pos_embed = nn.Parameter(torch.zeros(1, num_text_tokens, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)
            
            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"text_embed{i + 1}", text_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"text_pos_embed{i + 1}", text_pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

            trunc_normal_(pos_embed, std=.02)
            trunc_normal_(text_pos_embed, std=.02)
        
        # define PVLT-TextEmbedding
        bert_config = BertConfig.from_pretrained("bert-base-uncased")
        self.text_embeddings = BertEmbeddings(bert_config)

        # initialize head
        _config = {
            'vocab_size':30522, 'hidden_size':token_hidden_size, 'num_layers':2, 
            'num_heads':12, 'mlp_ratio':4, 'max_text_len':num_text_tokens, 
            'drop_rate':0.1, 'hidden_act':'gelu'
            }
        
        if self.loss_type['mlm'] == 1:
            print('>>> using masked language modeling (mlm) head')
            self.mlm_head_embed = nn.Sequential(
                nn.Linear(in_features=embed_dims[-1], 
                          out_features=token_hidden_size),
                nn.LayerNorm(token_hidden_size)
                )
            self.mlm_head = MLMHead(_config, self.text_embeddings.word_embeddings.weight)
            
        if self.loss_type['itm'] == 1:
            print('>>> using image text matching (itm) head')
            self.itm_head_embed = nn.Sequential(
                nn.Linear(in_features=embed_dims[-1], 
                          out_features=token_hidden_size),
                nn.LayerNorm(token_hidden_size)
                )
            self.itm_head = ITMHead(_config)
        
        if self.loss_type['cls'] == 1:
            print('>>> using sup/sub recognition head')
            self.sup_cls_head_embed = nn.Sequential(
                nn.Linear(in_features=embed_dims[-1], 
                          out_features=token_hidden_size),
                nn.LayerNorm(token_hidden_size)
                )
            self.sup_cls_head = CLSHead(_config, 48)    # 48 super-classes
            self.sub_cls_head_embed = nn.Sequential(
                nn.Linear(in_features=embed_dims[-1], 
                          out_features=token_hidden_size),
                nn.LayerNorm(token_hidden_size)
                )
            self.sub_cls_head = CLSHead(_config, 122)   # 122 sub-classes
        
        if self.loss_type['t2i'] == 1:
            print('>>> using text to image head')
            self.t2i_head = ITGHead(embed_dims=embed_dims, channel=64)  # the channel is an alternatives

        # init weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def generate_dis_data(self, logits, input_ids, mlm_labels, temperature=1.):
        
        b, t = mlm_labels.shape
        """generate labels in the discriminator of ELECTRA"""
        # valid_logits = logits.argmax(dim=-1)[mlm_labels != -1]
        sample_logits = logits[mlm_labels != -1]    # sample the valid logits
        sampled = gumbel_sample(sample_logits, temperature=temperature)
        disc_input_ids = input_ids.clone()
        disc_input_ids[mlm_labels != -1] = sampled.detach()
        # print('>>> Debug-L305: ', input_ids[0])

        # valid_mlm_labels = mlm_labels[mlm_labels != -1].view(b, t)
        offsets = torch.ones_like(mlm_labels)
        offsets[mlm_labels != -1] = -103
        # offsets = offsets + mlm_labels
        input_ids = input_ids + offsets + mlm_labels
        # print('>>> Debug-L310: ', input_ids[0], mlm_labels[0], offsets[0])

        dis_labels = (input_ids != disc_input_ids).detach() # get the different positions

        # print('>>> Debug-L309: ', input_ids[0], disc_input_ids[0], dis_labels[0])
        return disc_input_ids, dis_labels.long()

    def forward_pyramid_features_vl(self, x, y):    # x = vision, y = language
        img_feats, text_feats = [], []

        B = x.shape[0]
        y = self.text_embeddings(y) # torch.Size([5, 128]) -> torch.Size([5, 128, 768])

        for i in range(self.num_stages):    # four stages
            # init functions
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            text_embed = getattr(self, f"text_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            text_pos_embed = getattr(self, f"text_pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")

            # process
            x, (H, W) = patch_embed(x)  # 5, 7744, 64
            y = text_embed(y)   # torch.Size([5, 128, 768]) -> torch.Size([5, 128, 64]) # TODO

            if i == self.num_stages - 1:
                pos_embed = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)
            # print('debug376', i, x.shape, y.shape)
            x = pos_drop(torch.cat((x + pos_embed, y + text_pos_embed), dim=1)) # [bs, 88*88=7744, 64] [bs, 44*44=1936, 128], [bs, 22*22=484, 320], [bs, 11*11=121, 512]
            
            for blk in block:
                x = blk(x, H, W, self.T_num)
            x, y = torch.split(x, [H*W, self.T_num], dim=1)

            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            img_feats.append(x)
            text_feats.append(y)

        return img_feats, text_feats

    def forward(self, input_images, input_ids):
        logits_dict = dict()

        # get image/text features from PVLT
        img_feats, text_feats = self.forward_pyramid_features_vl(input_images, input_ids)

        ########## pretraining tasks ##########
        if self.loss_type['mlm']:
            # define pre-training tasks: mlm
            # TODO: mlm_loss has bugs in the evaluation
            mlm_feat = self.mlm_head_embed(text_feats[-1])
            mlm_logits = self.mlm_head(mlm_feat)    # bs, 128, 30522
            logits_dict.update(mlm_logits=mlm_logits)
        else:
            logits_dict.update(mlm_logits=None)

        if self.loss_type['itm']:
            itm_feat = self.itm_head_embed(text_feats[-1][:, 0:1, :])
            itm_logits = self.itm_head(itm_feat)    # bs, 1, 2
            logits_dict.update(itm_logits=itm_logits)
        else:
            logits_dict.update(itm_logits=None)
        
        if self.loss_type['cls']:
            sup_cls_feat = self.sup_cls_head_embed(text_feats[-1][:, 0:1, :])
            sup_cls_logits = self.sup_cls_head(sup_cls_feat)    # bs, 1, 48
            logits_dict.update(sup_cls_logits=sup_cls_logits)

            sub_cls_feat = self.sub_cls_head_embed(text_feats[-1][:, 0:1, :])
            sub_cls_logits = self.sub_cls_head(sub_cls_feat)    # bs, 1, 122
            logits_dict.update(sub_cls_logits=sub_cls_logits)
        else:
            logits_dict.update(sup_cls_logits=None)
            logits_dict.update(sub_cls_logits=None)
        
        if self.loss_type['t2i']:
            t2i_logits = self.t2i_head(img_feats[1], img_feats[2], img_feats[3])
            logits_dict.update(t2i_logits=t2i_logits)
        else:
            logits_dict.update(t2i_logits=None)

        # TODO: define some downstream tasks

        return logits_dict


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@register_model
def pvlt_tiny(pretrained, token_hidden_size, num_text_tokens, loss_type, pretrained_pth, **kwargs):
    print('>>> current model: PVLT-Tiny')
    model = PyramidVisionLanguageTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        token_hidden_size=token_hidden_size, num_text_tokens=num_text_tokens, loss_type=loss_type,
        **kwargs)
    
    model.default_cfg = _cfg()

    if pretrained_pth:
        model.load_state_dict(torch.load(pretrained_pth), strict=False)
        print('>>> load pretrained weights (backbone part) from:', pretrained_pth)

    return model


@register_model
def pvlt_small(pretrained, token_hidden_size, num_text_tokens, loss_type, pretrained_pth, **kwargs):
    print('>>> Current Model: PVLT-Small')
    model = PyramidVisionLanguageTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], 
        token_hidden_size=token_hidden_size, num_text_tokens=num_text_tokens, loss_type=loss_type,
        **kwargs)
    model.default_cfg = _cfg()

    if pretrained_pth:
        model.load_state_dict(torch.load(pretrained_pth), strict=False)
        print('>>> load pretrained weights (backbone part) from:', pretrained_pth)

    return model


@register_model
def pvlt_medium(pretrained, token_hidden_size, num_text_tokens, loss_type, pretrained_pth, **kwargs):
    print('>>> current model: PVLT-Medium')
    model = PyramidVisionLanguageTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        token_hidden_size=token_hidden_size, num_text_tokens=num_text_tokens, loss_type=loss_type,
        **kwargs)
    
    model.default_cfg = _cfg()

    if pretrained_pth:
        model.load_state_dict(torch.load(pretrained_pth), strict=False)
        print('>>> load pretrained weights (backbone part) from:', pretrained_pth)

    return model


@register_model
def pvlt_large(pretrained, token_hidden_size, num_text_tokens, loss_type, pretrained_pth, **kwargs):
    print('>>> current model: PVLT-Large')
    model = PyramidVisionLanguageTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        token_hidden_size=token_hidden_size, num_text_tokens=num_text_tokens, loss_type=loss_type,
        **kwargs)
    
    model.default_cfg = _cfg()

    if pretrained_pth:
        model.load_state_dict(torch.load(pretrained_pth), strict=False)
        print('>>> load pretrained weights (backbone part) from:', pretrained_pth)

    return model


if __name__ == "__main__":
    inps = torch.randn(5, 3, 352, 352).cuda()
    tokens = text_ids = torch.ones(5, 128).cuda().long()
    # tiny
    model = pvlt_tiny(
        pretrained=True, token_hidden_size=768, num_text_tokens=128, loss_type={'itm':1, 'mlm':0, 'cls':0, 'itg':0, 'i2t':1, 't2i':1, 'rtd':0, 'bartNSG': 0, 'bartMSS':0}, 
        pretrained_pth='/home/admin/workspace/daniel_ji/workspace/alibaba-vilt-daniel/weights/pvt_v1/pvt_tiny.pth').cuda()

    outs = model(inps, tokens, tokens) # (bs, 64, 88, 88) -> (bs, 128, 44, 44) -> (bs, 320, 22, 22) -> (bs, 512, 11, 11)

    pass
