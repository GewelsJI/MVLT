import torch
import math


def compute_mlm_score(logits, target, index=-1):
    """compute score/accuracy
    support to masked language modeling (mlm)
    @reference: https://github.com/dandelin/ViLT/blob/130eceaca8609cb31688e1d563e022bf43e35897/vilt/gadgets/my_metrics.py#L11
    """
    # init
    correct_num, total_num = 0, 0
    logits, target = logits.detach(), target.detach()
    # print('>>> Debug-L32', logits.shape, target.shape)

    # find the argmax and select the valid value
    preds = logits.argmax(dim=-1)
    preds = preds[target != index]
    target = target[target != index]
    
    # print('preds:\n{}\n'.format(preds), 'target:\n{}'.format(target))
    # when target has no [MASK]
    # if target.numel() == 0:
    #     return 1

    assert preds.shape == target.shape
    # print('>>> Debug-L40', preds, preds.shape, target, target.shape)

    # compute the accuracy
    correct_num += torch.sum(preds == target)
    total_num += target.numel()
    acc = correct_num / total_num

    # print('>>> Debug-L47', acc, correct_num, total_num)
    return acc.item()


def compute_score_with_logits(logits, labels):
    """compute score/accuracy
    support to image-text matching (itm)
    """
    if logits.shape[1] > 1:
        logits = torch.max(logits, 1)[1].data # argmax
        scores = logits == labels
        
    else:
        scores = torch.zeros_like(labels).cuda()
        for i, (logit, label) in enumerate(zip(logits, labels)):
            logit_ = torch.sigmoid(logit)
            if (logit_ >= 0.5 and label == 1) or (logit_ < 0.5 and label == 0):
                scores[i] = 1
    return scores


def compute_psnr(logits, labels):
    """
    @reference: https://github.com/aizvorski/video-quality/blob/master/psnr.py    
    """
    logits, labels = logits.detach(), labels.detach()
    mse = torch.mean((logits - labels) ** 2).item()
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


if __name__ == "__main__":
    # out = torch.randn(4, 128, 30522, requires_grad=True)
    # label = torch.empty((4, 128), dtype=torch.long).random_(30522)
    # scores = compute_mlm_score(out, label)

    # out = torch.randn(4, 1, 2, requires_grad=True)
    # label = torch.empty((4*1), dtype=torch.long).random_(128)
    # compute_masked_language_score(out, label)

    out = torch.randn(1, 3, 352, 352).cuda()
    label = torch.randn(1, 3, 352, 352).cuda()

    scores = compute_psnr(out, label)
    print(scores)