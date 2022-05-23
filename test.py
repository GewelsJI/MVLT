import torch

print(torch.__version__)

checkpoint = torch.load('checkpoints/pai_mvlt_exp21/checkpoint_retrieval.pth', map_location='cpu')

"""
like this:
- Image-Text Retrieval (ITR): `>>> retrieval ITR: acc@1: 0.331, acc@5: 0.772, acc@10: 0.911`
- Text-Image Retrieval (TIR): `>>> retrieval TIR: acc@1: 0.346, acc@5: 0.780, acc@10: 0.895`

>>> retrieval ITR: acc@1: 0.335, acc@5: 0.771, acc@10: 0.907
>>> retrieval TIR: acc@1: 0.345, acc@5: 0.777, acc@10: 0.893

# > logging-sup: accuracy (0.9825996064928677) macro_f1 (0.8954719842489123) micro_f1 (0.9825996064928677) weighted_f1 (0.9824654977888717)
# > logging-sub: accuracy (0.9356554353172651) macro_f1 (0.8285927576055913) micro_f1 (0.9356554353172651) weighted_f1 (0.9351514388782373)

> logging-sup: accuracy (0.9824766355140186) macro_f1 (0.8952777462208645) micro_f1 (0.9824766355140185) weighted_f1 (0.9823404044558934)
> logging-sub: accuracy (0.9357169208066897) macro_f1 (0.8286230480992728) micro_f1 (0.9357169208066897) weighted_f1 (0.9352230226818611)
"""