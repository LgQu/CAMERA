import torch
import evaluation_models


# flickr
evaluation_models.evalrank_single("pretrain_model/flickr/model_flickr_2.pth.tar", data_path='../data/VSRN_data/', split="test", fold5=False)
# evaluation_models.evalrank_ensemble("pretrain_model/flickr/model_flickr_1.pth.tar", "pretrain_model/flickr/model_flickr_2.pth.tar", \
#                     data_path='../data/VSRN_data/', split="test", fold5=False)

# coco
# evaluation_models.evalrank_single("pretrain_model/coco/model_coco_2.pth.tar", data_path='../data/VSRN_data/', split="testall", fold5=True)
# evaluation_models.evalrank_ensemble("pretrain_model/coco/model_coco_1.pth.tar", "pretrain_model/coco/model_coco_2.pth.tar", \
#                     data_path='../data/VSRN_data/', split="testall", fold5=True)