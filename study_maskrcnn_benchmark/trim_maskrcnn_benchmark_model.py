import torch
import argparse
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model

parser = argparse.ArgumentParser(description="Trim Maskrcnn_benchmark weights and save in PyTorch format.")
parser.add_argument(
    "--pretrained_path",
    default="/Users/rensike/Resources/models/pytorch/e2e_faster_rcnn_R_50_FPN_1x.pth",
    help="path to maskrcnn_benchmark pretrained weight(.pth)",
    type=str,
)
parser.add_argument(
    "--save_path",
    default="/Users/rensike/Resources/models/pytorch/e2e_faster_rcnn_R_50_FPN_1x_no_roi_head.pth",
    help="path to save the converted model",
    type=str,
)
parser.add_argument(
    "--cfg",
    default="configs/e2e_faster_rcnn_R_50_FPN_1x_caffe2.yaml",
    help="path to config file",
    type=str,
)
args = parser.parse_args()

opts = [
    "MODEL.DEVICE","cpu",
    "DATALOADER.NUM_WORKERS",0,
    "MODEL.ROI_BOX_HEAD.NUM_CLASSES",21
]

cfg.merge_from_file(args.cfg)
cfg.merge_from_list(opts)
cfg.freeze()

model = build_detection_model(cfg)
device = torch.device(cfg.MODEL.DEVICE)
model.to(device)

map_location = "cpu" if cfg.MODEL.DEVICE == "cpu" else None
pretrained_model = torch.load(args.pretrained_path,map_location=map_location)['model']
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict and "roi_heads" not in k}
model_dict.update(pretrained_dict)

torch.save(model_dict,args.save_path)
