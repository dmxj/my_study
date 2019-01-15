#encoding:utf-8
"""
mask rcnn模型评估
"""
import numpy as np
import mxnet as mx
from tqdm import tqdm
from mxnet import nd
from mxnet import gluon
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv.model_zoo import get_model
from gluoncv.data import batchify
from gluoncv.data.transforms.presets.rcnn import MaskRCNNDefaultValTransform
from gluoncv.utils.metrics.coco_instance import COCOInstanceMetric

from gluoncv.utils.metrics.segmentation import SegmentationMetric

def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    num_ctx = len(ctx_list)
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        new_batch.append(new_data)
    return new_batch

def validate(net, val_data, ctx, eval_metric, size):
    """Test on validation dataset."""
    clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    net.hybridize(static_alloc=True)
    with tqdm(total=size) as pbar:
        for ib, batch in enumerate(val_data):
            batch = split_and_load(batch, ctx_list=ctx)
            det_bboxes = []
            det_ids = []
            det_scores = []
            det_masks = []
            det_infos = []
            for x, im_info in zip(*batch):
                # get prediction results
                ids, scores, bboxes, masks = net(x)
                det_bboxes.append(clipper(bboxes, x))
                det_ids.append(ids)
                det_scores.append(scores)
                det_masks.append(masks)
                det_infos.append(im_info)
            # update metric
            for det_bbox, det_id, det_score, det_mask, det_info in zip(det_bboxes, det_ids, det_scores, det_masks, det_infos):
                for i in range(det_info.shape[0]):
                    # numpy everything
                    det_bbox = det_bbox[i].asnumpy()
                    det_id = det_id[i].asnumpy()
                    det_score = det_score[i].asnumpy()
                    det_mask = det_mask[i].asnumpy()
                    det_info = det_info[i].asnumpy()
                    # filter by conf threshold
                    im_height, im_width, im_scale = det_info
                    valid = np.where(((det_id >= 0) & (det_score >= 0.001)))[0]
                    det_id = det_id[valid]
                    det_score = det_score[valid]
                    det_bbox = det_bbox[valid] / im_scale
                    det_mask = det_mask[valid]
                    # fill full mask
                    im_height, im_width = int(round(im_height / im_scale)), int(round(im_width / im_scale))
                    full_masks = []
                    for bbox, mask in zip(det_bbox, det_mask):
                        full_masks.append(gcv.data.transforms.mask.fill(mask, bbox, (im_width, im_height)))
                    full_masks = np.array(full_masks)
                    eval_metric.update(det_bbox, det_id, det_score, full_masks)
            pbar.update(len(ctx))
    return eval_metric.get()




# init params
ctx = mx.cpu()
model_path = "/Users/rensike/.mxnet/models/mask_rcnn_resnet50_v1b_coco-a3527fdc.params"
num_workers = 0

# init model
net = get_model("mask_rcnn_resnet50_v1b_coco",pretrained=False,pretrained_base=False)
net.load_parameters(model_path)
net.collect_params().reset_ctx(ctx)

# load val dataset
val_dataset = gdata.COCOInstance(splits='instances_val',root="/Users/rensike/Files/temp/coco_mini", skip_empty=False)
# val_dataset = gdata.VOCSegmentation(root="/Users/rensike/Files/temp/voc_mini",split="val")
# eval_metric = SegmentationMetric(nclass=val_dataset.num_class)
eval_metric = COCOInstanceMetric(val_dataset,"coco_eval")

# load val dataloader
val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(2)])
val_data_loader = mx.gluon.data.DataLoader(
    val_dataset.transform(MaskRCNNDefaultValTransform(net.short, net.max_size)),
    1, False, batchify_fn=val_bfn, last_batch='keep', num_workers=num_workers)

# do evaluate
eval_metric.reset()
net.hybridize(static_alloc=True)

names, values = validate(net, val_data_loader, [ctx], eval_metric,len(val_dataset))
for k, v in zip(names, values):
    print(k, v)

