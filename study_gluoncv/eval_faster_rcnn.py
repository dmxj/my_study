#encoding:utf-8
'''
对预训练的faster-rcnn模型进行评估
'''
import mxnet as mx
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv.model_zoo import get_model
from gluoncv.data import batchify
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOCMApMetric
from gluoncv.utils.metrics.accuracy import Accuracy

def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    num_ctx = len(ctx_list)
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        new_batch.append(new_data)
    return new_batch

def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    net.hybridize(static_alloc=True)
    for batch in val_data:
        batch = split_and_load(batch, ctx_list=[ctx])
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y, im_scale in zip(*batch):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(clipper(bboxes, x))
            # rescale to original resolution
            im_scale = im_scale.reshape((-1)).asscalar()
            det_bboxes[-1] *= im_scale
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_bboxes[-1] *= im_scale
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in zip(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults):
            eval_metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff)
    return eval_metric.get()

# init params
ctx = mx.cpu()
model_path = "/Users/rensike/.mxnet/models/faster_rcnn_resnet50_v1b_voc-447328d8.params"
batch_size = 32
num_workers = 0

# init model
net = get_model("faster_rcnn_resnet50_v1b_voc",pretrained=False,pretrained_base=False)
net.load_parameters(model_path)
net.collect_params().reset_ctx(ctx)

# load val dataset
val_dataset = gdata.VOCDetection(root="/Users/rensike/Files/temp/voc_mini",
            splits=[(0, 'val')])
eval_metric = VOCMApMetric(iou_thresh=0.5, class_names=val_dataset.classes)

# load val dataloader
val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(3)])
val_data_loader = mx.gluon.data.DataLoader(
    val_dataset.transform(FasterRCNNDefaultValTransform(net.short, net.max_size)),
    1, False, batchify_fn=val_bfn, last_batch='keep', num_workers=num_workers)

# do evaluate
eval_metric.reset()
net.hybridize(static_alloc=True)

map_name, mean_ap = validate(net, val_data_loader, ctx, eval_metric)
val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
print('Validation: \n{}'.format(val_msg))
current_map = float(mean_ap[-1])
print("current_map:",current_map)
