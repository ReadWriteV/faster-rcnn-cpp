from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

root_path = '/mnt/889cdd89-1094-48ae-b221-146ffe543605/wr/faster-rcnn-cpp'

anno_file = root_path + '/data/voc2007test.easy.json'
res_file = root_path + '/result.json'

coco_gt = COCO(anno_file)
coco_dt = coco_gt.loadRes(res_file)

cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

print(cocoEval.stats)
