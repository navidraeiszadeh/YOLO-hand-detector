import sys
sys.path.append("D:\yolo-hand-detection-master\yolo-hand-detection-master\models\convertor_YOLO")
from yolov5.models.yolo import Model

from yolov5.utils.torch_utils import load_darknet_weights

model = Model("cross-hands-tiny.cfg")
load_darknet_weights(model, "cross-hands-tiny.weights")
model.save("cross-hands-tiny.pt")
print("YOLO Darknet model converted to PyTorch format successfully")
