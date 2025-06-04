# 验证
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'runs\train\YOLO11_CAA\weights\best.pt')
    model.val(data=r"C:\Workspace_yolo\新建文件夹\Origin_Dataset\train.yaml",
              split='val',
              imgsz=640,
              batch=32,
              iou=0.5,
              rect=False,
              save_json=False,
              project='runs/val',
              name='CAA',
              )