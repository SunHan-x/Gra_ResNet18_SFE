# 验证
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'C:\Workspace_yolo\ultralytics\runs\train\train_5.6_v8_None\weights\best.pt')
    model.val(data=r'C:\Workspace_yolo\ultralytics\single_defective_dataset_remove_wrongdata\train.yaml',
              split='val',
              imgsz=640,
              batch=32,
              iou=0.5,
              rect=False,
              save_json=False,
              project='runs/val',
              name='val_5_7_None_v8',
              )