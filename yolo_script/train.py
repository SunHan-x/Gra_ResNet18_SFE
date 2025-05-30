import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLO(model=r'C:\Workspace_yolo\ultralytics\runs\train\train_5.6_v8_None\weights\last.pt')
    model.train(data=r'C:\Workspace_yolo\ultralytics\single_defective_dataset_remove_wrongdata\train.yaml',
                imgsz=640,
                epochs=200,
                batch=64,
                workers=8,
                device='0',
                optimizer='SGD',
                close_mosaic=50,
                resume=True,
                project='runs/train',
                name='train_5.6_v8_None',
                single_cls=True,
                cache=False,
                )
