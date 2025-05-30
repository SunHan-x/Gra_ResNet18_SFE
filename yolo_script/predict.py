from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'C:\Workspace_yolo\ultralytics\runs\train\single_defective_4.23_remove_wrongdata\weights\best.pt')
    model.predict(source=r'C:\Workspace_yolo\ultralytics\single_defective_dataset_remove_wrongdata\test\images',
                  save=True,
                  conf=0.3,
                  )
