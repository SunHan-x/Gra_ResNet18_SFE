import os


def modify_class_ids(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)

            new_lines = []
            with open(file_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_id = int(parts[0])

                        if class_id == 0 or class_id == 4:
                            continue
                        elif class_id == 1:
                            parts[0] = '0'
                        elif class_id == 2:
                            parts[0] = '1'
                        elif class_id == 3:
                            parts[0] = '2'
                        else:
                            pass

                        new_line = ' '.join(parts) + '\n'
                        new_lines.append(new_line)

            # 写回文件
            with open(file_path, 'w') as f:
                f.writelines(new_lines)

    print(f"✅ 处理完成！已修改文件夹 {folder_path} 中所有txt文件。")


# 示例调用
folder_path = r"C:\Workspace_yolo\ultralytics\MultiClass_Dataset_remove_dirty\OriginLabels"
modify_class_ids(folder_path)
