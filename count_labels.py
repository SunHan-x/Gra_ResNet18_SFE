import os
from collections import defaultdict

def count_labels():
    path = '../backup/MultiClass_Dataset/OriginLabels'
    # 用于存储每张图片的类别
    image_class = {}
    all_images = set()
    
    for filename in os.listdir(path):
        if not filename.endswith('.txt') or filename == 'classes.txt':
            continue
            
        is_cplid = filename.startswith('CPLID')
        with open(os.path.join(path, filename), 'r') as f:
            classes_in_image = set()
            has_only_insulator = True
            
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    if class_id != 0:  # 不计数类别0（insulator）
                        has_only_insulator = False
                        if is_cplid and class_id == 1:  # CPLID文件中的类别1改为类别3
                            classes_in_image.add(3)
                        else:
                            classes_in_image.add(class_id)
            
            # 根据优先级确定图片类别
            if classes_in_image:
                # 优先级：missing(3) > flashover(2) > broken(1) > normal(7)
                if 3 in classes_in_image:
                    image_class[filename] = 3  # missing
                elif 2 in classes_in_image:
                    image_class[filename] = 2  # flashover
                elif 1 in classes_in_image:
                    image_class[filename] = 1  # broken
            elif has_only_insulator:
                image_class[filename] = 7  # normal
            
            all_images.add(filename)
    
    # 统计各类别数量
    class_counts = defaultdict(int)
    for class_id in image_class.values():
        class_counts[class_id] += 1
    
    # 打印统计结果
    print("图片统计结果（每张图片只属于一个类别）：")
    class_names = {
        1: "broken",
        2: "flashover",
        3: "missing",
        7: "normal"
    }
    
    total = 0
    for class_id in sorted(class_names.keys()):
        count = class_counts[class_id]
        total += count
        print(f"{class_names[class_id]}: {count}张图片")
    
    print(f"\n总图片数量: {total}")
    
    # 检查是否有图片未被分类
    unclassified = all_images - set(image_class.keys())
    if unclassified:
        print(f"\n警告：有{len(unclassified)}张图片未被分类！")
        print("示例：", list(unclassified)[:5])

if __name__ == '__main__':
    count_labels() 