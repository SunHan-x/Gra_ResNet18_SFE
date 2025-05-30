import os
import shutil

def copy_images(source_folder, destination_folder, start_num=1, end_num=600):
    """
    将CPLID{start_num}.jpg到CPLID{end_num}.jpg的文件复制到目标文件夹
    
    参数:
        source_folder: 源文件夹路径
        destination_folder: 目标文件夹路径
        start_num: 起始编号(默认601)
        end_num: 结束编号(默认1448)
    """
    # 确保目标文件夹存在
    os.makedirs(destination_folder, exist_ok=True)
    
    copied_count = 0
    
    for i in range(start_num, end_num + 1):
        filename = f"CPLID{i}.jpg"
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, destination_path)
            copied_count += 1
            print(f"已复制: {filename}")
        else:
            print(f"未找到: {filename}")
    
    print(f"\n复制完成! 共复制了{copied_count}个文件。")

# 使用示例
if __name__ == "__main__":
    source_dir = input("请输入源文件夹路径: ")
    dest_dir = input("请输入目标文件夹路径: ")
    
    copy_images(source_dir, dest_dir)