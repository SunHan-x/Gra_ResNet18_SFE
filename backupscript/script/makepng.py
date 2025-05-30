import matplotlib.pyplot as plt

# 数据
categories = ['broken', 'flashover', 'missing']
values = [0.9686, 0.9456, 0.9526]
colors = ['#4CAF50', '#FF9800', '#2196F3']  # 绿色、橙色、蓝色

# 创建柱状图
plt.figure(figsize=(10, 6))
bars = plt.bar(categories, values, color=colors)

# 添加数值标签
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval*100:.1f}%', ha='center', va='bottom')

# 设置标题和轴标签
plt.title('Per-Class Top-1 Accuracy (Patch-Level Retrieval)')
plt.ylabel('Top-1 Accuracy')
plt.ylim(0, 1)  # 设置y轴范围为0到1

# 显示图形
plt.show()