from graphviz import Digraph
import os

def create_network_architecture():
    # 创建有向图
    dot = Digraph(comment='SFDH-FGIR Network Architecture')
    dot.attr(rankdir='TB')
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue')
    
    # 设置节点样式
    dot.attr('node', shape='box', style='filled')
    
    # 添加输入节点
    dot.node('input', '输入图像\n224×224×3', fillcolor='lightgreen')
    
    # 骨干网络部分
    dot.node('backbone', 'ResNet18 骨干网络\n(预训练)', fillcolor='lightblue')
    
    # 特征提取模块
    dot.node('feature_extraction', '显著性特征提取模块\n(注意力机制)', fillcolor='lightyellow')
    
    # 特征融合部分
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='特征融合')
        c.node('global_pool', '全局平均池化', fillcolor='lightpink')
        c.node('local_feature', '局部显著性特征', fillcolor='lightpink')
        c.node('concat', '特征拼接', fillcolor='lightpink')
    
    # 哈希编码模块
    dot.node('hash_layer', '哈希编码层\n(Tanh激活)', fillcolor='lightblue')
    
    # 分类器模块
    dot.node('classifier', '分类器层', fillcolor='lightgreen')
    
    # 损失函数部分
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='损失函数')
        c.node('ce_loss', '交叉熵损失', fillcolor='lightyellow')
        c.node('hash_loss', '哈希正则项', fillcolor='lightyellow')
        c.node('center_loss', '中心损失', fillcolor='lightyellow')
    
    # 添加连接
    dot.edge('input', 'backbone')
    dot.edge('backbone', 'feature_extraction')
    dot.edge('feature_extraction', 'global_pool')
    dot.edge('feature_extraction', 'local_feature')
    dot.edge('global_pool', 'concat')
    dot.edge('local_feature', 'concat')
    dot.edge('concat', 'hash_layer')
    dot.edge('hash_layer', 'classifier')
    
    # 损失函数连接
    dot.edge('classifier', 'ce_loss')
    dot.edge('hash_layer', 'hash_loss')
    dot.edge('hash_layer', 'center_loss')
    
    # 保存图片
    dot.render('network_architecture', format='png', cleanup=True)
    print("架构图已生成: network_architecture.png")

if __name__ == '__main__':
    create_network_architecture() 