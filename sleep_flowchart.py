import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io
import warnings

# Filter out the ScriptRunContext warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

def create_flowchart():
    # 创建流程图
    # Set font that supports Chinese characters
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    
    # 定义节点位置
    nodes = {
        'start': (0.5, 0.95),
        'user_info': (0.5, 0.85),
        'upload': (0.5, 0.75),
        'preprocess': (0.5, 0.65),
        'model_load': (0.2, 0.55),
        'inference': (0.5, 0.45),
        'visualization': (0.5, 0.35),
        'report': (0.5, 0.25),
        'end': (0.5, 0.15),
        'model_path': (0.2, 0.65)
    }
    
    # 定义节点形状和颜色
    node_styles = {
        'start': {'shape': 'ellipse', 'color': 'lightgreen'},
        'end': {'shape': 'ellipse', 'color': 'lightgreen'},
        'model_path': {'shape': 'parallelogram', 'color': 'lightyellow'},
        'user_info': {'shape': 'rectangle', 'color': 'lightblue'},
        'upload': {'shape': 'rectangle', 'color': 'lightblue'},
        'preprocess': {'shape': 'rectangle', 'color': 'lightblue'},
        'model_load': {'shape': 'rectangle', 'color': 'lightcoral'},
        'inference': {'shape': 'rectangle', 'color': 'lightcoral'},
        'visualization': {'shape': 'rectangle', 'color': 'lightblue'},
        'report': {'shape': 'rectangle', 'color': 'lightblue'}
    }
    
    # 绘制节点
    for node, pos in nodes.items():
        style = node_styles[node]
        if style['shape'] == 'ellipse':
            ellipse = plt.Circle(pos, 0.05, color=style['color'], fill=True)
            ax.add_patch(ellipse)
        elif style['shape'] == 'rectangle':
            rectangle = plt.Rectangle((pos[0]-0.15, pos[1]-0.04), 0.3, 0.08, 
                                     color=style['color'], fill=True)
            ax.add_patch(rectangle)
        elif style['shape'] == 'parallelogram':
            x, y = pos
            parallelogram_x = [x-0.15, x-0.1, x+0.15, x+0.1]
            parallelogram_y = [y-0.04, y+0.04, y+0.04, y-0.04]
            ax.fill(parallelogram_x, parallelogram_y, style['color'])
    
    # 添加节点文本
    texts = {
        'start': '开始',
        'user_info': '用户信息录入\n(姓名、年龄、性别)',
        'upload': '上传多导睡眠图(PSG)数据',
        'preprocess': '数据预处理',
        'model_load': '加载CrossFusionSleepNet模型',
        'inference': '模型推理\n(自动睡眠分期)',
        'visualization': '睡眠分期结果可视化',
        'report': '生成睡眠分析报告',
        'end': '结束',
        'model_path': '预训练模型参数'
    }
    
    for node, pos in nodes.items():
        ax.text(pos[0], pos[1], texts[node], ha='center', va='center', fontsize=9)
    
    # 绘制连接线和箭头
    arrows = [
        ('start', 'user_info'),
        ('user_info', 'upload'),
        ('upload', 'preprocess'),
        ('preprocess', 'inference'),
        ('model_path', 'model_load'),
        ('model_load', 'inference'),
        ('inference', 'visualization'),
        ('visualization', 'report'),
        ('report', 'end')
    ]
    
    for start, end in arrows:
        start_pos = nodes[start]
        end_pos = nodes[end]
        
        # 特殊处理模型路径到模型加载的箭头
        if start == 'model_path' and end == 'model_load':
            ax.annotate('', xy=(end_pos[0], end_pos[1]), 
                       xytext=(start_pos[0], start_pos[1]),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        # 特殊处理模型加载到推理的箭头
        elif start == 'model_load' and end == 'inference':
            ax.annotate('', xy=(end_pos[0], end_pos[1]), 
                       xytext=(start_pos[0], start_pos[1]),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        else:
            ax.annotate('', xy=(end_pos[0], end_pos[1]-0.04), 
                       xytext=(start_pos[0], start_pos[1]+0.04),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # 添加标题
    ax.text(0.5, 1.0, '基于CrossFusionSleepNet的睡眠分期系统流程图', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 添加图例
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color='lightgreen', label='开始/结束'),
        plt.Rectangle((0, 0), 1, 1, color='lightblue', label='用户交互/数据处理'),
        plt.Rectangle((0, 0), 1, 1, color='lightcoral', label='模型操作'),
        plt.Rectangle((0, 0), 1, 1, color='lightyellow', label='数据源')
    ]
    ax.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, 0.02), ncol=4)
    
    plt.tight_layout()
    return fig

def main():
    st.title("睡眠分期系统流程图")
    
    flowchart = create_flowchart()
    st.pyplot(flowchart)
    
    # 保存流程图为图片
    buf = io.BytesIO()
    flowchart.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    # 提供下载按钮
    st.download_button(
        label="下载流程图",
        data=buf,
        file_name="sleep_system_flowchart.png",
        mime="image/png"
    )

if __name__ == "__main__":
    main()