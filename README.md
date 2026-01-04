# ZoeyTool Plugin​​
# 进入 ComfyUI 自定义插件目录 | Navigate to ComfyUI custom_nodes directory
cd ComfyUI/custom_nodes

# 克隆仓库 | Clone repository
git clone https://github.com/liangzoey/comfyui-ZoeyTool.git

# 安装依赖 | Install dependencies
cd comfyui-ZoeyTool
pip install -r requirements.txt



以下是基于您提供的 ​​Zoey Tool​​ 功能截图整理的 ​​完整使用指南​​，涵盖各模块核心功能与操作说明：
￼
<img width="2202" height="1020" alt="image" src="https://github.com/user-attachments/assets/a0ba4aac-8a7a-4976-8c74-2ca7f5fae734" />


🖼️ ​​一、图像处理工具​​
​​1. 多文件批量处理器​​<img width="1599" height="1083" alt="56beb106be48d8e11a4b6e982c000c0b" src="https://github.com/user-attachments/assets/50dea99e-4591-4d2c-a43f-73f9cee717fd" />

https://via.placeholder.com/400x200/333/FFF?text=Zoey-多文件批量处理器
​​功能​​：批量重命名/转换文件格式
​​参数配置​​：
yaml
￼
复制

input_folder: 输入目录路径  
output_folder: 输出目录路径  
name_prefix: FILE  # 文件名前缀  
file_types: jpg,png,txt,pdf,doc  # 支持格式  
start_index: 1  # 起始序号  
sort_method: filename  # 按文件名排序
￼
￼
​​输出​​：
•
file_list：生成文件路径列表 → 连接至预览模块


​​2. 多功能图像编辑器​​<img width="2031" height="1117" alt="533dd1c4b717b6cdefd4ebe2f33f1ed5" src="https://github.com/user-attachments/assets/1a70602e-29e1-41be-af69-779eff767f18" />

https://via.placeholder.com/400x200/333/FFF?text=多功能图像编辑器
​​操作类型​​：
•
水平翻转、锐化（强度1.5）、模糊（半径5.0）
•
分割图像（横向/纵向各2块）
​​参数示例​​：
yaml
￼
复制

模糊半径: 5.0  
锐化强度: 1.5  
融合权重: 0.50  
阈值: 128  # 二值化阈值
￼
￼


​​3. 遮罩边界框绘制​​<img width="3140" height="1224" alt="594ee600d6e5042a26317acc2f7ca0ae" src="https://github.com/user-attachments/assets/e3f8af5e-0051-4f43-a363-7dd27b195cc0" />

https://via.placeholder.com/400x200/333/FFF?text=遮罩边界框绘制
​​输入​​：图像 + 遮罩
​​绘制参数​​：
yaml
￼
复制

线宽: 10  
颜色: 红色（#FF0000）  
边距百分比: 5%  # 边界框内边距  
填充颜色: 透明（不透明度0）
￼
￼
​​输出​​：带边界框的图像（尺寸不变）
￼


🎥 ​​二、视频处理工具<img width="2686" height="750" alt="c3084c330435deac0cac0d2f5cc204bf" src="https://github.com/user-attachments/assets/255ecb66-409d-4ea0-99f7-d0f9a72125cf" />

​​1. 批量视频加载器​​
https://via.placeholder.com/400x200/333/FFF?text=批量视频加载器
​​配置​​：
yaml
￼
复制

源目录: D:/Videos/Input  
file_pattern: *.mp4;*.avi;*.mov  # 文件类型  
limit: 20  # 最大加载数
￼
￼
​​2. 视频批处理器​​
https://via.placeholder.com/400x200/333/FFF?text=视频批处理器
​​处理选项​​：
yaml
￼
复制

frame_rate: 30  # 输出帧率  
output_format: mp4  
device: auto  # 自动选择GPU/CPU  
skip_existing: 开启  # 跳过已处理文件
￼
￼
​​3. 智能视频存储器​​
https://via.placeholder.com/400x200/333/FFF?text=智能视频存储器
​​存储规则​​：
yaml
￼
复制

output_directory: D:/视频输出  
prefix: batch  # 文件名前缀  
date_folder: 启用  # 按日期分类存储  
start_index: 1  # 起始编号
￼
￼
￼


🌐 ​​三、辅助工具​​


​​1. 纯净翻译器​​<img width="2535" height="1137" alt="dc4e6a1eaa54692dd652478e9bf1e27e" src="https://github.com/user-attachments/assets/1dd23b6b-4700-40c6-b10f-b34c082f0f70" />

https://via.placeholder.com/400x200/333/FFF?text=纯净翻译器
​​配置​​：
yaml
￼
复制

source_lang: auto  # 自动检测源语言  
target_lang: 英文  
engine: 内置模型  # 或百度API/谷歌API  
输入: "你好" → 输出: "Hello."
￼
￼


​​2. 图像编辑提示词生成器​​<img width="2296" height="975" alt="adad1ed13ca0ce85459c4b1cc1f53557" src="https://github.com/user-attachments/assets/d21bdafb-1184-4fb6-833c-1d5032ae4f24" />

https://via.placeholder.com/400x200/333/FFF?text=提示词生成器
​​用例​​：
yaml
￼
复制

编辑类型: 对象编辑  
目标元素: 头巾 → 新值: 帽子  
视角方向: 正面  
自定义提示词: "把头巾替换为帽子"
￼
￼
​​
3. 图像批量裁剪器​​<img width="1317" height="1238" alt="df58da40f83a28039aeb86fc24c24ee5" src="https://github.com/user-attachments/assets/8f7d4d72-2550-4c2f-a616-11398999e86f" />

https://via.placeholder.com/400x200/333/FFF?text=图像批量裁剪器
​​裁剪参数​​：
yaml
￼
复制

left_crop: 0   # 左裁剪像素  
right_crop: 0  
top_crop: 0  
bottom_crop: 0  
preserve_names: true  # 保留原文件名  
prefix: cropped_  # 输出文件名前缀

<img width="851" height="765" alt="f2d98404744a2a339b6502d1c96e0cd9" src="https://github.com/user-attachments/assets/cd6e07ee-13b0-4c04-bffe-bc30c3b9ebe8" />





