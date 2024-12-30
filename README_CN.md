# SageMaker LLM 实例类型优化器

该 Streamlit 应用程序帮助通过计算推理和训练的内存需求，优化大型语言模型（LLMs）的 SageMaker 实例类型选择。

## 项目结构

- `main.py`: 主要的 Streamlit 应用程序文件
- `lang.py`: 包含语言相关的函数和字典
- `styles.py`: 包含用于样式设置的 CSS 和 JavaScript
- `calculations.py`: 包含内存计算函数
- `utils.py`: 包含用于读取和过滤 SageMaker 实例的实用函数
- `sagemaker_gpu_instances.csv`: 包含 SageMaker GPU 实例信息的 CSV 文件

## 设置和运行应用程序

1. 安装所需依赖：
   ```
   pip install -r requirements.txt
   ```

2. 运行 Streamlit 应用：
   ```
   streamlit run main.py
   ```

3. 在网络浏览器中打开 Streamlit 提供的 URL（通常是 http://localhost:8501）。

## 功能

- 计算 LLM 推理和训练的内存需求
- 支持多种精度（float32、float16、bfloat16、int8、int4）
- 可自定义模型参数
- 根据内存需求可视化合适的 SageMaker GPU 实例
- 详细的内存分解和计算中使用的公式

## 参考资料

- [Transformer Math 101](https://blog.eleuther.ai/transformer-math/)
- [计算用于服务 LLMs 的 GPU 内存](https://www.substratus.ai/blog/calculating-gpu-memory-for-llm)
- [LLM-System-Requirements](https://github.com/manuelescobar-dev/LLM-System-Requirements)
