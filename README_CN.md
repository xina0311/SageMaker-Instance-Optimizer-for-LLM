# SageMaker LLM 实例类型优化器

该 Streamlit 应用程序通过计算推理和训练的内存需求，帮助优化大型语言模型（LLMs）的 SageMaker 实例类型选择。

## 项目结构

- `main.py`: 主要的 Streamlit 应用程序文件
- `lang.py`: 包含语言相关的函数和字典
- `styles.py`: 包含用于样式设置的 CSS 和 JavaScript
- `calculations.py`: 包含内存计算函数
- `utils.py`: 包含用于读取和过滤 SageMaker 实例的实用函数
- `sagemaker_gpu_instances.csv`: 包含 SageMaker GPU 实例信息的 CSV 文件

## 要求

- Python 3.7 - 3.11（注意：由于某些依赖项的兼容性问题，不支持 Python 3.12）
- 依赖项列在 `requirements.txt` 中

## 本地设置和运行应用程序

1. 安装所需依赖：
   ```
   pip install -r requirements.txt
   ```

2. 运行 Streamlit 应用：
   ```
   streamlit run main.py
   ```

3. 在网络浏览器中打开 Streamlit 提供的 URL（通常是 http://localhost:8501）。

## 在 Streamlit.io 上部署

1. 将此仓库 fork 到您的 GitHub 账户。
2. 前往 [streamlit.io](https://streamlit.io/) 并使用您的 GitHub 账户登录。
3. 创建一个新的应用并选择您 fork 的仓库。
4. 将主文件路径设置为 `main.py`。
5. 部署应用。

注意：Streamlit.io 使用 `requirements.txt` 文件安装依赖项。如果在部署过程中遇到任何问题，您可能需要调整 requirements.txt 文件中的版本，以确保与 Streamlit.io 环境兼容。

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
