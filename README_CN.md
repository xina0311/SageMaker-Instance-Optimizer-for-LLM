# SageMaker LLM 实例类型优化器

该应用程序帮助根据推理和训练任务的内存需求优化 Amazon SageMaker 实例的选择，专为大型语言模型（LLMs）设计。

## 功能

- 估算 LLM 推理和训练的内存需求
- 根据计算的内存需求推荐合适的 SageMaker GPU 实例
- 支持多种精度类型（float32、float16、bfloat16、int8、int4）
- 提供详细的内存分解和计算中使用的公式

## 使用方法

1. 克隆仓库
2. 安装所需依赖：
   ```
   pip install -r requirements.txt
   ```
3. 运行 Streamlit 应用：
   ```
   streamlit run gpu_memory_calculator_app.py
   ```
4. 在侧边栏输入您的模型参数
5. 在推理和训练标签页查看结果

## 要求

- Python 3.7+
- Streamlit
- Pandas
- 其他依赖列在 `requirements.txt` 中

## 贡献

欢迎贡献！请随时提交 Pull Request。

## 许可证

本项目是开源的，遵循 [MIT 许可证](LICENSE)。
