# SageMaker Instance Type Optimizer for LLM

This application helps optimize the selection of Amazon SageMaker instances for Large Language Models (LLMs) based on memory requirements for both inference and training tasks.

## Features

- Estimates memory requirements for LLM inference and training
- Suggests suitable SageMaker GPU instances based on calculated memory needs
- Supports various precision types (float32, float16, bfloat16, int8, int4)
- Provides detailed memory breakdowns and formulas used in calculations

## How to Use

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run gpu_memory_calculator_app.py
   ```
4. Input your model parameters in the sidebar
5. View the results in the Inference and Training tabs

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Other dependencies listed in `requirements.txt`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
