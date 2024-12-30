# SageMaker Instance Type Optimizer for LLM

This Streamlit application helps optimize the selection of SageMaker instance types for Large Language Models (LLMs) by calculating memory requirements for both inference and training.

## Project Structure

- `main.py`: The main Streamlit application file
- `lang.py`: Contains language-related functions and dictionaries
- `styles.py`: Contains CSS and JavaScript for styling
- `calculations.py`: Contains memory calculation functions
- `utils.py`: Contains utility functions for reading and filtering SageMaker instances
- `sagemaker_gpu_instances.csv`: CSV file containing SageMaker GPU instance information

## Requirements

- Python 3.8 - 3.11
- Dependencies listed in `requirements.txt`

## Setup and Running the App Locally

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run main.py
   ```

3. Open your web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501).

## Deploying on Streamlit.io

1. Fork this repository to your GitHub account.
2. Go to [streamlit.io](https://streamlit.io/) and sign in with your GitHub account.
3. Create a new app and select the forked repository.
4. Set the main file path to `main.py`.
5. Deploy the app.

Note: Streamlit.io uses the `requirements.txt` file to install dependencies, so make sure it's up to date.

## Features

- Calculate memory requirements for LLM inference and training
- Support for various precisions (float32, float16, bfloat16, int8, int4)
- Customizable model parameters
- Visualization of suitable SageMaker GPU instances based on memory requirements
- Detailed memory breakdown and formulas used in calculations

## References

- [Transformer Math 101](https://blog.eleuther.ai/transformer-math/)
- [Calculating GPU memory for serving LLMs](https://www.substratus.ai/blog/calculating-gpu-memory-for-llm)
- [LLM-System-Requirements](https://github.com/manuelescobar-dev/LLM-System-Requirements)
