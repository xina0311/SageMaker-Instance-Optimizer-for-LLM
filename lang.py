import streamlit as st

def set_lang():
    if 'language' not in st.session_state:
        st.session_state.language = 'en'

    languages = {
        'en': {
            'lang_selector': 'Language',
            'sidebar_title': 'Model Parameters',
            'model': 'Model Name',
            'params': 'Number of parameters (in billions)',
            'precision': 'Precision',
            'optional_params': 'Optional Parameters for Estimating Training Memory',
            'batch_size': 'Batch Size',
            'seq_len': 'Sequence Length',
            'hidden_size': 'Hidden Size',
            'num_layers': 'Number of Layers',
            'num_heads': 'Number of Attention Heads',
            'optimizer': 'Optimizer',
            'total_inference': 'Total Inference Memory (with 20% overhead)',
            'total_training': 'Total Training Memory',
            'calc_details': 'Calculation details:',
            'model_mem': 'Model Memory',
            'opt_mem': 'Optimizer Memory',
            'grad_mem': 'Gradient Memory',
            'act_mem': 'Activation Memory',
            'total_mem': 'Total memory',
            'suitable_instances': 'Suitable SageMaker GPU Instances:',
            'formulas': 'Formulas',
            'inf_mem': 'Inference Memory',
            'train_mem': 'Training Memory',
            'bytes_per_param': 'Bytes per parameter:',
            'where': 'where',
            'notes': 'Notes',
            'note1': 'The calculations are based on theoretical estimations and may not reflect exact real-world usage.',
            'note2': 'Actual memory usage can vary based on implementation details and optimizations.',
            'note3': 'For training, we assume a simple optimizer like Adam. More complex optimizers may require additional memory.',
            'note4': 'We do not account for memory needed for dataset loading or preprocessing.',
            'note5': 'The 20% overhead for inference is an estimation and may vary based on the specific use case.',
            'note6': 'For more accurate results, consider profiling your model on the target hardware.',
            'note7': 'Always test your model on the chosen instance type to ensure compatibility and performance.',
            'reference': 'References',
            'model_desc': 'Enter the name of your model',
            'params_desc': 'Enter the number of parameters in billions (e.g., 70 for a 70B parameter model)',
            'precision_desc': 'Select the precision used for the model weights',
            'batch_size_desc': 'Enter the batch size for training',
            'seq_len_desc': 'Enter the sequence length',
            'hidden_size_desc': 'Enter the hidden size of the model',
            'num_layers_desc': 'Enter the number of layers in the model',
            'num_heads_desc': 'Enter the number of attention heads',
        },
        'zh': {
            'lang_selector': '语言',
            'sidebar_title': '模型参数',
            'model': '模型名称',
            'params': '参数数量（十亿）',
            'precision': '精度',
            'optional_params': '估算训练内存的可选参数',
            'batch_size': '批量大小',
            'seq_len': '序列长度',
            'hidden_size': '隐藏层大小',
            'num_layers': '层数',
            'num_heads': '注意力头数',
            'optimizer': '优化器',
            'total_inference': '总推理内存（含20%开销）',
            'total_training': '总训练内存',
            'calc_details': '计算详情：',
            'model_mem': '模型内存',
            'opt_mem': '优化器内存',
            'grad_mem': '梯度内存',
            'act_mem': '激活内存',
            'total_mem': '总内存',
            'suitable_instances': '适合的SageMaker GPU实例：',
            'formulas': '公式',
            'inf_mem': '推理内存',
            'train_mem': '训练内存',
            'bytes_per_param': '每个参数的字节数：',
            'where': '其中',
            'notes': '注意事项',
            'note1': '这些计算基于理论估算，可能与实际使用情况有所不同。',
            'note2': '实际内存使用可能因实现细节和优化而有所不同。',
            'note3': '对于训练，我们假设使用简单的优化器如Adam。更复杂的优化器可能需要额外的内存。',
            'note4': '我们没有考虑数据集加载或预处理所需的内存。',
            'note5': '推理的20%开销是一个估计值，可能因具体用例而有所不同。',
            'note6': '为获得更准确的结果，请考虑在目标硬件上分析您的模型。',
            'note7': '始终在选定的实例类型上测试您的模型，以确保兼容性和性能。',
            'reference': '参考资料',
            'model_desc': '输入您的模型名称',
            'params_desc': '输入参数数量（十亿为单位，例如70B参数模型输入70）',
            'precision_desc': '选择模型权重使用的精度',
            'batch_size_desc': '输入训练的批量大小',
            'seq_len_desc': '输入序列长度',
            'hidden_size_desc': '输入模型的隐藏层大小',
            'num_layers_desc': '输入模型的层数',
            'num_heads_desc': '输入注意力头数',
        }
    }

    return languages[st.session_state.language]

def change_lang():
    if st.session_state.language == 'en':
        st.session_state.language = 'zh'
    else:
        st.session_state.language = 'en'
