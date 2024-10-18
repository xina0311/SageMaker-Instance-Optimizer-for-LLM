import streamlit as st
import math
import pandas as pd

def set_lang():
    return {
        "sidebar_title": "Model Parameters",
        "model": "Model",
        "params": "Number of parameters (in billions)",
        "precision": "Precision",
        "batch_size": "Batch Size",
        "seq_len": "Sequence Length",
        "hidden_size": "Hidden Size",
        "num_layers": "Number of Layers",
        "num_heads": "Number of Attention Heads",
        "optional_params": "Optional Parameters",
        "total_inference": "Estimated Total Inference Memory",
        "total_training": "Estimated Total Training Memory",
        "calc_details": "Calculation details:",
        "model_mem": "Model Memory",
        "act_mem": "Activation Memory",
        "total_mem": "Total Memory (with 20% overhead)",
        "opt_mem": "Optimizer Memory",
        "grad_mem": "Gradient Memory",
        "formulas": "Formulas Used",
        "inf_mem": "Inference Memory",
        "train_mem": "Training Memory",
        "where": "Where:",
        "bytes_per_param": "Where bytes per param:",
        "optimizer": "Optimizer (for training)",
        "note": "Note: Hidden size, number of layers, and number of attention heads are estimated based on the model size, but can be adjusted manually.",
        "notes": "Notes:",
        "note1": "This is a simplified estimation and may not account for all factors affecting GPU memory usage.",
        "note2": "Actual memory usage may vary depending on the specific model architecture and implementation.",
        "note3": "For training, you may need additional memory for optimizer states and gradients.",
        "note4": "The activation memory calculation uses a selective recomputation approach for training, which is an approximation.",
        "note5": "Int4 and Int8 precisions are approximations and may not be supported by all hardware or frameworks.",
        "note6": "For int4 and int8, we assume the same gradient memory as mixed precision for training, which might not be accurate for all implementations.",
        "note7": "The inference memory calculation is a rough estimate and may not be as accurate for very large models or extreme batch sizes.",
        "reference": "Reference:",
        "suitable_instances": "Suitable SageMaker GPU Instances:",
    }

st.set_page_config(layout="wide")

# Custom CSS and JavaScript for resizable sidebar and styling
st.markdown("""
<style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 500px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500px;
        margin-left: -500px;
    }
    .resize-handle {
        position: absolute;
        top: 0;
        right: -5px;
        width: 10px;
        height: 100%;
        cursor: col-resize;
        z-index: 1000;
    }
    .memory-total {
        font-size: 24px !important;
        color: red !important;
    }
    .main-title {
        font-size: 36px !important;
        font-weight: bold !important;
        margin-bottom: 20px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 24px !important;
        font-weight: bold !important;
    }
</style>
<script>
    const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
    const resizeHandle = document.createElement('div');
    resizeHandle.className = 'resize-handle';
    sidebar.appendChild(resizeHandle);

    let isResizing = false;
    let lastDownX = 0;

    resizeHandle.addEventListener('mousedown', (e) => {
        isResizing = true;
        lastDownX = e.clientX;
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        const width = sidebar.offsetWidth - (lastDownX - e.clientX);
        sidebar.style.width = width + 'px';
        lastDownX = e.clientX;
    });

    document.addEventListener('mouseup', () => {
        isResizing = false;
    });
</script>
""", unsafe_allow_html=True)

def read_sagemaker_instances():
    df = pd.read_csv('sagemaker_gpu_instances.csv')
    return df

def calculate_inference_memory(model_params_billions, precision, batch_size, sequence_length):
    model_params = model_params_billions * 1e9
    
    # Model memory
    if precision == 'int4':
        bytes_per_param = 0.5
    elif precision == 'int8':
        bytes_per_param = 1
    elif precision in ['float16', 'bfloat16']:
        bytes_per_param = 2
    elif precision == 'float32':
        bytes_per_param = 4
    else:
        raise ValueError("Unsupported precision")
    
    model_memory = model_params * bytes_per_param
    
    # Estimate hidden size
    hidden_size = 128 * math.ceil(math.sqrt(model_params / 1e8))
    
    # Activation memory
    activation_memory = sequence_length * batch_size * hidden_size * 2  # 2 bytes per activation
    
    total_memory = (model_memory + activation_memory) * 1.2  # Add 20% overhead
    
    return model_memory / 1e9, activation_memory / 1e9, total_memory / 1e9

def calculate_training_memory(model_params_billions, precision, batch_size, sequence_length, optimizer, hidden_size, num_layers, num_attention_heads):
    model_params = model_params_billions * 1e9
    
    # Model memory
    if precision == 'float32':
        model_memory = model_params * 4
    elif precision == 'int4':
        model_memory = model_params * 0.5
    else:  # mixed precision (fp16/bf16 and fp32) or int8
        model_memory = model_params * 2
    
    # Optimizer memory
    if optimizer == 'AdamW':
        optimizer_memory = model_params * 12
    elif optimizer == '8-bit AdamW':
        optimizer_memory = model_params * 6
    elif optimizer == 'SGD with momentum':
        optimizer_memory = model_params * 8
    
    # Gradient memory
    gradient_memory = model_params * (4 if precision == 'float32' else 2)
    
    # Activation memory (using selective recomputation formula)
    activation_memory = 2 * batch_size * sequence_length * hidden_size * num_layers
    
    total_memory = model_memory + optimizer_memory + gradient_memory + activation_memory
    return model_memory / 1e9, optimizer_memory / 1e9, gradient_memory / 1e9, activation_memory / 1e9, total_memory / 1e9

def filter_instances(df, required_memory):
    filtered_df = df[df['Total GPU Memory (GB)'] >= required_memory].sort_values('Total GPU Memory (GB)')
    return filtered_df.reset_index(drop=True)

t = set_lang()

st.markdown('<h1 class="main-title">SageMaker Instance Type Optimizer for LLM</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title(t["sidebar_title"])

# Required parameters
model = st.sidebar.text_input(t["model"], value='Llama-3.1-70B-Instruct')
model_params = st.sidebar.text_input(t["params"], value='70.0')
precision = st.sidebar.selectbox(t["precision"], ['float16', 'float32', 'bfloat16', 'int8', 'int4'])
batch_size = st.sidebar.text_input(t["batch_size"], value='2')
sequence_length = st.sidebar.text_input(t["seq_len"], value='4096')

# Optional parameters
st.sidebar.markdown(f"### {t['optional_params']}")
hidden_size = st.sidebar.text_input(t["hidden_size"], value='8192')
num_layers = st.sidebar.text_input(t["num_layers"], value='80')
num_attention_heads = st.sidebar.text_input(t["num_heads"], value='64')

# Main content
tab1, tab2 = st.tabs(["Inference", "Training"])

sagemaker_instances = read_sagemaker_instances()

with tab1:
    model_memory, activation_memory, total_inference_memory = calculate_inference_memory(float(model_params), precision, int(batch_size), int(sequence_length))
    
    st.markdown(f'<p class="memory-total">{t["total_inference"]} {total_inference_memory:.2f} GB</p>', unsafe_allow_html=True)
    
    st.write(t["calc_details"])
    st.write(f"{t['model_mem']}: {model_memory:.2f} GB")
    st.write(f"{t['act_mem']}: {activation_memory:.2f} GB")
    st.write(f"{t['total_mem']}: {total_inference_memory:.2f} GB")
    st.latex(r"\text{Total Memory}_{\text{Inference}} \approx 1.2 \times (\text{Model Memory} + \text{Activation Memory})")
    
    st.markdown("---")
    st.subheader(t["suitable_instances"])
    suitable_instances = filter_instances(sagemaker_instances, total_inference_memory)
    st.dataframe(suitable_instances)

    st.markdown("---")
    st.subheader(t["formulas"])
    st.markdown(f"#### {t['inf_mem']}")
    st.latex(r"\text{Model Memory} = \text{params} \times \text{bytes per param}")
    st.latex(r"\text{Activation Memory} \approx \text{sequence length} \times \text{batch size} \times \text{hidden size} \times 2\text{ bytes}")
    st.latex(r"\text{Total Memory}_{\text{Inference}} \approx 1.2 \times (\text{Model Memory} + \text{Activation Memory})")
    
    st.markdown(t["bytes_per_param"])
    st.latex(r"\begin{cases} 0.5 \text{ bytes}, & \text{for int4} \\ 1 \text{ byte}, & \text{for int8} \\ 2 \text{ bytes}, & \text{for fp16/bf16} \\ 4 \text{ bytes}, & \text{for fp32} \end{cases}")

with tab2:
    optimizer = st.selectbox(t["optimizer"], ['AdamW', '8-bit AdamW', 'SGD with momentum'])
    
    model_memory, optimizer_memory, gradient_memory, activation_memory, total_training_memory = calculate_training_memory(float(model_params), precision, int(batch_size), int(sequence_length), optimizer, int(hidden_size), int(num_layers), int(num_attention_heads))
    
    st.markdown(f'<p class="memory-total">{t["total_training"]} {total_training_memory:.2f} GB</p>', unsafe_allow_html=True)
    
    st.write(t["calc_details"])
    st.write(f"{t['model_mem']}: {model_memory:.2f} GB")
    st.write(f"{t['opt_mem']}: {optimizer_memory:.2f} GB")
    st.write(f"{t['grad_mem']}: {gradient_memory:.2f} GB")
    st.write(f"{t['act_mem']}: {activation_memory:.2f} GB")
    st.write(f"{t['total_mem']}: {total_training_memory:.2f} GB")
    st.latex(r"\text{Total Memory}_{\text{Training}} = \text{Model Memory} + \text{Optimizer Memory} + \text{Activation Memory} + \text{Gradient Memory}")
    
    st.markdown("---")
    st.subheader(t["suitable_instances"])
    suitable_instances = filter_instances(sagemaker_instances, total_training_memory)
    st.dataframe(suitable_instances)

    st.markdown("---")
    st.subheader(t["formulas"])
    st.markdown(f"#### {t['train_mem']}")
    st.latex(r"\text{Total Memory}_{\text{Training}} = \text{Model Memory} + \text{Optimizer Memory} + \text{Activation Memory} + \text{Gradient Memory}")
    
    st.markdown(t["where"])
    st.latex(r"\text{Model Memory} = \begin{cases} 4 \times \text{params}, & \text{for fp32} \\ 2 \times \text{params}, & \text{for mixed precision (fp16/bf16) or int8} \\ 0.5 \times \text{params}, & \text{for int4} \end{cases}")
    st.latex(r"\text{Optimizer Memory} = \begin{cases} 12 \times \text{params}, & \text{for AdamW} \\ 6 \times \text{params}, & \text{for 8-bit AdamW} \\ 8 \times \text{params}, & \text{for SGD with momentum} \end{cases}")
    st.latex(r"\text{Gradient Memory} = \begin{cases} 4 \times \text{params}, & \text{for fp32} \\ 2 \times \text{params}, & \text{for mixed precision or int8/int4} \end{cases}")
    st.latex(r"\text{Activation Memory} = 2 \times \text{batch size} \times \text{sequence length} \times \text{hidden size} \times \text{num layers}")

st.markdown(t["note"])

st.markdown("---")
st.markdown(f"### {t['notes']}")
st.markdown(f"- {t['note1']}")
st.markdown(f"- {t['note2']}")
st.markdown(f"- {t['note3']}")
st.markdown(f"- {t['note4']}")
st.markdown(f"- {t['note5']}")
st.markdown(f"- {t['note6']}")
st.markdown(f"- {t['note7']}")

st.markdown(f"### {t['reference']}")
st.markdown("[Transformer Math 101](https://blog.eleuther.ai/transformer-math/)")
