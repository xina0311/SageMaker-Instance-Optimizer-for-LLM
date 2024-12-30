import streamlit as st
from lang import set_lang
from styles import get_styles
from calculations import calculate_inference_memory, calculate_training_memory
from utils import read_sagemaker_instances, filter_instances

st.set_page_config(layout="wide")
st.markdown(get_styles(), unsafe_allow_html=True)

t = set_lang()

# Sidebar width control
sidebar_width = st.sidebar.slider("Sidebar width", 300, 800, 400, 10)
st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {{
        width: {sidebar_width}px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="main-title">SageMaker Instance Type Optimizer for LLM</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title(t["sidebar_title"])

# Required parameters
model = st.sidebar.text_input(t["model"], value='Llama-3.1-70B-Instruct', help=t["model_desc"])
model_params = st.sidebar.text_input(t["params"], value='70.0', help=t["params_desc"])
precision = st.sidebar.selectbox(t["precision"], ['float16', 'float32', 'bfloat16', 'int8', 'int4'], help=t["precision_desc"])

# Optional parameters
st.sidebar.markdown(f"### {t['optional_params']}")
batch_size = st.sidebar.text_input(t["batch_size"], value='1', help=t["batch_size_desc"])
sequence_length = st.sidebar.text_input(t["seq_len"], value='2048', help=t["seq_len_desc"])
hidden_size = st.sidebar.text_input(t["hidden_size"], value='8192', help=t["hidden_size_desc"])
num_layers = st.sidebar.text_input(t["num_layers"], value='80', help=t["num_layers_desc"])
num_attention_heads = st.sidebar.text_input(t["num_heads"], value='64', help=t["num_heads_desc"])

# Main content
tab1, tab2 = st.tabs(["Inference", "Training"])

sagemaker_instances = read_sagemaker_instances()

with tab1:
    model_memory, total_inference_memory = calculate_inference_memory(float(model_params), precision)
    
    st.markdown(f'<p class="memory-total">{t["total_inference"]}: {total_inference_memory:.2f} GB</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="calc-details">', unsafe_allow_html=True)
    st.write(t["calc_details"])
    st.write(f"{t['model_mem']}: {model_memory:.2f} GB")
    st.write(f"{t['total_mem']}: {total_inference_memory:.2f} GB")
    st.latex(r"\text{Total Memory}_{\text{Inference}} \approx 1.2 \times \text{Model Memory}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="suitable-instances">', unsafe_allow_html=True)
    st.subheader(t["suitable_instances"])
    suitable_instances = filter_instances(sagemaker_instances, total_inference_memory)
    st.dataframe(suitable_instances)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader(t["formulas"])
    st.markdown(f"#### {t['inf_mem']}")
    st.latex(r"\text{Model Memory} = \text{params} \times \textbf{bytes per param}")
    st.latex(r"\text{Total Memory}_{\text{Inference}} \approx 1.2 \times \text{Model Memory}")
    
    st.markdown(t["bytes_per_param"])
    st.latex(r"\begin{cases} 0.5 \text{ bytes}, & \text{for int4} \\ 1 \text{ byte}, & \text{for int8} \\ 2 \text{ bytes}, & \text{for fp16/bf16} \\ 4 \text{ bytes}, & \text{for fp32} \end{cases}")

with tab2:
    optimizer = st.selectbox(t["optimizer"], ['AdamW', '8-bit AdamW', 'SGD with momentum'])
    
    model_memory, optimizer_memory, gradient_memory, activation_memory, total_training_memory = calculate_training_memory(
        float(model_params), precision, int(batch_size), int(sequence_length), optimizer, 
        int(hidden_size), int(num_layers), int(num_attention_heads)
    )
    
    st.markdown(f'<p class="memory-total">{t["total_training"]}: {total_training_memory:.2f} GB</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="calc-details">', unsafe_allow_html=True)
    st.write(t["calc_details"])
    st.write(f"{t['model_mem']}: {model_memory:.2f} GB")
    st.write(f"{t['opt_mem']}: {optimizer_memory:.2f} GB")
    st.write(f"{t['grad_mem']}: {gradient_memory:.2f} GB")
    st.write(f"{t['act_mem']}: {activation_memory:.2f} GB")
    st.write(f"{t['total_mem']}: {total_training_memory:.2f} GB")
    st.latex(r"\text{Total Memory}_{\text{Training}} = \text{Model Memory} + \text{Optimizer Memory} + \text{Activation Memory} + \text{Gradient Memory}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="suitable-instances">', unsafe_allow_html=True)
    st.subheader(t["suitable_instances"])
    suitable_instances = filter_instances(sagemaker_instances, total_training_memory)
    st.dataframe(suitable_instances)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader(t["formulas"])
    st.markdown(f"#### {t['train_mem']}")

    st.latex(r"\text{Total Memory}_{\text{Training}} = \text{Model Memory} + \text{Optimizer Memory} + \text{Activation Memory} + \text{Gradient Memory}")
    
    st.markdown(t["where"])
    st.latex(r"\text{Model Memory} = \begin{cases} 4 \times \text{params}, & \text{for fp32} \\ 2 \times \text{params}, & \text{for fp16 or Mixed-precision (fp16/bf16 and fp32)} \end{cases}")
    st.latex(r"\text{Optimizer Memory} = \begin{cases} 12 \times \text{params}, & \text{for AdamW} \\ 6 \times \text{params}, & \text{for 8-bit AdamW} \\ 8 \times \text{params}, & \text{for SGD with momentum} \end{cases}")
    st.latex(r"\text{Gradient Memory} = \begin{cases} 4 \times \text{params}, & \text{for fp32} \\ 2 \times \text{params}, & \text{for fp16} \end{cases}")
    st.latex(r"\text{Activation Memory (with Full Recomputation)} = 2 \times \text{Batch Size} \cdot \text{Sequence Length} \cdot \text{Hidden Size} \cdot \text{Number of Layers}")

    st.markdown('<h4 style="color: red;">Disclaimer:</h4>', unsafe_allow_html=True)
    st.markdown('<ul style="color: red;">'
            '<li>Hidden size, number of layers, and number of attention heads are estimated based on the model size, but can be adjusted manually.</li>'
            '<li>We assume fp16 for simplicity in memory estimation, though models support fp32, fp16, and mixed-precision training. For fp32, select float32 in parameters.</li>'
            '<li>This calculation excludes additional memory needs for distributed training, which may increase GPU memory usage.</li>'
            '</ul>', unsafe_allow_html=True)

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
st.markdown("[Calculating GPU memory for serving LLMs](https://www.substratus.ai/blog/calculating-gpu-memory-for-llm)")
st.markdown("[LLM-System-Requirements](https://github.com/manuelescobar-dev/LLM-System-Requirements)")
