def calculate_inference_memory(model_params_billions, precision):
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
    
    total_memory = model_memory * 1.2  # Add 20% overhead
    
    return model_memory / 1e9, total_memory / 1e9

def calculate_training_memory(model_params_billions, precision, batch_size, sequence_length, optimizer, hidden_size, num_layers, num_attention_heads):
    model_params = model_params_billions * 1e9
    
    # Model memory (assuming fp16 for training if precision is lower)
    model_memory = model_params * (4 if precision == 'float32' else 2)
    
    # Optimizer memory
    if optimizer == 'AdamW':
        optimizer_memory = model_params * 12
    elif optimizer == '8-bit AdamW':
        optimizer_memory = model_params * 6
    elif optimizer == 'SGD with momentum':
        optimizer_memory = model_params * 8
    
    # Gradient memory (assuming fp16 for training if precision is lower)
    gradient_memory = model_params * (4 if precision == 'float32' else 2)
    
    # Activation memory (using selective recomputation formula)
    activation_memory = 2 * batch_size * sequence_length * hidden_size * num_layers
    
    total_memory = model_memory + optimizer_memory + gradient_memory + activation_memory
    return model_memory / 1e9, optimizer_memory / 1e9, gradient_memory / 1e9, activation_memory / 1e9, total_memory / 1e9
