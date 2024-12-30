import pandas as pd

def read_sagemaker_instances():
    df = pd.read_csv('sagemaker_gpu_instances.csv')
    return df

def filter_instances(df, required_memory):
    filtered_df = df[df['Total GPU Memory (GB)'] >= required_memory].sort_values('Total GPU Memory (GB)')
    return filtered_df.reset_index(drop=True)
