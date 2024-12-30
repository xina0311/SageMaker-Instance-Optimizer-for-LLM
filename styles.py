def get_styles():
    return """
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    .main-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .memory-total {
        font-size: 20px;
        color: #FF4B4B;
        font-weight: bold;
        background-color: #FFF3F3;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .calc-details {
        margin-bottom: 20px;
    }
    .suitable-instances {
        background-color: #E6F2FF;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
</style>
"""
