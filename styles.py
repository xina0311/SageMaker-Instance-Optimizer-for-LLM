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
        font-size: 28px !important;
        font-weight: bold !important;
        margin-bottom: 20px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 20px !important;
        font-weight: bold !important;
        color: #0066CC !important;
        padding: 5px 10px !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #E6F2FF !important;
    }
    .memory-total {
        font-size: 24px !important;
        color: #FF4B4B !important;
        font-weight: bold !important;
        background-color: #FFF3F3 !important;
        padding: 10px !important;
        border-radius: 5px !important;
        margin-bottom: 20px !important;
    }
    .calc-details {
        background-color: #F0F0F0 !important;
        padding: 10px !important;
        border-radius: 5px !important;
        margin-bottom: 20px !important;
    }
    .suitable-instances {
        background-color: #E6F2FF !important;
        padding: 10px !important;
        border-radius: 5px !important;
        margin-bottom: 20px !important;
    }
    .red-text {
        color: red;
    }
    /* Ensure input fields adapt to sidebar width */
    .stTextInput, .stNumberInput, .stSelectbox {
        width: 100% !important;
    }
    /* Improve responsiveness for smaller screens */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 100%;
        }
    }
</style>
"""
