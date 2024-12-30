def get_styles():
    return """
<style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
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
    .main-title {
        font-size: 28px !important;
        font-weight: bold !important;
        margin-bottom: 20px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
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
"""
