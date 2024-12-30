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
        min-width: 200px;
        max-width: 800px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        min-width: 200px;
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
    #sidebar-content .stTextInput, 
    #sidebar-content .stNumberInput, 
    #sidebar-content .stSelectbox {
        width: 100% !important;
    }
    /* Add custom CSS for the resize handle */
    .resize-handle {
        position: absolute;
        right: -5px;
        top: 0;
        bottom: 0;
        width: 10px;
        cursor: col-resize;
        z-index: 1000;
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
        if (width > 200 && width < 800) {
            sidebar.style.width = width + 'px';
            lastDownX = e.clientX;
            adjustSidebarContent();
        }
    });

    document.addEventListener('mouseup', () => {
        isResizing = false;
        adjustSidebarContent();
    });

    function adjustSidebarContent() {
        const sidebarWidth = sidebar.offsetWidth;
        const sidebarContent = sidebar.querySelector('#sidebar-content');
        if (sidebarContent) {
            sidebarContent.style.width = (sidebarWidth - 40) + 'px';  // Adjust for padding
        }
        const inputs = sidebar.querySelectorAll('input, select');
        inputs.forEach(input => {
            input.style.width = '100%';
        });
    }

    // Initial adjustment
    adjustSidebarContent();

    // Adjust on window resize
    window.addEventListener('resize', adjustSidebarContent);
</script>
"""
