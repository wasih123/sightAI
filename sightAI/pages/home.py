import streamlit as st

def set_page_config():
    st.set_page_config(
        page_title="SightAI",
        page_icon="👁️",
        layout="wide",
        menu_items={}
    )
    st.markdown("""
        <style>
        /* Hide sidebar */
        [data-testid="stSidebar"] {
            display: none;
        }
        /* Hide header buttons */
        button[kind="header"] {
            display: none !important;
        }
        /* Hide back arrow button */
        button.st-emotion-cache-1on073z {
            display: none !important;
        }
        /* Hide hamburger menu and footer */
        #MainMenu {
            visibility: hidden;
        }
        footer {
            visibility: hidden;
        }
        </style>
    """, unsafe_allow_html=True)


def set_page_styles():
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        
        /* Home button styles */
        .home-button {
            display: inline-flex;
            align-items: center;
            padding: 0.5rem 1rem;
            background-color: #f0f2f6;
            color: #333333;
            border-radius: 5px;
            text-decoration: none !important;
            margin-bottom: 2rem;
            transition: all 0.3s ease;
            border: 1px solid #e0e3e9;
            cursor: pointer;
            font-weight: 500;
        }
        
        .home-button:hover {
            background-color: #e1e4e8;
            border-color: #d0d3d9;
            text-decoration: none !important;
        }
        
        /* Main container styles */
        .main-header {
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem;
            background: linear-gradient(to right, #FFDAB9, #FFE4B5);
            color: #333333;
            border-radius: 10px;
        }
        
        .main-title {
            font-size: 3.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
            justify-content: center;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .main-description {
            font-size: 1.5rem;
            font-style: italic;
            opacity: 0.9;
        }
        
        /* Section styles */
        .section-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 2rem 0;
            padding: 1rem 0;
            border-bottom: 2px solid #eee;
        }
        
        .section-title {
            font-size: 2rem;
            color: #1976D2;
            margin: 0;
        }
        
        /* Feature box styles */
        .feature-box {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            height: 100%;
            border: 1px solid #eee;
        }
        
        .feature-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            border-color: #1976D2;
        }
        
        .feature-icon {
            font-size: 2rem;
            color: #1976D2;
            margin-bottom: 1rem;
        }
        
        .feature-title {
            font-size: 1.5rem;
            color: #333;
            margin: 1rem 0;
        }
        
        /* Application box styles */
        .app-box {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: all 0.3s ease;
            height: 100%;
            border: 1px solid #eee;
        }
        
        .app-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            background: #1976D2;
            color: white !important;
        }
        
        .app-box:hover * {
            color: white !important;
        }
        
        /* Technology section styles */
        .tech-container {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin: 2rem 0;
            flex-wrap: wrap;
        }
        
        .tech-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.5rem;
            padding: 1rem;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        
        .tech-item:hover {
            transform: translateY(-5px);
        }
        
        .tech-icon {
            width: 50px;
            height: 50px;
            object-fit: contain;
        }
        
        /* Hide Streamlit components */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

def render_header():
    st.markdown("""
        <div class="main-header">
            <h1 class="main-title">👁️ SightAI</h1>
            <p class="main-description">Your gateway to powerful, intelligent AI-driven vision</p>
        </div>
    """, unsafe_allow_html=True)


def render_features_section():
   st.markdown("## 🚀 Key Features")
   st.markdown("---")  # horizontal line
   
   st.markdown("### 🎯 Enhanced Feature Extraction")
   st.markdown("### ⚡ Optimized Efficiency and Speed")
   st.markdown("### 📈 Greater Accuracy with Fewer Parameters") 
   st.markdown("### 🌐 Adaptability Across Environments")
   st.markdown("### 🔄 Broad Range of Supported Tasks")



def render_applications_section():
    st.markdown("""
        <div class="section-header">
            <h2 class="section-title">💡 Applications</h2>
        </div>
        
        <style>
        .app-heading {
            font-size: 1.8rem;
            color: #1976D2;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        
        .app-description {
            font-size: 1.1rem;
            color: #666;
            margin-bottom: 1.5rem;
            line-height: 1.6;
        }
        
        /* Image styling */
        [data-testid="image"] {
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        [data-testid="image"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        .launch-btn {
            margin-top: 1rem;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="app-heading">🔍 Object Detection</h3>', unsafe_allow_html=True)
        st.markdown(
            '<p class="app-description">SightAI transforms images into a world of instantly recognized objects, using cutting-edge YOLO models to detect and label with speed and precision. Unlock new levels of insight and clarity, empowering decisions and discoveries in every frame.</p>',
            unsafe_allow_html=True
        )
        with st.container():
            if st.button("Launch Object Detection", key="obj_det", use_container_width=True):
                #st.switch_page("pages/detection.py")
                st.switch_page("pages/detection_2.py")
    
    with col2:
        st.image("od_2.jpeg", caption="Object Detection Demo")
    
    st.markdown("<br><br>", unsafe_allow_html=True)  # Add spacing between sections
    

    col3, col4 = st.columns(2)
    
    with col3:
        st.image("ot.jpeg", caption="Object Tracking Demo")
    with col4:
        st.markdown('<h3 class="app-heading">🎯 Object Tracking</h3>', unsafe_allow_html=True)
        st.markdown(
            '<p class="app-description">With SightAI’s intelligent object tracking, follow movement with seamless precision, keeping pace with dynamic action and capturing what matters most. Stay focused, stay informed, and experience the future of vision-driven technology.</p>',
            unsafe_allow_html=True
        )
        with st.container():
            if st.button("Launch Object Tracking", key="obj_track", use_container_width=True):
                st.switch_page("pages/tracking.py")

def render_tech_stack():
    st.markdown("""
        <div class="section-header">
            <h2 class="section-title">⚡ Powered By</h2>
        </div>
        <div class="tech-container">
            <div class="tech-item">
                <img src="./yolo.png" class="tech-icon" alt="YOLO">
                <span>YOLO</span>
            </div>
            <div class="tech-item">
                <img src="https://opencv.org/wp-content/uploads/2020/07/OpenCV_logo_black.png" class="tech-icon" alt="OpenCV">
                <span>OpenCV</span>
            </div>
            <div class="tech-item">
                <img src="https://pytorch.org/assets/images/pytorch-logo.png" class="tech-icon" alt="PyTorch">
                <span>PyTorch</span>
            </div>
            <div class="tech-item">
                <img src="https://streamlit.io/images/brand/streamlit-mark-color.png" class="tech-icon" alt="Streamlit">
                <span>Streamlit</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


# Function to render the entire UI
def render_page():
    set_page_config()
    set_page_styles()
    render_header()
    render_features_section()
    render_applications_section()
    render_tech_stack()