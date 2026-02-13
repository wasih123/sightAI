import streamlit as st
from ultralytics import YOLO
from ultralytics import SAM
import cv2
from PIL import Image
import numpy as np
import time
from termcolor import colored
import random
import numpy
import pandas as pd
import os
import tempfile
import win32file
import win32con

def set_page_config():
    st.set_page_config(
        page_title="Object Detection - SightAI",
        page_icon="🔍",
        layout="wide"
    )
    
    # Remove sidebar and its toggle button
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        button[kind="header"] {
            display: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

def set_styles():
    st.markdown("""
        <style>
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
        
        /* Parameter section styles */
        .parameter-section {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1.5rem 0;
            border: 1px solid #e0e3e9;
        }
        
        /* Upload section styles */
        .upload-section {
            background-color: #f8f9fa;
            padding: 2rem;
            border-radius: 10px;
            border: 2px dashed #1976D2;
            text-align: center;
            margin: 2rem 0;
        }
        
        /* Results section styles */
        .results-section {
            margin-top: 2rem;
            padding: 1rem;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Rest of your styles remain the same */
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model(model_size='n'):
    """Load YOLOv8 model with caching"""
    #model = SAM("sam_b.pt")
    return YOLO(f'yolov5{model_size}.pt')
    #return model

def draw_text_with_background(img, text, position, font_scale=0.6):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    bg_rect = [
        (x, y - text_height - baseline),
        (x + text_width, y)
    ]
    cv2.putText(img, text, (x, y - baseline), font, font_scale, (0, 0, 0), thickness)

def pil_to_cv2(pil_image):

    rgb_array = np.array(pil_image)
    
    if rgb_array.shape[-1] == 4:
        rgb_array = rgb_array[:, :, :3]
    
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    
    return bgr_array



def process_image(image, model, conf_threshold):

    image_cv = pil_to_cv2(image)
    image_org = image_cv.copy()
    results = model.predict(image_cv, conf=conf_threshold)
    model2 = YOLO("yolo11x-cls.pt")
    class_names = model2.names
    boxes = results[0].boxes
    
    detection_info = []
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        img_cropped = image_org[y1:y2, x1:x2]
        results2 = model2(img_cropped)
        top1_class_idx = results2[0].probs.top1
        top1_conf = results2[0].probs.top1conf
        class_name = class_names[top1_class_idx]
        
        detection_info.append({
            'class_name': class_name,
            'confidence': float(top1_conf),
            'bbox': {
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            }
        })
        
        label_text = f"{class_name}: {top1_conf:.2f}"
        color = [random.randint(0, 255) for _ in range(3)]
        cv2.rectangle(image_org, (x1, y1), (x2, y2), color, 2)
        
        if y1 > 10:
            text_position = (x1, y1 - 3)
        else:
            text_position = (x1, y2 + 5)
        draw_text_with_background(image_org, label_text, text_position)

    return [image_org, len(boxes), detection_info]
    #return results[0]




def process_video(video_file, model, conf_threshold):
    """Process video for object detection"""

    tfile = tempfile.NamedTemporaryFile(delete=False)
    temp_filename = tfile.name
    
    try:
 
        tfile.write(video_file.read())
        tfile.close() 
    
        cap = cv2.VideoCapture(temp_filename)
        
        if not cap.isOpened():
            raise ValueError("Failed to open video file")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        progress_bar = st.progress(0)
        frame_placeholder = st.empty()
        

        detection_counts = {} 
        total_detections = 0
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
   
            progress = int((frame_count / total_frames) * 100)
            progress_bar.progress(progress)

            results = model.predict(frame, conf=conf_threshold)
            
            if results and results[0].boxes:
                boxes = results[0].boxes

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[class_id]

                    detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
                    total_detections += 1
 
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label = f"{class_name} {conf:.2f}"

                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_size[0], y1), color, -1)

                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 0, 0), 2)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
        
        return {
            'total_frames': total_frames,
            'total_detections': total_detections,
            'detection_counts': detection_counts,
            'fps': fps
        }
        
    except Exception as e:
        raise e
    
    finally:
        try:
            cap.release()
            cv2.destroyAllWindows()
            
            time.sleep(0.1)
            
            if os.path.exists(temp_filename):
                try:
                    os.unlink(temp_filename)
                except PermissionError:
                    import win32file
                    import win32con
                    win32file.MoveFileEx(
                        temp_filename, 
                        None,
                        win32file.MOVEFILE_DELAY_UNTIL_REBOOT
                    )
                except Exception as e:
                    st.warning(f"Could not remove temporary file: {temp_filename}")
        except Exception as cleanup_error:
            st.warning(f"Error during cleanup: {cleanup_error}")

def display_detection_results(results):
    """Display detection results and statistics"""
    st.markdown("### 📊 Detection Statistics")
    

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Frames", results['total_frames'])
    with col2:
        st.metric("Total Detections", results['total_detections'])
    with col3:
        st.metric("Video FPS", results['fps'])
    
    st.markdown("### 📈 Detections by Class")
    
    if results['detection_counts']:

        detection_df = pd.DataFrame(
            list(results['detection_counts'].items()),
            columns=['Class', 'Count']
        ).sort_values(by='Count', ascending=False)
        

        st.bar_chart(detection_df.set_index('Class'))
        
        st.markdown("### 📋 Detailed Detection Counts")
        st.dataframe(detection_df)
    else:
        st.info("No objects were detected in this video.")







def main():
    set_page_config()
    set_styles()
    

    st.markdown("""
        <a href="/" class="home-button" target="_self">
            👁️ Home - SightAI
        </a>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="page-header">
            <h1>🔍 Object Detection</h1>
            <p>Upload an image or video to detect and analyze objects using state-of-the-art YOLO model</p>
        </div>
    """, unsafe_allow_html=True)
    
    image_tab, video_tab = st.tabs(["📷 Image Detection", "🎥 Video Detection"])
    
    with image_tab:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        image_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image for object detection",
            key="image_uploader"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if image_file is not None:

            image = Image.open(image_file)
            st.image(image, caption="Original Image", use_column_width=True)
            

            st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
            st.markdown("### Detection Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                model_size = st.selectbox(
                    "Select Model Size",
                    options=['nano', 'small', 'medium', 'large'],
                    format_func=lambda x: f"{x.capitalize()} - {'Fastest' if x == 'nano' else 'Fast' if x == 'small' else 'Balanced' if x == 'medium' else 'Accurate'}",
                    help="Larger models are more accurate but slower",
                    key="image_model_size"
                )
                
                if model_size == 'nano':
                    st.markdown("Speed: ⚡⚡⚡ | Accuracy: ★☆☆")
                elif model_size == 'small':
                    st.markdown("Speed: ⚡⚡ | Accuracy: ★★☆")
                elif model_size == 'medium':
                    st.markdown("Speed: ⚡ | Accuracy: ★★★")
                else:  # large
                    st.markdown("Speed: 🐌 | Accuracy: ★★★★")
            
            with col2:
                confidence = st.slider(
                    "Confidence Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.25,
                    help="Higher values mean more confident detections",
                    key="image_confidence"
                )
                
                if confidence < 0.3:
                    st.markdown("Mode: High Sensitivity (more detections)")
                elif confidence < 0.7:
                    st.markdown("Mode: Balanced Detection")
                else:
                    st.markdown("Mode: High Precision (fewer, more certain detections)")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        if st.button("Detect Objects", type="primary", use_container_width=True):
            try:
                with st.spinner("Processing image..."):

                    model = load_model(model_size[0])
                    start_time = time.time()
                    results = process_image(image, model, confidence)
                    process_time = time.time() - start_time
                    
                    result_image = results[0]
                    #result_image = results.plot()
                    st.image(result_image, caption="Detection Result", use_column_width=True)
                    

                    st.markdown("""
                        <style>
                        .metric-box {
                            background-color: #f8f9fa;
                            padding: 1.5rem;
                            border-radius: 8px;
                            border: 1px solid #e0e3e9;
                            text-align: center;
                            height: 100%;
                        }
                        .metric-label {
                            color: #666;
                            font-size: 1rem;
                            margin-bottom: 0.5rem;
                        }
                        .metric-value {
                            color: #1976D2;  /* Blue color for the values */
                            font-size: 1.5rem;
                            font-weight: bold;
                        }
                        </style>
                    """, unsafe_allow_html=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-label">Objects Detected</div>
                                <div class="metric-value">{results[1]}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-label">Process Time</div>
                                <div class="metric-value">{process_time:.2f}s</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-label">Model Size</div>
                                <div class="metric-value">{model_size.capitalize()}</div>
                            </div>
                        """, unsafe_allow_html=True)


                    if results[2]:  
                        st.markdown("""
                            <style>
                            .detection-table {
                                margin-top: 2rem;
                            }
                            .confidence-value {
                                color: #dc3545; 
                                font-weight: bold;
                            }
                            .label-value {
                                color: #000080;  
                                font-weight: bold;
                            }
                            </style>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("### Detection Details")
                        
                        detection_data = []
                        for idx, det in enumerate(results[2], 1):
                            detection_data.append({
                                "No.": idx,
                                "Label": f"<span class='label-value'>{det['class_name']}</span>",
                                "Confidence": f"<span class='confidence-value'>{det['confidence']:.2f}</span>"
                            })
                        
                        df = pd.DataFrame(detection_data)
                        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

 
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please make sure you have all required packages installed.")

    
    with video_tab:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        video_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video for object detection",
            key="video_uploader"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if video_file is not None:

            st.video(video_file)
            
            st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
            st.markdown("### Detection Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                video_model_size = st.selectbox(
                    "Select Model Size",
                    options=['nano', 'small', 'medium', 'large'],
                    format_func=lambda x: f"{x.capitalize()} - {'Fastest' if x == 'nano' else 'Fast' if x == 'small' else 'Balanced' if x == 'medium' else 'Accurate'}",
                    help="Larger models are more accurate but slower",
                    key="video_model_size"
                )
                
                if video_model_size == 'nano':
                    st.markdown("Speed: ⚡⚡⚡ | Accuracy: ★☆☆")
                elif video_model_size == 'small':
                    st.markdown("Speed: ⚡⚡ | Accuracy: ★★☆")
                elif video_model_size == 'medium':
                    st.markdown("Speed: ⚡ | Accuracy: ★★★")
                else: 
                    st.markdown("Speed: 🐌 | Accuracy: ★★★★")
            
            with col2:
                video_confidence = st.slider(
                    "Confidence Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.25,
                    help="Higher values mean more confident detections",
                    key="video_confidence"
                )
                
                if video_confidence < 0.3:
                    st.markdown("Mode: High Sensitivity (more detections)")
                elif video_confidence < 0.7:
                    st.markdown("Mode: Balanced Detection")
                else:
                    st.markdown("Mode: High Precision (fewer, more certain detections)")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        if st.button("Start Detection", type="primary", use_container_width=True):
            try:
                with st.spinner("Processing video..."):

                    model = YOLO(f'yolov8{video_model_size[0]}.pt')

                    start_time = time.time()
                    results = process_video(video_file, model, video_confidence)
                    process_time = time.time() - start_time
                    st.success(f"Processing completed in {process_time:.2f} seconds")
                    display_detection_results(results)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please make sure you have all required packages installed.")


if __name__ == "__main__":
    main()
