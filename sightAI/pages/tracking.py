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
        page_title="Object Tracking - SightAI",
        page_icon="🎯",
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




def process_video(video_file, model, conf_threshold):
    """Process video for object detection and tracking"""

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
        
        tracking_history = {}  
        detection_counts = {}  
        

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            progress = int((frame_count / total_frames) * 100)
            progress_bar.progress(progress)
            
            results = model.track(frame, conf=conf_threshold, persist=True)
            
            if results and results[0].boxes:
                boxes = results[0].boxes
                

                for box in boxes:
 
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    track_id = int(box.id[0]) if box.id is not None else None
                    

                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[class_id]
                    
                    
                    if track_id is not None:
                        if track_id not in tracking_history:
                            tracking_history[track_id] = {
                                'class': class_name,
                                'frames': [],
                                'positions': []
                            }
                        
                        tracking_history[track_id]['frames'].append(frame_count)
                        tracking_history[track_id]['positions'].append((x1, y1, x2, y2))
                    
                   
                    detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
                    
                    
                    color = (0, 255, 0) 
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label = f"{class_name} #{track_id}" if track_id is not None else class_name
                    label += f" {conf:.2f}"
                    
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_size[0], y1), color, -1)
                    
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 0, 0), 2)
                    
                    if track_id is not None and len(tracking_history[track_id]['positions']) > 1:
                        positions = tracking_history[track_id]['positions'][-30:]  # Last 30 positions
                        for i in range(len(positions) - 1):
                            pt1 = ((positions[i][0] + positions[i][2]) // 2, 
                                  (positions[i][1] + positions[i][3]) // 2)
                            pt2 = ((positions[i+1][0] + positions[i+1][2]) // 2, 
                                  (positions[i+1][1] + positions[i+1][3]) // 2)
                            cv2.line(frame, pt1, pt2, color, 1)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
        

        tracking_stats = []
        for track_id, data in tracking_history.items():
            tracking_stats.append({
                'Track ID': track_id,
                'Class': data['class'],
                'Frame Count': len(data['frames']),
                'First Seen': min(data['frames']),
                'Last Seen': max(data['frames'])
            })
        
        return {
            'total_frames': total_frames,
            'tracking_stats': tracking_stats,
            'detection_counts': detection_counts
        }
        
    except Exception as e:
        raise e
    
    finally:
        # Clean up
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







def display_tracking_results(results):
    """Display tracking results and statistics"""

    st.markdown("### 📊 Tracking Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Frames Processed", results['total_frames'])
        st.metric("Unique Objects Tracked", len(results['tracking_stats']))
    
    with col2:
        st.metric("Total Detections", sum(results['detection_counts'].values()))
        st.metric("Object Classes Detected", len(results['detection_counts']))
    

    st.markdown("### 📈 Detections by Class")
    detection_df = pd.DataFrame(list(results['detection_counts'].items()),
                              columns=['Class', 'Count'])
    st.bar_chart(detection_df.set_index('Class'))
    
    st.markdown("### 🎯 Object Tracking Details")
    if results['tracking_stats']:
        tracking_df = pd.DataFrame(results['tracking_stats'])
        st.dataframe(tracking_df.style.highlight_max(subset=['Frame Count']))
    else:
        st.info("No objects were tracked in this video.")


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
            <h1>🎯 Video Tracking</h1>
            <p>Upload a video to track objects using state-of-the-art YOLO model</p>
        </div>
    """, unsafe_allow_html=True)

    video_tab = st.tabs(["🎥 Video Tracking"])

    with video_tab[0]:

        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        video_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video for object detection",
            key="video_uploader_tracking"
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
            
        if st.button("Start Tracking", type="primary", use_container_width=True):
            try:
                with st.spinner("Processing video..."):

                    model = YOLO(f'yolov8{video_model_size[0]}.pt')

                    start_time = time.time()
                    results = process_video(video_file, model, video_confidence)
                    process_time = time.time() - start_time
                    st.success(f"Processing completed in {process_time:.2f} seconds")
                    display_tracking_results(results)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please make sure you have all required packages installed.")


if __name__ == "__main__":
    main()


