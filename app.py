"""
🛡️ MASK DETECTION INTELLIGENCE DASHBOARD
==========================================
Hugging Face Spaces Version - Optimized for Production
Dark Theme | YOLO ONNX | Real-time Detection
"""

import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import tempfile
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

stats = {
    'total_scanned': 0,
    'violations': 0,
    'safe_count': 0,
    'warning_count': 0,
    'risk_count': 0,
    'activity_log': []
}

MODEL_PATH = 'best.onnx'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_yolo_model():
    """Load YOLO ONNX model"""
    try:
        if not os.path.exists(MODEL_PATH):
            return None, f"❌ Model file '{MODEL_PATH}' not found."
        model = YOLO(MODEL_PATH, task='detect')
        return model, "✅ Model loaded successfully!"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def create_donut_chart(safe, warning, risk):
    """Create Plotly donut chart"""
    labels = ['✅ Safe (With Mask)', '⚠️ Warning (Incorrect Mask)', '🔴 Risk (No Mask)']
    values = [safe, warning, risk]
    colors = ['#00ff00', '#ffaa00', '#ff0000']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color='#0a0a0a', width=3)),
        textfont=dict(size=16, color='white', family='Arial Black'),
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text='<b>🎯 DETECTION DISTRIBUTION</b>',
            font=dict(size=24, color='#ff0000', family='Arial Black'),
            x=0.5, xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            font=dict(color='white', size=14),
            orientation='v', yanchor='middle', y=0.5,
            xanchor='left', x=1.05, bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        height=400,
        margin=dict(t=100, b=50, l=50, r=150)
    )
    
    total = safe + risk
    if total > 0:
        fig.add_annotation(
            text=f'<b>{total}</b><br><span style="font-size:14px">TOTAL</span>',
            x=0.5, y=0.5,
            font=dict(size=32, color='#ff0000', family='Arial Black'),
            showarrow=False
        )
    
    return fig

def format_activity_log():
    """Format activity log as HTML"""
    if not stats['activity_log']:
        return "<div style='text-align:center;padding:20px;color:#888'>📋 No violations detected yet</div>"
    
    df = pd.DataFrame(stats['activity_log'][:20])
    
    html = """
    <div style='background:#1a1a1a;padding:15px;border-radius:10px;border:2px solid #ff0000;max-height:300px;overflow-y:auto'>
        <h3 style='color:#ff0000;margin-top:0'>🚨 RECENT VIOLATIONS</h3>
        <table style='width:100%;border-collapse:collapse;color:white'>
            <thead>
                <tr style='background:#ff0000;color:white'>
                    <th style='padding:10px;text-align:left'>Timestamp</th>
                    <th style='padding:10px;text-align:left'>Status</th>
                    <th style='padding:10px;text-align:center'>Confidence</th>
                    <th style='padding:10px;text-align:center'>Alert</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for _, row in df.iterrows():
        html += f"""
                <tr style='border-bottom:1px solid #333'>
                    <td style='padding:8px'>{row['Timestamp']}</td>
                    <td style='padding:8px;color:#ff0000'>{row['Status']}</td>
                    <td style='padding:8px;text-align:center'>{row['Confidence']}</td>
                    <td style='padding:8px;text-align:center'><span style='background:#ff0000;padding:2px 8px;border-radius:3px;font-weight:bold'>{row['Alert']}</span></td>
                </tr>
        """
    
    html += "</tbody></table></div>"
    return html

def reset_stats():
    """Reset statistics"""
    stats['total_scanned'] = 0
    stats['violations'] = 0
    stats['safe_count'] = 0
    stats['warning_count'] = 0
    stats['risk_count'] = 0
    stats['activity_log'] = []

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_video(video_file, confidence_threshold=0.5, process_every_n_frames=2):
    """Process video with YOLO detection"""
    
    if video_file is None:
        return None, "⚠️ Please upload a video first!", "0", "0.0%", "0", "0 ms", create_donut_chart(0, 0, 0), format_activity_log()
    
    reset_stats()
    
    # Load model
    model, msg = load_yolo_model()
    if model is None:
        return None, msg, "0", "0.0%", "0", "0 ms", create_donut_chart(0, 0, 0), format_activity_log()
    
    # Open video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return None, "❌ Cannot open video", "0", "0.0%", "0", "0 ms", create_donut_chart(0, 0, 0), format_activity_log()
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Resize
    target_width = 640
    target_height = int(640 * height / width)
    
    # Output video
    output_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    
    frame_count = 0
    processing_times = []
    
    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame = cv2.resize(frame, (target_width, target_height))
        
        # Detection
        if frame_count % process_every_n_frames == 0:
            start = time.time()
            results = model(frame, conf=confidence_threshold, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    if cls == 0:  # With mask
                        label, color = "WITH MASK", (0, 255, 0)
                        stats['safe_count'] += 1
                    elif cls == 1:  # Without mask
                        label, color = "NO MASK", (0, 0, 255)
                        stats['risk_count'] += 1
                        stats['violations'] += 1
                        stats['activity_log'].insert(0, {
                            'Timestamp': datetime.now().strftime("%H:%M:%S"),
                            'Status': '🔴 No Mask Detected',
                            'Confidence': f"{conf:.1%}",
                            'Alert': 'HIGH'
                        })
                    else:  # cls == 2: Incorrect mask
                        label, color = "INCORRECT MASK", (255, 170, 0)
                        stats['warning_count'] += 1
                        stats['violations'] += 1
                        stats['activity_log'].insert(0, {
                            'Timestamp': datetime.now().strftime("%H:%M:%S"),
                            'Status': '⚠️ Incorrect Mask Detected',
                            'Confidence': f"{conf:.1%}",
                            'Alert': 'MEDIUM'
                        })
                    
                    stats['total_scanned'] += 1
                    
                    # Draw
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    text = f"{label} {conf:.2f}"
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1-h-10), (x1+w, y1), color, -1)
                    cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            processing_times.append(time.time() - start)
        
        # Frame info
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    # Metrics
    compliance = (stats['safe_count']/stats['total_scanned']*100) if stats['total_scanned'] > 0 else 0
    latency = np.mean(processing_times)*1000 if processing_times else 0
    
    chart = create_donut_chart(stats['safe_count'], stats['warning_count'], stats['risk_count'])
    log = format_activity_log()
    
    status = f"""
✅ **PROCESSING COMPLETE!**

📊 **Summary:**
- Total Frames: {frame_count:,}
- Total Detections: {stats['total_scanned']:,}
- Processing Time: {sum(processing_times):.1f}s
- Average FPS: {frame_count/sum(processing_times) if processing_times else 0:.1f}
    """
    
    return (
        output_path, status,
        f"{stats['total_scanned']:,}",
        f"{compliance:.1f}%",
        f"{stats['violations']:,}",
        f"{latency:.0f} ms",
        chart, log
    )

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Custom CSS
custom_css = """
.gradio-container {
    background: #0a0a0a !important;
}
#header {
    background: linear-gradient(90deg, #1a1a1a 0%, #330000 100%);
    padding: 30px;
    border-radius: 15px;
    border-left: 5px solid #ff0000;
    margin-bottom: 20px;
}
"""

# Build interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.HTML("""
        <div id='header'>
            <h1 style='color:white;font-size:36px;margin:0;text-align:center'>🛡️ MASK DETECTION INTELLIGENCE DASHBOARD</h1>
            <p style='color:#ff0000;font-size:14px;letter-spacing:3px;margin-top:10px;text-align:center'>REAL-TIME AI-POWERED MONITORING SYSTEM</p>
        </div>
    """)
    
    with gr.Accordion("📖 How to Use", open=True):
        gr.Markdown("""
        ### Quick Start Guide:
        1. **Upload a video** (MP4, AVI, MOV, MKV) - Max recommended: 2 minutes
        2. **Adjust settings** (optional):
           - **Confidence Threshold**: Higher = stricter detection (0.5 recommended)
           - **Process Every N Frames**: Higher = faster but less accurate (2 recommended)
        3. **Click "🚀 Process Video"** and wait for results
        4. **View results**: Processed video, metrics, charts, and activity log
        
        ### Model Information:
        - **YOLO v8** with ONNX optimization
        - **Classes**: 
          - 0 = With Mask (Safe) ✅
          - 1 = Without Mask (High Risk) 🔴
          - 2 = Incorrect Mask (Warning) ⚠️
        - **Real-time detection** with visual bounding boxes
        
        ### Tips:
        - For faster processing: Increase "Process Every N Frames" to 3-5
        - For better accuracy: Use Confidence = 0.4-0.5
        - Shorter videos process faster
        """)
    
    with gr.Row():
        with gr.Column(scale=2):
            video_input = gr.Video(label="📹 Upload Video File")
        with gr.Column(scale=1):
            conf_slider = gr.Slider(
                0.1, 0.9, 0.5, 0.05,
                label="🎯 Confidence Threshold",
                info="Higher = stricter detection"
            )
            frame_slider = gr.Slider(
                1, 10, 2, 1,
                label="⚡ Process Every N Frames",
                info="Higher = faster processing"
            )
            btn = gr.Button("🚀 Process Video", variant="primary", size="lg")
    
    status = gr.Markdown("📊 **Status:** Ready to process")
    
    gr.Markdown("---")
    gr.Markdown("## 📈 KEY PERFORMANCE INDICATORS")
    with gr.Row():
        m1 = gr.Textbox("0", label="👥 Total Scanned", interactive=False)
        m2 = gr.Textbox("0.0%", label="✅ Compliance Rate", interactive=False)
        m3 = gr.Textbox("0", label="⚠️ Violations Detected", interactive=False)
        m4 = gr.Textbox("0 ms", label="⚡ System Latency", interactive=False)
    
    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=6):
            gr.Markdown("## 🎥 PROCESSED VIDEO")
            video_out = gr.Video(label="Detection Results")
        with gr.Column(scale=4):
            gr.Markdown("## 📊 DETECTION CHART")
            chart = gr.Plot(label="Distribution")
    
    gr.Markdown("---")
    gr.Markdown("## 🚨 ACTIVITY LOG - RECENT VIOLATIONS")
    log = gr.HTML()
    
    gr.Markdown("---")
    gr.HTML("""
        <div style='text-align:center;padding:20px;color:#888'>
            <p>🛡️ Built with YOLO, Gradio & Ultralytics</p>
            <p style='color:#ff0000;font-weight:bold'>Stay Safe • Wear Your Mask • Save Lives</p>
        </div>
    """)
    
    btn.click(
        fn=process_video,
        inputs=[video_input, conf_slider, frame_slider],
        outputs=[video_out, status, m1, m2, m3, m4, chart, log]
    )

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    demo.launch()