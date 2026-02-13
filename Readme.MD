# 🛡️ Mask Detection Intelligence Dashboard

A professional AI-powered dashboard for real-time mask detection and compliance monitoring using YOLO v8.

## Features

- 🎯 **Real-time Detection**: Frame-by-frame video analysis with YOLO v8
- 📊 **Live Metrics**: Total scanned, compliance rate, violations, and latency
- 📈 **Interactive Charts**: Plotly donut chart showing distribution
- 🚨 **Activity Log**: Real-time logging of violations
- 🎨 **Dark Theme**: Executive-grade dark interface with red accents
- ⚡ **Optimized Performance**: Adjustable processing speed

## How to Use

1. Upload a video file (MP4, AVI, MOV, MKV)
2. Adjust settings (optional):
   - **Confidence Threshold**: 0.5 recommended
   - **Process Every N Frames**: 2 recommended (lower = more accurate, higher = faster)
3. Click "Process Video"
4. View results: processed video, metrics, charts, and activity log

## Model Information

- **Architecture**: YOLO v8 with ONNX optimization
- **Classes**:
  - Class 0: Person with mask correctly worn (Safe) ✅
  - Class 1: Person without mask (High Risk) 🔴
  - Class 2: Person with mask worn incorrectly (Warning) ⚠️
- **Detection**: Real-time bounding boxes with confidence scores

## Technical Stack

- **Framework**: Gradio
- **Detection**: Ultralytics YOLO
- **Video Processing**: OpenCV
- **Visualization**: Plotly
- **Runtime**: ONNX Runtime

## Performance

- **Processing Speed**: ~10-30 FPS (CPU)
- **Accuracy**: Configurable via confidence threshold
- **Supported Formats**: MP4, AVI, MOV, MKV
- **Max Video Length**: Recommended under 2 minutes for optimal performance

## License

MIT License - See LICENSE file for details

## Credits

Built with ❤️ using YOLO, Gradio, and Ultralytics

---

**Stay Safe • Wear Your Mask • Save Lives** 🛡️