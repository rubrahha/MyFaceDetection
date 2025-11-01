import cv2
import pandas as pd
import matplotlib.pyplot as plt

def log_caffe_confidences(video_path, output_csv):
    """
    Log confidence scores for Caffe detections
    """
    print(f"Logging Caffe confidences from: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    net = cv2.dnn.readNetFromCaffe(
        'models/deploy.prototxt',
        'models/res10_300x300_ssd_iter_140000.caffemodel'
    )
    
    confidences = []
    frame_num = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            [104.0, 177.0, 123.0]
        )
        net.setInput(blob)
        detections = net.forward()
        
        h, w = frame.shape[:2]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.1:
                confidences.append({
                    'frame': frame_num,
                    'confidence': float(confidence),
                    'model': 'Caffe SSD'
                })
        
        frame_num += 1
        if frame_num % 100 == 0:
            print(f"  Frame {frame_num}...")
    
    cap.release()
    df = pd.DataFrame(confidences)
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved {len(confidences)} detections to {output_csv}\n")
    return df

def log_yunet_confidences(video_path, output_csv):
    """
    Log confidence scores for YuNet detections
    """
    print(f"Logging YuNet confidences from: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    detector = cv2.FaceDetectorYN.create(
        'models/face_detection_yunet_2023mar.onnx',
        '',
        (320, 320),
        0.0,
        0.3,
        5000
    )
    
    confidences = []
    frame_num = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        detector.setInputSize((frame_width, frame_height))
        _, faces = detector.detect(frame)
        
        if faces is not None:
            for face in faces:
                confidence = face[-1]
                if confidence > 0.1:
                    confidences.append({
                        'frame': frame_num,
                        'confidence': float(confidence),
                        'model': 'YuNet'
                    })
        
        frame_num += 1
        if frame_num % 100 == 0:
            print(f"  Frame {frame_num}...")
    
    cap.release()
    df = pd.DataFrame(confidences)
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved {len(confidences)} detections to {output_csv}\n")
    return df

# Log confidences
df_caffe = log_caffe_confidences('videos/video_benchmark.mp4', 'caffe_confidences.csv')
df_yunet = log_yunet_confidences('videos/video_benchmark.mp4', 'yunet_confidences.csv')

# Create histograms
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Caffe histogram
axes[0].hist(df_caffe['confidence'], bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[0].set_title('Caffe SSD: Confidence Score Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Confidence Score', fontsize=12)
axes[0].set_ylabel('Number of Detections', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=0.5, color='red', linestyle='--', label='Threshold: 0.5')
axes[0].legend()

# YuNet histogram
axes[1].hist(df_yunet['confidence'], bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1].set_title('YuNet: Confidence Score Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Confidence Score', fontsize=12)
axes[1].set_ylabel('Number of Detections', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=0.5, color='red', linestyle='--', label='Threshold: 0.5')
axes[1].legend()

plt.tight_layout()
plt.savefig('confidence_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved histogram comparison to: confidence_comparison.png")

# Print statistics
print("\nCaffe SSD Statistics:")
print(f"  Total detections: {len(df_caffe)}")
print(f"  Average confidence: {df_caffe['confidence'].mean():.3f}")
print(f"  Min confidence: {df_caffe['confidence'].min():.3f}")
print(f"  Max confidence: {df_caffe['confidence'].max():.3f}")

print("\nYuNet Statistics:")
print(f"  Total detections: {len(df_yunet)}")
print(f"  Average confidence: {df_yunet['confidence'].mean():.3f}")
print(f"  Min confidence: {df_yunet['confidence'].min():.3f}")
print(f"  Max confidence: {df_yunet['confidence'].max():.3f}")