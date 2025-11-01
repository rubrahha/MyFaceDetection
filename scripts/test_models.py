import cv2
import time

print("Testing if models can be loaded...")

# Test 1: Load Caffe SSD
print("\n1. Loading Caffe SSD model...")
try:
    caffe_net = cv2.dnn.readNetFromCaffe(
        'models/deploy.prototxt',
        'models/res10_300x300_ssd_iter_140000.caffemodel'
    )
    print("✓ Caffe SSD loaded successfully!")
except Exception as e:
    print(f"✗ Error loading Caffe: {e}")

# Test 2: Load YuNet
print("\n2. Loading YuNet model...")
try:
    yunet = cv2.FaceDetectorYN.create(
        'models/face_detection_yunet_2023mar.onnx',
        '',
        (320, 320),
        0.6,
        0.3,
        5000
    )
    print("✓ YuNet loaded successfully!")
except Exception as e:
    print(f"✗ Error loading YuNet: {e}")

# Test 3: Load Video
print("\n3. Testing video loading...")
try:
    cap = cv2.VideoCapture('videos/video_benchmark.mp4')
    ret, frame = cap.read()
    if ret:
        print(f"✓ Video loaded! Frame size: {frame.shape}")
    else:
        print("✗ Could not read video frame")
    cap.release()
except Exception as e:
    print(f"✗ Error loading video: {e}")

print("\n✓ All basic tests complete!")