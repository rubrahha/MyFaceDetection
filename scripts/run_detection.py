import cv2
import time

def detect_faces_caffe(video_path, output_video_path):
    """
    Run Caffe SSD on a video and create output video with boxes drawn
    """
    print(f"\n=== CAFFE SSD Detection ===")
    print(f"Input: {video_path}")
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}, Total frames: {total_frames}")
    
    # Load Caffe model
    net = cv2.dnn.readNetFromCaffe(
        'models/deploy.prototxt',
        'models/res10_300x300_ssd_iter_140000.caffemodel'
    )
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, 
                          (frame_width, frame_height))
    
    frame_count = 0
    start_time = time.time()
    total_detections = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create blob for Caffe (must be 300x300)
        blob = cv2.dnn.blobFromImage(
            frame, 
            1.0, 
            (300, 300),
            [104.0, 177.0, 123.0]
        )
        
        # Run detection
        net.setInput(blob)
        detections = net.forward()
        
        # Draw boxes on the frame
        h, w = frame.shape[:2]
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Only draw if confidence is high enough
            if confidence > 0.5:
                # Get box coordinates
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x1, y1, x2, y2) = box.astype(int)
                
                # Draw green rectangle around face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Write confidence score
                text = f'{confidence:.2f}'
                cv2.putText(frame, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           (0, 255, 0), 2)
                
                total_detections += 1
        
        # Write frame to output video
        out.write(frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")
    
    elapsed = time.time() - start_time
    fps_achieved = frame_count / elapsed if elapsed > 0 else 0
    
    print(f"\n✓ Caffe SSD Results:")
    print(f"  - Frames processed: {frame_count}")
    print(f"  - Total detections: {total_detections}")
    print(f"  - FPS: {fps_achieved:.2f}")
    print(f"  - Output saved: {output_video_path}")
    
    cap.release()
    out.release()
    
    return fps_achieved, total_detections

def detect_faces_yunet(video_path, output_video_path):
    """
    Run YuNet on a video and create output video with boxes drawn
    """
    print(f"\n=== YUNET Detection ===")
    print(f"Input: {video_path}")
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}, Total frames: {total_frames}")
    
    # Load YuNet model
    detector = cv2.FaceDetectorYN.create(
        'models/face_detection_yunet_2023mar.onnx',
        '',
        (320, 320),
        0.6,
        0.3,
        5000
    )
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, 
                          (frame_width, frame_height))
    
    frame_count = 0
    start_time = time.time()
    total_detections = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Set input size for YuNet
        detector.setInputSize((frame_width, frame_height))
        
        # Run detection
        _, faces = detector.detect(frame)
        
        # Draw boxes on frame
        if faces is not None:
            for face in faces:
                # face = [x, y, w, h, confidence, ...]
                x, y, w, h = face[:4].astype(int)
                confidence = face[-1]
                
                # Draw green rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Write confidence
                text = f'{confidence:.2f}'
                cv2.putText(frame, text, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (0, 255, 0), 2)
                
                total_detections += 1
        
        # Write frame to output video
        out.write(frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")
    
    elapsed = time.time() - start_time
    fps_achieved = frame_count / elapsed if elapsed > 0 else 0
    
    print(f"\n✓ YuNet Results:")
    print(f"  - Frames processed: {frame_count}")
    print(f"  - Total detections: {total_detections}")
    print(f"  - FPS: {fps_achieved:.2f}")
    print(f"  - Output saved: {output_video_path}")
    
    cap.release()
    out.release()
    
    return fps_achieved, total_detections

# Run on both videos
print("STARTING FACE DETECTION TESTS")
print("=" * 50)

# Test on benchmark video
caffe_fps_b, caffe_det_b = detect_faces_caffe(
    'videos/video_benchmark.mp4', 
    'caffe_benchmark_output.mp4'
)

yunet_fps_b, yunet_det_b = detect_faces_yunet(
    'videos/video_benchmark.mp4',
    'yunet_benchmark_output.mp4'
)

# Test on stress test video
caffe_fps_s, caffe_det_s = detect_faces_caffe(
    'videos/video_stress_test.mp4',
    'caffe_stress_output.mp4'
)

yunet_fps_s, yunet_det_s = detect_faces_yunet(
    'videos/video_stress_test.mp4',
    'yunet_stress_output.mp4'
)

# Summary
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"\nBenchmark Video:")
print(f"  Caffe SSD: {caffe_fps_b:.2f} FPS, {caffe_det_b} detections")
print(f"  YuNet:     {yunet_fps_b:.2f} FPS, {yunet_det_b} detections")

print(f"\nStress Test Video:")
print(f"  Caffe SSD: {caffe_fps_s:.2f} FPS, {caffe_det_s} detections")
print(f"  YuNet:     {yunet_fps_s:.2f} FPS, {yunet_det_s} detections")