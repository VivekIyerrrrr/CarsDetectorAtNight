import cv2

# Open the video file
cap = cv2.VideoCapture(r"C:\Users\vivek\OneDrive\Desktop\Madison.mp4")

# Create the background subtractor object
object_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=80, detectShadows=False)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Reached the end of the video or failed to read the frame.")
        break

    height, width, _ = frame.shape
    print(height, width)

    # Resize the frame to fit the window
    scale_percent = 24  # percent of original size
    new_width = int(frame.shape[1] * scale_percent / 100)
    new_height = int(frame.shape[0] * scale_percent / 100)
    dim = (new_width, new_height)

    # Resize frame
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # Define the region of interest (ROI) within the resized frame
    roi = resized_frame[420:750, 0:650]  # Adjust these values based on your requirements

    # Apply the background subtractor to the ROI
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Remove smaller elements
        area = cv2.contourArea(cnt)
        if area > 25000:
            # cv2.drawContours(roi, [cnt], -100, (0, 255, 0), 3)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
            detections.append([x, y, w, h])


    # Display the frames and the mask
    cv2.imshow("Mask", mask)
    cv2.imshow("ROI", roi)
    cv2.imshow('Resized Video Frame', resized_frame)

    # Press 'q' to exit the video window
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
