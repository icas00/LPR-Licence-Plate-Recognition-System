import cv2
import pytesseract

# Set the Tesseract executable path (for Windows users)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\SAAD\Downloads\HCI\tesseract.exe'

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 100, 200)
    dilated_edges = cv2.dilate(edges, None, iterations=1)
    
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        roi = image[y:y + h, x:x + w]  # Crop the license plate region
        resized_image = cv2.resize(roi, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    else:
        resized_image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    return resized_image

def recognize_text_from_image(image):
    processed_image = preprocess_image(image)
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(processed_image, config=custom_config)
    return text.strip()

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

captured_image = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    h, w, _ = frame.shape
    roi = frame[int(h * 0.05):int(h * 100), int(w * 0.05):int(w * 100)]  # Region of Interest

    cv2.rectangle(frame, (int(w * 0.3), int(h * 0.3)), (int(w * 0.7), int(h * 0.7)), (0, 255, 0), 2)
    cv2.imshow('Webcam Feed', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        captured_image = roi.copy()
        cv2.imwrite('captured_license_plate.jpg', captured_image)
        print("Captured image saved as 'captured_license_plate.jpg'.")
        recognized_text = recognize_text_from_image(captured_image)
        print("Recognized License Plate Number:", recognized_text)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()