
import cv2
import numpy as np

def create_shape_shadow(image_path, output_path):
    # Đọc ảnh từ đường dẫn
    image = cv2.imread(image_path)

    # Chuyển đổi ảnh thành ảnh xám
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Phát hiện biên bằng phương pháp Canny
    edges = cv2.Canny(gray_image, 50, 150)

    # Tạo bóng hình dạng theo đường biên
    shadow = cv2.bitwise_not(edges)

    # Hiển thị ảnh gốc và ảnh có bóng hình dạng
    cv2.imshow('Original Image', image)
    cv2.imshow('Shape Shadow', shadow)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Lưu ảnh có bóng hình dạng
    cv2.imwrite(output_path, shadow)


def draw_face_sketch(image_path, output_path):
    # Đọc ảnh từ đường dẫn
    image = cv2.imread(image_path)

    # Chuyển đổi ảnh thành ảnh xám
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt bằng phương pháp Haarcascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    # Vẽ tranh phác họa khuôn mặt
    for (x, y, w, h) in faces:
        face_gray = gray_image[y:y+h, x:x+w]
        _, face_sketch = cv2.threshold(face_gray, 127, 255, cv2.THRESH_BINARY_INV)
        image[y:y+h, x:x+w] = cv2.cvtColor(face_sketch, cv2.COLOR_GRAY2BGR)

    # Hiển thị ảnh gốc và ảnh có tranh phác họa khuôn mặt
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Lưu ảnh có tranh phác họa khuôn mặt
    cv2.imwrite(output_path, image)


# Kịch bản thực nghiệm
create_shape_shadow('path/to/your/image.jpg', 'output/shape_shadow_result.jpg')
draw_face_sketch('path/to/your/face_image.jpg', 'output/face_sketch_result.jpg')
