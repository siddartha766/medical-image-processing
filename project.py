import cv2
import matplotlib.pyplot as plt

# Step 1: Read image
img = cv2.imread('xray.jpg')

# Step 2: Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 3: Apply Gaussian Blur (noise reduction)
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Step 4: Edge Detection
edges = cv2.Canny(gray, 100, 200)

# Step 5: Thresholding (Segmentation)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Step 6: Show all images
titles = ['Original', 'Grayscale', 'Blur', 'Edges', 'Threshold']
images = [img, gray, blur, edges, thresh]

plt.figure(figsize=(10,8))

for i in range(5):
    plt.subplot(2,3,i+1)
    if i == 0:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()