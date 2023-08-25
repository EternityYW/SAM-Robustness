import numpy as np
import cv2
from PIL import Image, ImageEnhance

def add_gaussian_noise(image, mean=0, stddev=30):
    noise = np.random.normal(mean, stddev, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_pepper_noise(image, prob=0.04):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rnd = np.random.random()
            if rnd < prob:
                output[i][j] = 0
            elif rnd > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def apply_defocus_blur(image, ksize=25, sigma_x=8, sigma_y=None): # ksize must be odd
    if sigma_y is None:
        sigma_y = sigma_x
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma_x, sigma_y)
    return blurred

def apply_motion_blur(image, kernel_size=20):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(image, -1, kernel)

def apply_chromatic_aberration(image, shift_x=15, shift_y=15):
    b, g, r = cv2.split(image)
    # Shift the blue channel
    b = np.roll(b, shift_x, axis=1)
    b = np.roll(b, shift_y, axis=0)
    # Shift the red channel
    r = np.roll(r, -shift_x, axis=1)
    r = np.roll(r, -shift_y, axis=0)
    # Merge the shifted channels
    result = cv2.merge((b, g, r))
    return result

def change_brightness(image, factor=1.5):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def change_contrast(image, factor=2):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def save_jpeg_compressed_image(image, output_image_path, quality=15):
    # JPEG quality ranges from 0 (lowest) to 100 (highest)
    cv2.imwrite(output_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

def elastic_transform(image, alpha=100, sigma=10, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape[:-1]) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape[:-1]) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    return np.transpose(np.array([map_coordinates(image[:,:,i], indices, order=1).reshape(shape[:-1]) for i in range(shape[2])]), (1, 2, 0)).astype(np.uint8)

def add_fog(image, fog_intensity=0.5):
    fog = np.ones_like(image) * 255 * fog_intensity
    fog = fog.astype(image.dtype)  # Ensure the fog has the same data type as the image
    return cv2.addWeighted(image, 1 - fog_intensity, fog, fog_intensity, 0)

def add_snow(image, snow_coefficient=0.3):
    snow_layer = np.random.normal(size=image.shape[:2], loc=255, scale=snow_coefficient * 255).astype(np.uint8)
    snow_layer = cv2.cvtColor(snow_layer, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image, 1 - snow_coefficient, snow_layer, snow_coefficient, 0)

def add_shot_noise(image, shot_noise_intensity=0.1):
    noise = np.random.poisson(image * shot_noise_intensity) / shot_noise_intensity
    return np.clip(noise, 0, 255).astype(np.uint8)

def adjust_saturation(image, saturation_coefficient=0.5):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image = np.array(hsv_image, dtype=np.float64)
    hsv_image[..., 1] = hsv_image[..., 1] * saturation_coefficient
    hsv_image[..., 1] = np.clip(hsv_image[..., 1], 0, 255)
    return cv2.cvtColor(np.array(hsv_image, dtype=np.uint8), cv2.COLOR_HSV2BGR)

def gaussian_blur(image, kernel_size=15):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def add_radial_distortion(image, k1=-0.5, k2=0.05, k3=0, p1=0, p2=0):
    h, w = image.shape[:2]
    camera_matrix = np.array([[w, 0, w / 2], [0, h, h / 2], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
    return undistorted_image