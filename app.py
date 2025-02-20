import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import scipy.interpolate as interp

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh

def detect_nose_landmarks(image, profile="front", increase_factor=1.3):
    img_copy = image.copy()
    h, w, _ = image.shape
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None, None, None

        landmarks = results.multi_face_landmarks[0]
        
        if profile == "front":
            nose_points = [4, 5, 195, 6, 19, 94, 97, 2, 98, 327, 168, 122, 50, 280, 279, 239]
        else:
            nose_points = [1, 2, 98, 327, 168, 122, 50, 280, 279, 239, 142, 171, 96, 97]

        nose_coords = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in nose_points]

        # for (x, y) in nose_coords:
        #     cv2.circle(img_copy, (x, y), 3, (0, 255, 0), -1)
        #     cv2.circle(img_copy, (x, y), 1, (255, 0, 0), -1)

        xs, ys = zip(*nose_coords)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        nose_w, nose_h = x_max - x_min, y_max - y_min

        # Improved boundary checking
        padding = int(max(nose_w, nose_h) * (increase_factor - 1) / 2)
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        nose_w = min(w - x_min, nose_w + 2 * padding)
        nose_h = min(h - y_min, nose_h + 2 * padding)

         # Calculate center of the nose region for seamless cloning
        nose_center_x = x_min + nose_w // 2
        nose_center_y = y_min + nose_h // 2

        return (x_min, y_min, nose_w, nose_h), img_copy, nose_coords, (nose_center_x, nose_center_y)

def modify_nose(image, nose_region, nose_coords, nose_center, scale_x=1.2, scale_y=1.2):
    if not nose_region:
        return image

    x, y, w, h = nose_region
    nose_center_x, nose_center_y = nose_center

    # Ensure coordinates are within image bounds
    x = max(0, min(x, image.shape[1] - 1))
    y = max(0, min(y, image.shape[0] - 1))
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    if w <= 0 or h <= 0:
        return image

    nose_roi = image[y:y+h, x:x+w].copy()

    if nose_roi.size == 0:
        return image

    # Calculate new dimensions
    new_w = int(w * scale_x)
    new_h = int(h * scale_y)

    # Ensure new dimensions don't exceed image bounds
    new_w = min(new_w, image.shape[1] - x)
    new_h = min(new_h, image.shape[0] - y)

    if new_w <= 0 or new_h <= 0:
        return image

    try:
        resized_nose = cv2.resize(nose_roi, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    except cv2.error:
        return image

    # --- Color Correction (attempt to match skin tone) ---
    # Calculate average color of original nose and resized nose
    avg_color_original = np.mean(nose_roi, axis=(0, 1))
    avg_color_resized = np.mean(resized_nose, axis=(0, 1))

    # Calculate color correction factor
    color_correction = avg_color_original / avg_color_resized
    color_correction[np.isnan(color_correction)] = 1  # Handle potential NaNs

    # Apply color correction to the resized nose
    resized_nose = np.clip(resized_nose * color_correction, 0, 255).astype(np.uint8)

    # Create a mask with feathering
    mask = np.zeros((new_h, new_w), dtype=np.uint8)
    cv2.ellipse(mask, (new_w // 2, new_h // 2), (new_w // 2, new_h // 2), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (25, 25), 0)  # Adjust blur size

    # Calculate the position to place the nose (centered on the original nose center)
    x_offset = new_w // 2
    y_offset = new_h // 2

    final_x = nose_center_x - x_offset
    final_y = nose_center_y - y_offset

    # Ensure the final coordinates are within image bounds
    final_x = max(0, min(final_x, image.shape[1] - new_w))
    final_y = max(0, min(final_y, image.shape[0] - new_h))

    # --- Use Poisson Blending for seamless cloning ---
    try:
        # Create a region of interest (ROI) with the same size as the resized nose
        roi = image[final_y:final_y + new_h, final_x:final_x + new_w]

        # Ensure the sizes of the source and destination regions match
        if roi.shape[:2] == resized_nose.shape[:2]:
            # Perform seamless cloning using Poisson blending
            center = (final_x + new_w // 2, final_y + new_h // 2) # Corrected center for seamless clone
            image = cv2.seamlessClone(resized_nose, image, mask, center, cv2.NORMAL_CLONE)
        else:
            print("ROI and resized_nose shapes do not match for Poisson blending.")
            return image

    except cv2.error as e:
        print(f"Error in seamlessClone: {e}")
        return image

    return image

def create_all_variations(image, nose_region, nose_coords, nose_center):
    variations = {
        "Original": (1.0, 1.0),
        "Longer": (1.0, 1.15),
        "Wider": (1.2, 1.0),
        "Sharpened Tip": (1.0, 0.85),
        "Rounded Tip": (1.1, 1.15),
        "Shorter": (1.0, 0.8),
        "Slimmer": (0.8, 1.0)
    }
    
    results = {}
    for name, (scale_x, scale_y) in variations.items():
        try:
            results[name] = modify_nose(image.copy(), nose_region, nose_coords, nose_center, scale_x, scale_y)
        except Exception as e:
            st.error(f"Error processing {name} variation: {str(e)}")
            results[name] = image.copy()
    
    return results

def main():
    st.title("AI-Powered Rhinoplasty Simulation")
    st.write("Upload your photo to visualize potential nose shape changes.")
    
    # Sidebar for adjustments
    with st.sidebar:
        st.header("Refinement Options")
        profile_option = st.radio("Choose profile to modify:", 
                                 ["Front Profile", "Side Profile"])
        
        increase_factor = st.slider("Nose Region Expansion", 1.0, 1.5, 1.3, step=0.05,
                                    help="Expand the detected nose region for better modification.")

        # Custom Adjustments
        st.subheader("Fine-tune your nose")
        scale_x = st.slider("Adjust Nose Width", 0.5, 2.0, 1.0, step=0.05)
        scale_y = st.slider("Adjust Nose Height", 0.5, 2.0, 1.0, step=0.05)

    uploaded_file = st.file_uploader("Upload your photo", 
                                    type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        try:
            image = np.array(Image.open(uploaded_file))
            if len(image.shape) == 2:  # Convert grayscale to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # Convert RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                
            profile_type = "front" if profile_option == "Front Profile" else "side"
            
            nose_region, marked_image, nose_coords, nose_center = detect_nose_landmarks(image, profile_type, increase_factor)
            
            if nose_region:
                st.image(marked_image, caption="Detected Nose Landmarks", 
                        use_column_width=True)
                
                # Display variations and custom adjustment side-by-side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Preset Variations")
                    variations = create_all_variations(image, nose_region, nose_coords, nose_center)
                    
                    # Display all variations in a grid
                    cols = st.columns(3)
                    images = list(variations.items())
                    
                    for i in range(len(images)):
                        name, img = images[i]
                        with cols[i % 3]:
                            st.write(f"**{name}**")
                            st.image(img, use_column_width=True)
                            
                with col2:
                    st.subheader("Custom Modification")
                    if st.button("Apply Custom Nose Modification"):
                        result_image = modify_nose(image.copy(), nose_region, nose_coords, nose_center,
                                                scale_x, scale_y)
                        st.image(result_image, caption="Custom Modified Nose", 
                                use_column_width=True)
            else:
                st.error("No face/nose detected. Please upload a clear photo.")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.write("Please try uploading a different photo.")

if __name__ == "__main__":
    main()
