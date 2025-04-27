from flask import Flask, render_template, Response, request, send_from_directory
import cv2
import os
from datetime import datetime
import numpy as np
import random

app = Flask(__name__, static_folder='static')

# Directory to store captured photos
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'photos')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Camera setup
camera = None
# Load the birthday hat image - use an absolute path or a path relative to your Flask app
birthday_hat = None
# Add modern accessory for sweet seventeen
sweet_seventeen_accessory = None
# Add text overlay
sweet_seventeen_text = "17"

def load_hat_image():
    global birthday_hat, sweet_seventeen_accessory
    try:
        hat_path = os.path.join('static', 'hat_birthday.png')  # Change path as needed
        birthday_hat = cv2.imread(hat_path, cv2.IMREAD_UNCHANGED)
        
        # Try to load sweet seventeen accessory
        accessory_path = os.path.join('static', 'accessory_seventeen.png')  # You'll need to add this image
        sweet_seventeen_accessory = cv2.imread(accessory_path, cv2.IMREAD_UNCHANGED)
        
        if birthday_hat is None:
            print(f"Error: Could not load birthday hat image from {hat_path}")
        if sweet_seventeen_accessory is None:
            print(f"Warning: Could not load accessory image from {accessory_path}, will use birthday hat instead")
            sweet_seventeen_accessory = birthday_hat  # Fallback to birthday hat
    except Exception as e:
        print(f"Error loading birthday hat/accessory: {e}")

def init_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open camera, refresh sumpah!.")
            camera = None
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def apply_sweet_seventeen_filter(frame):
    if birthday_hat is None or sweet_seventeen_accessory is None:
        load_hat_image()
        if birthday_hat is None:
            return frame  # Return original frame if hat image can't be loaded
    
    # 1. Subtle image enhancement - increase contrast slightly
    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=7)
    
    # 2. Cool color tone - slight blue tint for a techy look
    blue_overlay = np.zeros_like(frame, dtype=np.float32)
    blue_overlay[:, :] = [128, 90, 80]  # Subtle cool tone (BGR)
    frame = cv2.addWeighted(frame, 0.92, blue_overlay.astype(np.uint8), 0.08, 0)
    
    # 3. Add minimal glitter effect - fewer sparkles
    for _ in range(30):  # Reduced number of sparkles
        x = np.random.randint(0, frame.shape[1])
        y = np.random.randint(0, frame.shape[0])
        radius = np.random.randint(1, 4)  # Smaller radius
        
        # More subdued colors
        if random.random() < 0.5:  # Mix of white/silver sparkles
            brightness = np.random.randint(200, 255)
            color = (brightness, brightness, brightness)  # White/silver
        else:  # And some blue sparkles
            color = (np.random.randint(200, 255), np.random.randint(150, 200), np.random.randint(100, 150))
            
        cv2.circle(frame, (x, y), radius, color, -1)
    
    # 4. Add binary code "1" and "0" floating in background (programmer theme)
    for _ in range(15):
        x = np.random.randint(10, frame.shape[1]-20)
        y = np.random.randint(10, frame.shape[0]-20)
        binary_digit = "1" if random.random() > 0.5 else "0"
        size = np.random.randint(2, 5) * 0.3  # Smaller size
        opacity = np.random.randint(60, 120)  # Semi-transparent
        cv2.putText(frame, binary_digit, (x, y), cv2.FONT_HERSHEY_PLAIN, 
                    size, (70, 130, opacity), 1, cv2.LINE_AA)
    
    # 5. Add the accessory (hat or modern accessory)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Use birthday hat
        selected_accessory = birthday_hat
        
        # Resize the accessory to fit the width of the detected face
        accessory_width = w
        accessory_height = int(accessory_width * selected_accessory.shape[0] / selected_accessory.shape[1])
        
        try:
            resized_accessory = cv2.resize(selected_accessory, (accessory_width, accessory_height))
            
            # Calculate placement position
            y_offset = max(0, y - accessory_height)  # Place hat above face
            x_offset = x
            
            # Ensure x_offset is within frame boundaries
            x_offset = max(0, min(frame.shape[1] - accessory_width, x_offset))
            
            # Get the region where the accessory will be placed
            roi_height = min(resized_accessory.shape[0], frame.shape[0] - y_offset)
            roi_width = min(resized_accessory.shape[1], frame.shape[1] - x_offset)
            
            if roi_height <= 0 or roi_width <= 0:
                continue  # Skip if roi is invalid
                
            # Create a mask from the alpha channel
            alpha_mask = resized_accessory[:roi_height, :roi_width, 3] / 255.0
            alpha_mask = np.stack([alpha_mask] * 3, axis=2)  # Convert to 3 channels
            
            # Get the region of the frame where the accessory will be placed
            roi = frame[y_offset:y_offset + roi_height, x_offset:x_offset + roi_width]
            
            # Extract the RGB channels from the accessory
            accessory_rgb = resized_accessory[:roi_height, :roi_width, :3]
            
            # Blend the accessory with the frame based on the alpha mask
            blended = (1 - alpha_mask) * roi + alpha_mask * accessory_rgb
            
            # Place the blended image back on the frame
            frame[y_offset:y_offset + roi_height, x_offset:x_offset + roi_width] = blended
            
            # Add "17" text with digital effect
            text_size = cv2.getTextSize(sweet_seventeen_text, cv2.FONT_HERSHEY_DUPLEX, 1, 2)[0]
            text_x = x + w + 10  # To the right of the face
            text_y = y + h // 2  # Middle of the face height
            
            # Digital-looking text for programmer theme
            cv2.putText(frame, sweet_seventeen_text, (text_x, text_y), 
                      cv2.FONT_HERSHEY_DUPLEX, 1, (50, 50, 50), 4, cv2.LINE_AA)  # Shadow
            cv2.putText(frame, sweet_seventeen_text, (text_x, text_y), 
                      cv2.FONT_HERSHEY_DUPLEX, 1, (100, 200, 255), 2, cv2.LINE_AA)  # Text
            
        except Exception as e:
            print(f"Error applying accessory: {e}")
            continue  # Skip to the next face

    # 6. Add a subtle visual tech border effect
    h, w = frame.shape[:2]
    border_thickness = 15
    
    # Create corner brackets like code editor brackets
    corner_length = 40
    bracket_color = (130, 200, 250)  # Light blue
    line_thickness = 2
    
    # Top-left bracket
    cv2.line(frame, (0, 0), (corner_length, 0), bracket_color, line_thickness)
    cv2.line(frame, (0, 0), (0, corner_length), bracket_color, line_thickness)
    
    # Top-right bracket
    cv2.line(frame, (w-1, 0), (w-corner_length-1, 0), bracket_color, line_thickness)
    cv2.line(frame, (w-1, 0), (w-1, corner_length), bracket_color, line_thickness)
    
    # Bottom-left bracket
    cv2.line(frame, (0, h-1), (corner_length, h-1), bracket_color, line_thickness)
    cv2.line(frame, (0, h-1), (0, h-corner_length-1), bracket_color, line_thickness)
    
    # Bottom-right bracket
    cv2.line(frame, (w-1, h-1), (w-corner_length-1, h-1), bracket_color, line_thickness)
    cv2.line(frame, (w-1, h-1), (w-1, h-corner_length-1), bracket_color, line_thickness)
    
    # Add subtle version text like software
    cv2.putText(frame, "v.17.0", (w-70, h-10), 
                cv2.FONT_HERSHEY_PLAIN, 0.8, (150, 200, 250), 1, cv2.LINE_AA)

    return frame

def gen_frames():
    init_camera()
    if camera is None:
        return
    
    # Ensure images are loaded
    global birthday_hat, sweet_seventeen_accessory
    if birthday_hat is None:
        try:
            hat_path = os.path.join('static', 'hat_birthday.png')
            birthday_hat = cv2.imread(hat_path, cv2.IMREAD_UNCHANGED)
            if birthday_hat is None:
                print(f"Error: Could not load birthday hat image from {hat_path}")
                
            # Try to load sweet seventeen accessory
            accessory_path = os.path.join('static', 'accessory_seventeen.png')
            sweet_seventeen_accessory = cv2.imread(accessory_path, cv2.IMREAD_UNCHANGED)
            if sweet_seventeen_accessory is None:
                print(f"Warning: Could not load accessory image from {accessory_path}, will use hat instead")
                sweet_seventeen_accessory = birthday_hat  # Fallback
        except Exception as e:
            print(f"Error loading images: {e}")
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Apply sweet seventeen filter
            try:
                frame = apply_sweet_seventeen_filter(frame)
            except Exception as e:
                print(f"Error applying filter: {e}")
            
            # Encode and return frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
@app.route('/')
def index():
    photos = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.png')]
    return render_template('index.html', photos=photos)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture', methods=['POST'])
def capture():
    init_camera()
    if camera is None:
        return "Camera not available, refresh coba!", 500
    success, frame = camera.read()
    if success:
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Apply sweet seventeen birthday filter
        frame = apply_sweet_seventeen_filter(frame)
        
        # Save photo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'photo_{timestamp}.png'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, frame)
        return filename
    return "Failed to capture photo", 500
    # finally:
    #     release_camera()  

@app.route('/photos/<filename>')
def serve_photo(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download_letter')
def download_letter():
    letter_content = """Hi, my Dear...
    Congratulations on reaching level17th !
    Today, the world quietly celebrates you — not just for getting older, but for surviving everything that 
    once tried to break you.
    I know the nights you spent crying silently, fighting battles that no one else ever knew about. I know how heavy the
    weight felt when life kept throwing errors you never asked for.
    But look at you now — standing, breathing, living, even after all the pain that tried to crush you. You are proof
    that something broken can still grow and bloom beautifully.
    You have become a quiet masterpiece — painted with your scars, your smiles, your falls, and your rises.
    And even if the world doesn’t always see how strong you are, I am here... seeing you, admiring you, and being
    endlessly proud of you.
    Keep walking. Keep dreaming. Even when your legs tremble, even when your heart feels small — keep moving forward.
    Because you are so much more precious than you realize. And your story... is far from over.
    I will always be here, just a heartbeat away, quietly celebrating every little victory you win.
    Thank you for surviving this far. I am proud of you, always.

    Even the stars had to collapse first before they could shine brighter.
    Happy 17th, my brave star. Keep burning.
    
                With love, Kak Yen"""
    return Response(
        letter_content,
        mimetype='text/plain',
        headers={'Content-Disposition': 'attachment;filename=Surat_Ulang_Tahun_Dinda.txt'}
    )

if __name__ == '__main__':
    app.run(debug=True)