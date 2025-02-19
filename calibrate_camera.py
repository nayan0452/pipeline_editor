import cv2
import numpy as np
from .lens_calibration import LensCalibration, Pattern, LensModel
from .lens_calibration_params import LensCalibrationParams
from controllers.camera_controller import CameraController
import os
import time

def calculate_mm_per_pixel(corners, pattern_size_mm, pattern_cols, pattern_rows):
    """Calculate mm per pixel using the detected chessboard corners"""
    # Calculate distances between adjacent corners
    pixel_distances_x = []
    pixel_distances_y = []
    
    # Reshape corners array for easier access
    corners_reshaped = corners.reshape(pattern_rows, pattern_cols, 2)
    
    # Calculate horizontal distances (along rows)
    for row in range(pattern_rows):
        for col in range(pattern_cols - 1):
            pt1 = corners_reshaped[row, col]
            pt2 = corners_reshaped[row, col + 1]
            distance = np.sqrt(np.sum((pt2 - pt1) ** 2))
            pixel_distances_x.append(distance)
    
    # Calculate vertical distances (along columns)
    for col in range(pattern_cols):
        for row in range(pattern_rows - 1):
            pt1 = corners_reshaped[row, col]
            pt2 = corners_reshaped[row + 1, col]
            distance = np.sqrt(np.sum((pt2 - pt1) ** 2))
            pixel_distances_y.append(distance)
    
    # Calculate average pixels per square
    avg_pixels_per_square_x = np.mean(pixel_distances_x)
    avg_pixels_per_square_y = np.mean(pixel_distances_y)
    
    # Calculate mm per pixel
    mm_per_pixel_x = pattern_size_mm / avg_pixels_per_square_x
    mm_per_pixel_y = pattern_size_mm / avg_pixels_per_square_y
    
    return mm_per_pixel_x, mm_per_pixel_y

def capture_calibration_image(frame, calibration, patterns_found, grid_size):
    """Process a frame for calibration pattern detection"""
    # Ensure frame is in BGR format for OpenCV processing
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        if frame[0,0,0] > frame[0,0,2]:  # Simple check if RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Use the calibration object's apply method to detect and store the pattern
    processed_frame = calibration.apply(frame)
    if processed_frame is not None:
        # Check if a new pattern was actually added
        new_pattern_count = calibration.get_pattern_found_count()
        if new_pattern_count > patterns_found:
            # Get the last added pattern corners
            if calibration.img_points:
                corners = calibration.img_points[-1]
                return True, processed_frame, corners
    
    return False, frame, None

class CameraCalibrator:
    def __init__(self, camera_controller, camera_name):
        self.camera_controller = camera_controller
        self.camera_name = camera_name
        
        # Get camera index based on name
        self.camera_index = None
        if camera_name == 'pi_camera1':
            self.camera_index = 1
        elif camera_name == 'pi_camera2':
            self.camera_index = 2
        elif camera_name == 'usb_camera1':
            self.camera_index = 3
        elif camera_name == 'usb_camera2':
            self.camera_index = 4
        else:
            raise ValueError(f"Invalid camera name: {camera_name}. Valid names are: pi_camera1, pi_camera2, usb_camera1, usb_camera2")

        # Calibration parameters
        self.grid_cols = 8
        self.grid_rows = 6
        self.pattern_size_mm = 1.5
        self.grid_size = (self.grid_cols, self.grid_rows)
        
        # Create calibration object with proper parameters for chessboard
        self.calibration = LensCalibration(
            lens_model=LensModel.PINHOLE,
            pattern_type=Pattern.CHESSBOARD,  # Explicitly set to chessboard
            pattern_rows=self.grid_rows,
            pattern_cols=self.grid_cols,
            pattern_size_mm=self.pattern_size_mm,
            min_area=100  # Lower minimum area threshold
        )
        
        # State variables
        self.calibration_goal = 25
        self.calibration_complete = False
        self.capture_delay = 1.5  # Increased delay for more stable captures
        self.last_capture_time = 0
        self.patterns_found = 0
        self.capturing = False
        self.mm_per_pixel_x = None
        self.mm_per_pixel_y = None
        
        # XML file path setup
        config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "config")
        os.makedirs(config_dir, exist_ok=True)
        self.xml_file = os.path.join(config_dir, "camera_calibration.xml")
        print(f"Calibration file will be saved to: {self.xml_file}")

    def check_existing_calibration(self):
        """Check if calibration exists for this camera"""
        if os.path.exists(self.xml_file):
            existing_cameras = LensCalibrationParams.list_calibrated_cameras(self.xml_file)
            return self.camera_name in existing_cameras
        return False

    def reset_calibration(self):
        """Reset the calibration process"""
        self.calibration = LensCalibration(
            lens_model=LensModel.PINHOLE,
            pattern_type=Pattern.CHESSBOARD,
            pattern_rows=self.grid_rows,
            pattern_cols=self.grid_cols,
            pattern_size_mm=self.pattern_size_mm
        )
        self.patterns_found = 0
        self.capturing = False
        self.mm_per_pixel_x = None
        self.mm_per_pixel_y = None
        self.calibration_complete = False

    def toggle_capture(self):
        """Toggle capture mode"""
        self.capturing = not self.capturing
        self.last_capture_time = time.time()
        return self.capturing

    def perform_calibration(self):
        """Perform the final calibration step"""
        if self.patterns_found >= self.calibration_goal and not self.calibration_complete:
            try:
                # Perform lens calibration
                rms = self.calibration.calibrate()
                
                # Create calibration parameters with camera name
                params = LensCalibrationParams(self.camera_name)
                params.set_camera_matrix(self.calibration.get_camera_matrix())
                params.set_distortion_coefficients(self.calibration.get_distortion_coefficients())
                params.set_mm_per_pixel(self.mm_per_pixel_x, self.mm_per_pixel_y)
                params.set_enabled(True)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.xml_file), exist_ok=True)
                
                # Save to XML file
                params.save_parameters(self.xml_file)
                print(f"Calibration saved to: {self.xml_file}")
                
                self.calibration_complete = True
                return True, {
                    'rms': rms,
                    'camera_matrix': self.calibration.get_camera_matrix(),
                    'distortion_coefficients': self.calibration.get_distortion_coefficients(),
                    'mm_per_pixel_x': self.mm_per_pixel_x,
                    'mm_per_pixel_y': self.mm_per_pixel_y
                }
            except Exception as e:
                print(f"Calibration failed: {str(e)}")
                return False, str(e)
        return False, f"Need {self.calibration_goal - self.patterns_found} more patterns before calibration."

    def process_frame(self):
        """Process a single frame and return the result with overlay information"""
        frame = self.camera_controller.get_distorted_frame(self.camera_index)
        if frame is None:
            print("Failed to get frame from camera")
            return None, "Failed to get frame from camera"
        
        current_time = time.time()
        
        # Always try to detect the pattern for visualization
        if not self.calibration_complete:
            # Convert frame to BGR if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                if frame[0,0,0] > frame[0,0,2]:  # Simple check if RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Process frame through calibration object
            processed_frame = self.calibration.apply(frame.copy())
            if processed_frame is not None:
                frame = processed_frame
                
                # If in capture mode and enough time has passed, check if we can store the pattern
                if self.capturing and (current_time - self.last_capture_time) >= self.capture_delay:
                    # Get the latest pattern if one was found
                    if self.calibration.pattern_found_count > self.patterns_found:
                        self.patterns_found = self.calibration.pattern_found_count
                        
                        # Calculate mm per pixel using the last stored corners
                        if self.calibration.img_points:
                            corners = self.calibration.img_points[-1]
                            mpx, mpy = calculate_mm_per_pixel(corners, self.pattern_size_mm, self.grid_cols, self.grid_rows)
                            if self.mm_per_pixel_x is None:
                                self.mm_per_pixel_x = mpx
                                self.mm_per_pixel_y = mpy
                            else:
                                self.mm_per_pixel_x = (self.mm_per_pixel_x * (self.patterns_found - 1) + mpx) / self.patterns_found
                                self.mm_per_pixel_y = (self.mm_per_pixel_y * (self.patterns_found - 1) + mpy) / self.patterns_found
                        
                        self.last_capture_time = current_time
                        cv2.putText(frame, "Pattern captured!", (frame.shape[1] - 300, 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        if self.patterns_found >= self.calibration_goal:
                            self.capturing = False
                    else:
                        # Pattern was found but not stored (too similar)
                        cv2.putText(frame, "Pattern too similar to previous", (frame.shape[1] - 400, 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        remaining_time = self.capture_delay - (current_time - self.last_capture_time)
                        cv2.putText(frame, f"Next capture in: {remaining_time:.1f}s", 
                                  (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                elif self.capturing:
                    cv2.putText(frame, "Hold pattern steady...", (frame.shape[1] - 300, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                if self.capturing:
                    cv2.putText(frame, "No pattern detected", (frame.shape[1] - 300, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            frame = self.calibration.undistort_image(frame)
        
        # Draw overlay information
        cv2.putText(frame, f"Patterns: {self.patterns_found}/{self.calibration_goal}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Pattern: {self.grid_cols}x{self.grid_rows}", 
                   (20, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        if self.mm_per_pixel_x is not None:
            scale_text = f"Scale: {self.mm_per_pixel_x:.4f}mm/px(X), {self.mm_per_pixel_y:.4f}mm/px(Y)"
            cv2.putText(frame, scale_text, (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Show help text
        if self.calibration_complete:
            cv2.putText(frame, "Calibration Complete", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            if self.capturing:
                cv2.putText(frame, "Hold pattern steady for capture", (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                if self.patterns_found < self.calibration_goal:
                    cv2.putText(frame, "Press 'Start Capture1' to begin", (20, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "Press 'Calibrate' to complete", (20, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Always show camera name
        cv2.putText(frame, f"Camera: {self.camera_name}", 
                   (frame.shape[1] - 300, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        return frame, None

def main(camera_controller=None):
    """
    Run camera calibration process
    Args:
        camera_controller: Optional CameraController instance. If not provided, will create a new instance.
    """
    # Get camera name from user
    print("\n=== Camera Calibration Tool ===")
    camera_name = input("Enter camera name (e.g., 'pi_camera1', 'pi_camera2'): ").strip()
    
    # Create new camera controller only if not provided
    if camera_controller is None:
        camera_controller = CameraController()
        should_cleanup_camera = True
    else:
        should_cleanup_camera = False
    
    calibrator = CameraCalibrator(camera_controller, camera_name)
    
    print("\n=== Calibration Process ===")
    print(f"Using {calibrator.grid_cols}x{calibrator.grid_rows} chessboard pattern")
    print("Step 1: Lens Distortion Calibration")
    print("- Collect 15 different views of the chessboard")
    print("- Hold pattern at different angles")
    print("Step 2: Scale Calibration")
    print("- System will calculate mm per pixel")
    print("\nControls:")
    print("- Press 's' to start/stop capture sequence")
    print("- Press 'r' to reset collection")
    print("- Press 'q' to quit")
    
    while True:
        # Get frame from camera controller
        frame, overlay_info = calibrator.process_frame()
        if frame is None:
            print("Failed to get frame from camera")
            break
        
        # Update overlay information
        status_text = overlay_info['status_text']
        pattern_text = overlay_info['pattern_text']
        scale_text = overlay_info['scale_text']
        help_text = overlay_info['help_text']
        countdown_text = overlay_info['countdown_text']
        
        # Draw overlay on frame
        cv2.putText(frame, status_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, pattern_text, (20, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        if scale_text:
            cv2.putText(frame, scale_text, (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        if help_text:
            cv2.putText(frame, help_text, (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if countdown_text:
            cv2.putText(frame, countdown_text, (20, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        #cv2.imshow('Camera Calibration', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            calibrator.toggle_capture()
        elif key == ord('r'):
            calibrator.reset_calibration()
        elif key == ord('c'):
            success, calibration_data = calibrator.perform_calibration()
            if success:
                print("\nStep 1: Performing Lens Calibration...")
                print(f"RMS: {calibration_data['rms']}")
                print("\nCamera Matrix:")
                print(calibration_data['camera_matrix'])
                print("\nDistortion Coefficients:")
                print(calibration_data['distortion_coefficients'])
                    
                print("\nStep 2: Calculating Final Scale...")
                print(f"X: {calibration_data['mm_per_pixel_x']:.4f} mm/pixel")
                print(f"Y: {calibration_data['mm_per_pixel_y']:.4f} mm/pixel")
                
                print(f"\nCalibration saved to {os.path.abspath(calibrator.xml_file)}")
                print(f"Camera name: {calibrator.camera_name}")
                print(f"Pattern size: {calibrator.grid_cols}x{calibrator.grid_rows}")
            else:
                print(f"\nCalibration failed: {calibration_data}")
    
    # Clean up
    if should_cleanup_camera:
        camera_controller.stop_all_cameras()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
