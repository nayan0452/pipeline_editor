import numpy as np
import cv2
from enum import Enum
from .lens_calibration_params import LensCalibrationParams

class LensModel(Enum):
    PINHOLE = 1
    FISHEYE = 2

class Pattern(Enum):
    CHESSBOARD = 1
    CIRCLES_GRID = 2
    ASYMMETRIC_CIRCLES_GRID = 3

class LensCalibration:
    def __init__(self, lens_model=LensModel.PINHOLE, pattern_type=Pattern.ASYMMETRIC_CIRCLES_GRID,
                 pattern_rows=4, pattern_cols=11, pattern_size_mm=15, min_area=750):
        self.lens_model = lens_model
        self.pattern_type = pattern_type
        self.pattern_rows = pattern_rows
        self.pattern_cols = pattern_cols
        self.pattern_size_mm = pattern_size_mm
        self.min_area = min_area
        
        # Initialize storage for calibration points
        self.obj_points = []  # 3D points in real world space
        self.img_points = []  # 2D points in image plane
        self.pattern_found_count = 0
        self.image_size = None  # Store image size for calibration
        
        # Create calibration parameters object
        self.calibration_params = LensCalibrationParams()
        
        # Create object points for the calibration pattern
        self.pattern_points = self._create_pattern_points()
        
    def _create_pattern_points(self):
        """Create 3D points for the calibration pattern"""
        if self.pattern_type == Pattern.ASYMMETRIC_CIRCLES_GRID:
            pattern_points = np.zeros((self.pattern_rows * self.pattern_cols, 3), np.float32)
            pattern_points[:, :2] = np.mgrid[0:self.pattern_cols, 0:self.pattern_rows].T.reshape(-1, 2)
            pattern_points[:, :2] *= self.pattern_size_mm
            # For asymmetric grid, shift x coordinates of points in odd rows
            for row in range(self.pattern_rows):
                if row % 2 == 1:
                    pattern_points[row*self.pattern_cols:(row+1)*self.pattern_cols, 0] += self.pattern_size_mm/2
            return pattern_points
        else:
            # Regular grid pattern
            pattern_points = np.zeros((self.pattern_rows * self.pattern_cols, 3), np.float32)
            pattern_points[:, :2] = np.mgrid[0:self.pattern_cols, 0:self.pattern_rows].T.reshape(-1, 2)
            pattern_points *= self.pattern_size_mm
            return pattern_points
    
    def apply(self, frame):
        """Process a frame for calibration pattern detection"""
        if frame is None:
            print("Frame is None")
            return None
            
        # Ensure frame is valid and has proper dimensions
        if len(frame.shape) < 2:
            print("Invalid frame shape")
            return None
            
        # Store image size for calibration (always update it)
        h, w = frame.shape[:2]
        self.image_size = (w, h)  # width, height
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find the calibration pattern
        found = False
        corners = None
        if self.pattern_type == Pattern.ASYMMETRIC_CIRCLES_GRID:
            found, corners = cv2.findCirclesGrid(
                gray, (self.pattern_cols, self.pattern_rows),
                cv2.CALIB_CB_ASYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING
            )
        elif self.pattern_type == Pattern.CIRCLES_GRID:
            found, corners = cv2.findCirclesGrid(
                gray, (self.pattern_cols, self.pattern_rows),
                cv2.CALIB_CB_SYMMETRIC_GRID
            )
        else:  # CHESSBOARD
            found, corners = cv2.findChessboardCorners(
                gray, (self.pattern_cols, self.pattern_rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH +
                cv2.CALIB_CB_NORMALIZE_IMAGE +
                cv2.CALIB_CB_FAST_CHECK
            )
            if found:
                # Refine corners for chessboard
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        if found and corners is not None:
            # Draw the corners
            cv2.drawChessboardCorners(frame, (self.pattern_cols, self.pattern_rows), corners, found)
            
            # Store the points if they're not too close to existing points
            if self._is_new_pattern(corners):
                self.obj_points.append(self.pattern_points.copy())  # Make a copy to ensure independence
                self.img_points.append(corners.copy())  # Make a copy to ensure independence
                self.pattern_found_count += 1
                print(f"Pattern {self.pattern_found_count} stored")
                return frame
            else:
                print("Pattern too similar to previous ones")
                return frame
        
        return frame  # Return frame even if no pattern found, to show video feed
    
    def _is_new_pattern(self, corners):
        """Check if the detected pattern is significantly different from previous ones"""
        if not self.img_points:
            return True
            
        # Calculate the center of the current pattern
        current_center = np.mean(corners, axis=0)
        
        # Compare with previous patterns
        for prev_corners in self.img_points:
            prev_center = np.mean(prev_corners, axis=0)
            
            # Check center distance
            center_distance = np.linalg.norm(current_center - prev_center)
            
            # Check pattern orientation
            current_orientation = np.arctan2(*(corners[-1] - corners[0])[0])
            prev_orientation = np.arctan2(*(prev_corners[-1] - prev_corners[0])[0])
            angle_diff = abs(np.degrees(current_orientation - prev_orientation))
            
            # Accept if EITHER the position OR the angle is different enough
            if center_distance < 30 and angle_diff < 10:  # Both must be similar to reject
                return False
        
        return True
    
    def calibrate(self):
        """Perform the camera calibration"""
        if len(self.obj_points) < 3:
            raise ValueError(f"Need at least 3 valid patterns for calibration, but only have {len(self.obj_points)}")
            
        if self.image_size is None:
            raise ValueError("Image size is None. No frames have been processed.")
            
        if not isinstance(self.image_size, tuple) or len(self.image_size) != 2:
            raise ValueError(f"Invalid image size format: {self.image_size}")
            
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError(f"Invalid image dimensions: width={self.image_size[0]}, height={self.image_size[1]}")
            
        # Initialize camera matrix with rough values
        camera_matrix = np.eye(3, dtype=np.float64)
        camera_matrix[0, 0] = camera_matrix[1, 1] = 1000.0  # Approximate focal length
        camera_matrix[0, 2] = self.image_size[0] / 2
        camera_matrix[1, 2] = self.image_size[1] / 2
        
        # Set calibration flags
        flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_USE_INTRINSIC_GUESS
        
        # Convert object points and image points to numpy arrays and ensure they're float32
        obj_points = [np.asarray(points, dtype=np.float32) for points in self.obj_points]
        img_points = [np.asarray(points, dtype=np.float32) for points in self.img_points]
        
        try:
            # Perform calibration
            rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, self.image_size, camera_matrix,
                None, flags=flags
            )
            
            # Store the calibration results
            self.calibration_params.set_camera_matrix(camera_matrix)
            self.calibration_params.set_distortion_coefficients(dist_coeffs.ravel())
            self.calibration_params.set_enabled(True)
            
            return rms
            
        except Exception as e:
            print(f"Calibration failed: {str(e)}")
            print(f"Image size: {self.image_size}")
            print(f"Number of patterns: {len(self.obj_points)}")
            raise
    
    def get_pattern_found_count(self):
        """Return the number of valid patterns found"""
        return self.pattern_found_count
    
    def get_camera_matrix(self):
        """Return the calibrated camera matrix"""
        return self.calibration_params.get_camera_matrix()
    
    def get_distortion_coefficients(self):
        """Return the calibrated distortion coefficients"""
        return self.calibration_params.get_distortion_coefficients()
    
    def save_calibration(self, filename):
        """Save the calibration parameters"""
        self.calibration_params.save_parameters(filename)
    
    def load_calibration(self, filename):
        """Load calibration parameters"""
        self.calibration_params.load_parameters(filename)
        
    def undistort_image(self, image):
        """Apply undistortion to an image using the calibrated parameters"""
        if not self.calibration_params.is_enabled():
            return image
            
        return cv2.undistort(
            image,
            self.calibration_params.get_camera_matrix(),
            self.calibration_params.get_distortion_coefficients()
        ) 