import numpy as np
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

class LensCalibrationParams:
    def __init__(self, camera_name="default"):
        self.camera_name = camera_name
        self.enabled = False
        # Initialize camera matrix (3x3)
        self.camera_matrix = np.zeros((3, 3), dtype=np.float64)
        # Initialize distortion coefficients (5x1)
        self.distortion_coefficients = np.zeros((5, 1), dtype=np.float64)
        # Initialize mm per pixel values
        self.mm_per_pixel_x = None
        self.mm_per_pixel_y = None
        
    def set_camera_matrix(self, matrix):
        """Set the camera matrix after validation"""
        if matrix.shape != (3, 3):
            raise ValueError("Camera matrix must be 3x3")
        self.camera_matrix = matrix.copy()
        
    def set_distortion_coefficients(self, coefficients):
        """Set the distortion coefficients after validation"""
        if len(coefficients) != 5:
            raise ValueError("Distortion coefficients must have 5 elements")
        # Check for extreme values
        if np.any(np.abs(coefficients) > 1e6):
            print("Warning: Distortion coefficients have extreme values - resetting to zero")
            self.distortion_coefficients = np.zeros((5, 1), dtype=np.float64)
        else:
            self.distortion_coefficients = coefficients.reshape(5, 1)
    
    def set_mm_per_pixel(self, mm_per_pixel_x, mm_per_pixel_y):
        """Set the mm per pixel values"""
        self.mm_per_pixel_x = mm_per_pixel_x
        self.mm_per_pixel_y = mm_per_pixel_y
    
    def is_enabled(self):
        return self.enabled
    
    def set_enabled(self, enabled):
        self.enabled = enabled
    
    def get_camera_matrix(self):
        return self.camera_matrix
    
    def get_distortion_coefficients(self):
        return self.distortion_coefficients
    
    def get_mm_per_pixel(self):
        return self.mm_per_pixel_x, self.mm_per_pixel_y
    
    def save_parameters(self, filename):
        """Save calibration parameters to an XML file"""
        # Check if file exists and load existing calibrations
        existing_calibrations = {}
        if os.path.exists(filename):
            try:
                tree = ET.parse(filename)
                root = tree.getroot()
                for camera in root.findall('camera'):
                    cam_name = camera.get('name')
                    existing_calibrations[cam_name] = camera
            except ET.ParseError:
                root = ET.Element("camera_calibrations")
        else:
            root = ET.Element("camera_calibrations")
        
        # Create or update camera element
        if self.camera_name in existing_calibrations:
            camera_elem = existing_calibrations[self.camera_name]
            root.remove(camera_elem)
        
        camera_elem = ET.SubElement(root, "camera")
        camera_elem.set("name", self.camera_name)
        
        # Save enabled state
        enabled_elem = ET.SubElement(camera_elem, "enabled")
        enabled_elem.text = str(self.enabled)
        
        # Save camera matrix
        camera_matrix_elem = ET.SubElement(camera_elem, "camera_matrix")
        for i in range(3):
            row_elem = ET.SubElement(camera_matrix_elem, f"row_{i}")
            row_elem.text = " ".join([f"{x:.10f}" for x in self.camera_matrix[i]])
        
        # Save distortion coefficients
        dist_coeffs_elem = ET.SubElement(camera_elem, "distortion_coefficients")
        dist_coeffs_elem.text = " ".join([f"{x:.10f}" for x in self.distortion_coefficients.ravel()])
        
        # Save mm per pixel values if available
        if self.mm_per_pixel_x is not None and self.mm_per_pixel_y is not None:
            scale_elem = ET.SubElement(camera_elem, "mm_per_pixel")
            x_elem = ET.SubElement(scale_elem, "x")
            x_elem.text = f"{self.mm_per_pixel_x:.10f}"
            y_elem = ET.SubElement(scale_elem, "y")
            y_elem.text = f"{self.mm_per_pixel_y:.10f}"
        
        # Create a pretty-printed XML string
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
        
        # Save to file
        with open(filename, "w") as f:
            f.write(xml_str)
    
    def load_parameters(self, filename, camera_name=None):
        """Load calibration parameters from an XML file"""
        if camera_name is None:
            camera_name = self.camera_name
            
        tree = ET.parse(filename)
        root = tree.getroot()
        
        # Find the camera element with matching name
        camera_elem = None
        for cam in root.findall('camera'):
            if cam.get('name') == camera_name:
                camera_elem = cam
                break
        
        if camera_elem is None:
            raise ValueError(f"No calibration data found for camera '{camera_name}'")
        
        # Load enabled state
        self.enabled = camera_elem.find("enabled").text.lower() == "true"
        
        # Load camera matrix
        camera_matrix_elem = camera_elem.find("camera_matrix")
        for i in range(3):
            row_text = camera_matrix_elem.find(f"row_{i}").text
            self.camera_matrix[i] = [float(x) for x in row_text.split()]
        
        # Load distortion coefficients
        dist_coeffs_text = camera_elem.find("distortion_coefficients").text
        self.distortion_coefficients = np.array([float(x) for x in dist_coeffs_text.split()], 
                                              dtype=np.float64).reshape(5, 1)
        
        # Load mm per pixel values if available
        scale_elem = camera_elem.find("mm_per_pixel")
        if scale_elem is not None:
            self.mm_per_pixel_x = float(scale_elem.find("x").text)
            self.mm_per_pixel_y = float(scale_elem.find("y").text)
            
    @staticmethod
    def list_calibrated_cameras(filename):
        """List all calibrated cameras in the XML file"""
        if not os.path.exists(filename):
            return []
            
        tree = ET.parse(filename)
        root = tree.getroot()
        return [cam.get('name') for cam in root.findall('camera')] 