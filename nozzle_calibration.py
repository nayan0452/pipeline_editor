import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from PySide6.QtWidgets import QMessageBox
from vision.pipelines.pipeline import Pipeline

class NozzleCalibrator:
    # Single class variable to store fiducial position for all nozzles
    _fiducial_position = None
    
    # Mapping of nozzle IDs to their corresponding rotation axes
    NOZZLE_AXES = {
        'N1': 'A',
        'N2': 'B',
        'N3': 'C',
        'N4': 'D'
    }

    def __init__(self, camera_controller, machine_controller, nozzle_id):
        self.camera_controller = camera_controller
        self.machine_controller = machine_controller
        self.nozzle_id = nozzle_id
        self.rotation_axis = self.NOZZLE_AXES.get(nozzle_id, 'A')  # Default to A if nozzle ID not found
        self.calibration_complete = False
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.xml_path = os.path.join(self.base_path, "resources", "config", "nozzle_calibration.xml")
        
        # Load nozzle detection pipeline
        pipeline_path = os.path.join(self.base_path, "vision", "pipelines", "configs", "nozzle_detection.xml")
        self.pipeline = Pipeline.load_from_xml(pipeline_path)

    def check_existing_calibration(self):
        """Check if calibration data exists for this nozzle"""
        if not os.path.exists(self.xml_path):
            return False
            
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        nozzle_elem = root.find(f".//nozzle[@id='{self.nozzle_id}']")
        
        return nozzle_elem is not None

    def calibrate_nozzle(self):
        """Perform nozzle calibration procedure"""
        try:
            # Calculate runout at different angles
            runout_data = self.measure_nozzle_runout()
            if not runout_data:
                raise Exception("Failed to measure nozzle runout")

            # Save calibration data
            self.save_calibration_data(runout_data)
            
            self.calibration_complete = True
            return True, "Calibration completed successfully"
            
        except Exception as e:
            return False, str(e)

    def measure_nozzle_runout(self):
        """Measure nozzle runout at different angles"""
        runout_data = []
        angles = [0, 45, 90, 135, 180, 225, 270, 315]  # Measure at 8 positions
        
        # Get a sample frame to calculate dimensions
        frame = self.camera_controller.get_frame(3)  # Using bottom camera
        if frame is None:
            raise Exception("Failed to get camera frame")
            
        # Calculate frame center in pixels
        height, width = frame.shape[:2]
        camera_center = (width // 2, height // 2)
        print(f"Camera center (pixels): {camera_center}")
        
        # Get mm per pixel values for camera 3
        mm_per_pixel_x, mm_per_pixel_y = self.camera_controller.get_mm_per_pixel(3)
        if mm_per_pixel_x is None or mm_per_pixel_y is None:
            raise Exception("Camera 3 is not calibrated. Please calibrate the camera first.")
        print(f"Camera calibration: {mm_per_pixel_x:.6f} mm/px (X), {mm_per_pixel_y:.6f} mm/px (Y)")
        
        for angle in angles:
            try:
                # 1. Rotate nozzle to angle using the appropriate axis
                print(f"Moving {self.rotation_axis} axis to angle: {angle}")
                self.machine_controller.send_gcode_without_response(f"G0 {self.rotation_axis}{angle}", "btt")
                
                # Wait for movement to complete
                import time
                time.sleep(2.0)
                
                # 2. Run pipeline to get nozzle center in pixels
                self.pipeline.process(self.camera_controller)
                detection = self.pipeline.results.get('detect')
                
                if not detection or 'circles' not in detection or len(detection['circles']) == 0:
                    print(f"No nozzle detected at angle {angle}")
                    continue
                
                # Get nozzle center in pixels
                circle = detection['circles'][0]
                nozzle_center = (int(circle[0]), int(circle[1]))
                print(f"Nozzle center (pixels): {nozzle_center}")
                
                # 3. Calculate offset in pixels
                offset_x_px = nozzle_center[0] - camera_center[0]
                offset_y_px = nozzle_center[1] - camera_center[1]
                print(f"Offset (pixels): X={offset_x_px}, Y={offset_y_px}")
                
                # 4. Convert offset to mm
                offset_x_mm = offset_x_px * mm_per_pixel_x
                offset_y_mm = offset_y_px * mm_per_pixel_y
                print(f"Offset (mm): X={offset_x_mm:.3f}, Y={offset_y_mm:.3f}")
                
                # Store the data
                runout_data.append({
                    'angle': angle,
                    'x': offset_x_mm,  # Store in mm
                    'y': offset_y_mm   # Store in mm
                })
                self.machine_controller.send_gcode_without_response(f"G0 {self.rotation_axis}0", "btt")

            except Exception as e:
                print(f"Error at angle {angle}: {str(e)}")
                continue
        
        if not runout_data:
            raise Exception("No valid runout measurements collected")
            
        return runout_data

    def save_calibration_data(self, runout_data):
        """Save calibration data to XML file"""
        try:
            if os.path.exists(self.xml_path):
                tree = ET.parse(self.xml_path)
                root = tree.getroot()
            else:
                root = ET.Element("nozzle_calibration")
                nozzles = ET.SubElement(root, "nozzles")
                tree = ET.ElementTree(root)

            # Find or create nozzle element
            nozzle_elem = root.find(f".//nozzle[@id='{self.nozzle_id}']")
            if nozzle_elem is None:
                nozzle_elem = ET.SubElement(root.find("nozzles"), "nozzle")
                nozzle_elem.set("id", self.nozzle_id)

            # Clear existing calibration data
            for child in list(nozzle_elem):
                nozzle_elem.remove(child)

            # Add position data
            position = ET.SubElement(nozzle_elem, "position")
            ET.SubElement(position, "x").text = "150.0"  # Default position
            ET.SubElement(position, "y").text = "150.0"
            ET.SubElement(position, "z").text = "0.0"

            # Add runout data
            runout = ET.SubElement(nozzle_elem, "runout")
            for data in runout_data:
                point = ET.SubElement(runout, "point")
                ET.SubElement(point, "angle").text = str(data['angle'])
                ET.SubElement(point, "x").text = str(data['x'])
                ET.SubElement(point, "y").text = str(data['y'])

            # Calculate runout statistics
            x_values = np.array([data['x'] for data in runout_data])
            y_values = np.array([data['y'] for data in runout_data])
            
            # Calculate radial distances from center
            radial_distances = np.sqrt(x_values**2 + y_values**2)
            
            stats = ET.SubElement(nozzle_elem, "statistics")
            ET.SubElement(stats, "max_runout").text = str(float(np.max(radial_distances)))
            ET.SubElement(stats, "avg_runout").text = str(float(np.mean(radial_distances)))
            ET.SubElement(stats, "std_runout").text = str(float(np.std(radial_distances)))

            # Save the file
            tree.write(self.xml_path, encoding="UTF-8", xml_declaration=True)
            
        except Exception as e:
            raise Exception(f"Failed to save calibration data: {str(e)}")

    def store_fiducial_position(self, x, y):
        """Store the fiducial position for all nozzles"""
        NozzleCalibrator._fiducial_position = (x, y)
        print(f"Stored common fiducial position: X={x:.3f}, Y={y:.3f}")

    def calculate_and_save_offsets(self, overwrite=False):
        """Calculate offsets from fiducial position and save to XML"""
        try:
            if NozzleCalibrator._fiducial_position is None:
                return False, "No fiducial position recorded. Please record fiducial position first."

            # Get current position
            position_response = self.machine_controller.send_gcode("M114", "btt")
            if not position_response:
                return False, "Could not get current machine position"

            # Parse current position
            parts = position_response.split()
            current_x = float(parts[0].split(':')[1])
            current_y = float(parts[1].split(':')[1])

            # Calculate offsets
            x_offset = current_x - NozzleCalibrator._fiducial_position[0]
            y_offset = current_y - NozzleCalibrator._fiducial_position[1]

            # Save offsets to XML
            if os.path.exists(self.xml_path):
                tree = ET.parse(self.xml_path)
                root = tree.getroot()
            else:
                root = ET.Element("nozzle_calibration")
                nozzles = ET.SubElement(root, "nozzles")
                tree = ET.ElementTree(root)

            # Find or create nozzle element
            nozzle_elem = root.find(f".//nozzle[@id='{self.nozzle_id}']")
            if nozzle_elem is None:
                nozzle_elem = ET.SubElement(root.find("nozzles"), "nozzle")
                nozzle_elem.set("id", self.nozzle_id)

            # Check if offsets already exist
            offsets_elem = nozzle_elem.find("offsets")
            if offsets_elem is not None:
                if not overwrite:
                    return False, "EXISTING_OFFSETS"  # Special return value to indicate existing offsets
                nozzle_elem.remove(offsets_elem)

            # Add new offset values
            offsets_elem = ET.SubElement(nozzle_elem, "offsets")
            ET.SubElement(offsets_elem, "x_offset_to_camera").text = str(x_offset)
            ET.SubElement(offsets_elem, "y_offset_to_camera").text = str(y_offset)

            # Save the file
            tree.write(self.xml_path, encoding="UTF-8", xml_declaration=True)
            print(f"Saved offsets for {self.nozzle_id}: X={x_offset:.3f}, Y={y_offset:.3f}")
            
            return True, "Offsets saved successfully"

        except Exception as e:
            return False, str(e)

    def get_offset_values(self):
        """Get the current offset values for this nozzle"""
        try:
            if not os.path.exists(self.xml_path):
                return None, None

            tree = ET.parse(self.xml_path)
            root = tree.getroot()

            # Find nozzle element
            nozzle_elem = root.find(f".//nozzle[@id='{self.nozzle_id}']")
            if nozzle_elem is None:
                return None, None

            # Get offset values
            offsets_elem = nozzle_elem.find("offsets")
            if offsets_elem is None:
                return None, None

            x_offset = float(offsets_elem.find("x_offset_to_camera").text)
            y_offset = float(offsets_elem.find("y_offset_to_camera").text)

            return x_offset, y_offset

        except Exception as e:
            print(f"Error getting offset values: {str(e)}")
            return None, None

    def get_nozzle_offset(self):
        """Get the offset values for this nozzle from XML"""
        try:
            if not os.path.exists(self.xml_path):
                return None, None

            tree = ET.parse(self.xml_path)
            root = tree.getroot()

            # Find nozzle element
            nozzle_elem = root.find(f".//nozzle[@id='{self.nozzle_id}']")
            if nozzle_elem is None:
                return None, None

            # Get offset values
            offsets_elem = nozzle_elem.find("offsets")
            if offsets_elem is None:
                return None, None

            x_offset = float(offsets_elem.find("x_offset_to_camera").text)
            y_offset = float(offsets_elem.find("y_offset_to_camera").text)

            return x_offset, y_offset
        except Exception as e:
            print(f"Error getting nozzle offset: {str(e)}")
            return None, None

    def move_nozzle_to_camera(self):
        """Move nozzle to camera position using stored offsets"""
        try:
            # Get current position
            position_response = self.machine_controller.send_gcode("M114", "btt")
            if not position_response:
                return False, "Could not get current machine position"

            # Parse current position
            parts = position_response.split()
            current_x = float(parts[0].split(':')[1])
            current_y = float(parts[1].split(':')[1])

            # Get offsets
            x_offset, y_offset = self.get_nozzle_offset()
            if x_offset is None or y_offset is None:
                return False, "No offset values found for this nozzle"

            # Calculate target position (subtract offsets to move nozzle to camera)
            target_x = current_x + x_offset
            target_y = current_y + y_offset

            # Send movement command
            response = self.machine_controller.send_gcode(f"G0 X{target_x:.3f} Y{target_y:.3f}", "btt")
            print(f"Moving nozzle to camera. Response: {response}")
            return True, "Movement completed"

        except Exception as e:
            return False, str(e)

    def move_camera_to_nozzle(self):
        """Move camera to nozzle position using stored offsets"""
        try:
            # Get current position
            position_response = self.machine_controller.send_gcode("M114", "btt")
            if not position_response:
                return False, "Could not get current machine position"

            # Parse current position
            parts = position_response.split()
            current_x = float(parts[0].split(':')[1])
            current_y = float(parts[1].split(':')[1])

            # Get offsets
            x_offset, y_offset = self.get_nozzle_offset()
            if x_offset is None or y_offset is None:
                return False, "No offset values found for this nozzle"

            # Calculate target position (add offsets to move camera to nozzle)
            target_x = current_x - x_offset
            target_y = current_y - y_offset

            # Send movement command
            response = self.machine_controller.send_gcode(f"G0 X{target_x:.3f} Y{target_y:.3f}", "btt")
            print(f"Moving camera to nozzle. Response: {response}")
            return True, "Movement completed"

        except Exception as e:
            return False, str(e)
