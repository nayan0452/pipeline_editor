import os
import xml.etree.ElementTree as ET

class Constants:
    def __init__(self):
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.nozzle_cal_path = os.path.join(self.base_path, "resources", "config", "nozzle_calibration.xml")
        self.camera_cal_path = os.path.join(self.base_path, "resources", "config", "camera_calibration.xml")

    def get_nozzle_offset(self, nozzle_id):
        """Get X and Y offsets for a specific nozzle
        Args:
            nozzle_id (str): Nozzle ID (N1, N2, N3, N4)
        Returns:
            tuple: (x_offset, y_offset) in mm or (None, None) if not found
        """
        try:
            if not os.path.exists(self.nozzle_cal_path):
                print("Nozzle calibration file not found")
                return None, None

            tree = ET.parse(self.nozzle_cal_path)
            root = tree.getroot()

            # Find nozzle element
            nozzle_elem = root.find(f".//nozzle[@id='{nozzle_id}']")
            if nozzle_elem is None:
                print(f"No calibration data found for nozzle {nozzle_id}")
                return None, None

            # Get offset values
            offsets_elem = nozzle_elem.find("offsets")
            if offsets_elem is None:
                print(f"No offset data found for nozzle {nozzle_id}")
                return None, None

            x_offset = float(offsets_elem.find("x_offset_to_camera").text)
            y_offset = float(offsets_elem.find("y_offset_to_camera").text)

            return x_offset, y_offset

        except Exception as e:
            print(f"Error getting nozzle offset: {str(e)}")
            return None, None

    def get_runout_data(self, nozzle_id):
        """Get runout data for a specific nozzle
        Args:
            nozzle_id (str): Nozzle ID (N1, N2, N3, N4)
        Returns:
            dict: Dictionary containing runout points and statistics or None if not found
        """
        try:
            if not os.path.exists(self.nozzle_cal_path):
                print("Nozzle calibration file not found")
                return None

            tree = ET.parse(self.nozzle_cal_path)
            root = tree.getroot()

            # Find nozzle element
            nozzle_elem = root.find(f".//nozzle[@id='{nozzle_id}']")
            if nozzle_elem is None:
                print(f"No calibration data found for nozzle {nozzle_id}")
                return None

            # Get runout data
            runout_elem = nozzle_elem.find("runout")
            if runout_elem is None:
                print(f"No runout data found for nozzle {nozzle_id}")
                return None

            # Collect points
            points = []
            for point in runout_elem.findall("point"):
                points.append({
                    'angle': float(point.find("angle").text),
                    'x': float(point.find("x").text),
                    'y': float(point.find("y").text)
                })

            # Get statistics
            stats_elem = nozzle_elem.find("statistics")
            stats = None
            if stats_elem is not None:
                stats = {
                    'max_runout': float(stats_elem.find("max_runout").text),
                    'avg_runout': float(stats_elem.find("avg_runout").text),
                    'std_runout': float(stats_elem.find("std_runout").text)
                }

            return {
                'points': points,
                'statistics': stats
            }

        except Exception as e:
            print(f"Error getting runout data: {str(e)}")
            return None

    def get_mm_per_pixel(self, camera_index):
        """Get mm per pixel values for a specific camera from calibration XML
        Args:
            camera_index (int): Camera index (1, 2, 3, 4)
        Returns:
            tuple: (mm_per_pixel_x, mm_per_pixel_y) or (None, None) if not found
        """
        try:
            if not os.path.exists(self.camera_cal_path):
                print("Camera calibration file not found")
                return None, None

            tree = ET.parse(self.camera_cal_path)
            root = tree.getroot()

            # Map camera index to camera name
            camera_names = {
                1: "pi_camera1",
                2: "pi_camera2",
                3: "usb_camera1",
                4: "usb_camera2"
            }

            camera_name = camera_names.get(camera_index)
            if not camera_name:
                print(f"Invalid camera index: {camera_index}")
                return None, None

            # Find camera element
            camera_elem = root.find(f".//camera[@name='{camera_name}']")
            if camera_elem is None:
                print(f"No calibration data found for camera {camera_name}")
                return None, None

            # Get mm per pixel values
            mm_per_pixel_elem = camera_elem.find("mm_per_pixel")
            if mm_per_pixel_elem is None:
                print(f"No mm_per_pixel data found for camera {camera_name}")
                return None, None

            mm_per_pixel_x = float(mm_per_pixel_elem.find("x").text)
            mm_per_pixel_y = float(mm_per_pixel_elem.find("y").text)

            print(f"Loaded calibration for {camera_name}: {mm_per_pixel_x:.6f} mm/px (X), {mm_per_pixel_y:.6f} mm/px (Y)")
            return mm_per_pixel_x, mm_per_pixel_y

        except Exception as e:
            print(f"Error getting mm per pixel values: {str(e)}")
            return None, None

# Create a singleton instance
constants = Constants()

# Helper functions for easy access
def get_nozzle_offset(nozzle_id):
    return constants.get_nozzle_offset(nozzle_id)

def get_runout_data(nozzle_id):
    return constants.get_runout_data(nozzle_id)

def get_mm_per_pixel(camera_index):
    return constants.get_mm_per_pixel(camera_index)
