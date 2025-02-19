from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTreeWidget, 
                              QTreeWidgetItem, QPushButton, QLabel, QSpinBox,
                              QLineEdit, QCheckBox, QMessageBox, QComboBox,
                              QFileDialog, QGroupBox, QScrollArea, QWidget,
                              QDoubleSpinBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
import os
import cv2
import importlib
import inspect
from .pipelines.pipeline import Pipeline
from .pipelines.stage import PipelineStage
from typing import Optional, Dict, Any, Type, List
import numpy as np

class PipelineEditor(QDialog):
    def __init__(self, camera_controller, parent=None):
        super().__init__(parent)
        self.camera_controller = camera_controller
        self.pipeline: Optional[Pipeline] = None
        self.available_stages: Dict[str, Type[PipelineStage]] = {}
        self.current_frame = None
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_preview)
        self.load_available_stages()
        self.init_ui()
        
    def load_available_stages(self):
        """Load all available pipeline stages from the stages directory and subdirectories"""
        try:
            stages_dir = os.path.join(os.path.dirname(__file__), "pipelines", "stages")
            if not os.path.exists(stages_dir):
                raise Exception(f"Stages directory not found: {stages_dir}")
                
            for root, _, files in os.walk(stages_dir):
                for file in files:
                    if file.endswith(".py") and not file.startswith("__"):
                        # Get relative path from stages dir
                        rel_path = os.path.relpath(root, stages_dir)
                        if rel_path == ".":
                            module_path = file[:-3]
                        else:
                            # Convert path to module notation
                            module_path = f"{rel_path.replace(os.sep, '.')}.{file[:-3]}"
                            
                        try:
                            module = importlib.import_module(f".pipelines.stages.{module_path}", package="vision")
                            for name, obj in inspect.getmembers(module):
                                if (inspect.isclass(obj) and issubclass(obj, PipelineStage) 
                                    and obj != PipelineStage):
                                    self.available_stages[name] = obj
                                    print(f"Loaded stage: {name}")  # Debug print
                        except Exception as e:
                            print(f"Failed to load stage {module_path}: {str(e)}")
                        
            if not self.available_stages:
                raise Exception("No pipeline stages found")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load pipeline stages: {str(e)}")
        
    def init_ui(self):
        self.setWindowTitle("Pipeline Editor")
        #self.resize(1200, 800)
        self.resize(800, 600)
        
        main_layout = QHBoxLayout(self)
        
        # Left side - Pipeline stages tree, controls, and properties
        left_panel = QVBoxLayout()
        
        # Pipeline controls
        pipeline_group = QGroupBox("Pipeline Controls")
        pipeline_layout = QVBoxLayout()
        
        # Pipeline file controls
        file_controls = QHBoxLayout()
        self.pipeline_path = QLineEdit()
        self.pipeline_path.setReadOnly(True)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_pipeline)
        file_controls.addWidget(self.pipeline_path)
        file_controls.addWidget(browse_btn)
        pipeline_layout.addLayout(file_controls)
        
        # Pipeline action buttons
        action_buttons = QHBoxLayout()
        load_btn = QPushButton("Load Pipeline")
        save_btn = QPushButton("Save Pipeline")
        load_btn.clicked.connect(self.load_pipeline)
        save_btn.clicked.connect(self.save_pipeline)
        action_buttons.addWidget(load_btn)
        action_buttons.addWidget(save_btn)
        pipeline_layout.addLayout(action_buttons)
        
        pipeline_group.setLayout(pipeline_layout)
        left_panel.addWidget(pipeline_group)
        
        # Stages tree
        stages_group = QGroupBox("Pipeline Stages")
        stages_layout = QVBoxLayout()
        
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Stage", "Enabled"])
        self.tree.itemClicked.connect(self.stage_selected)
        self.tree.setDragEnabled(True)
        self.tree.setAcceptDrops(True)
        self.tree.setDragDropMode(QTreeWidget.InternalMove)
        self.tree.dropEvent = self.handle_drop
        stages_layout.addWidget(self.tree)
        
        # Stage controls
        stage_buttons = QHBoxLayout()
        
        # Stage type selector
        self.stage_type = QComboBox()
        self.stage_type.addItems(sorted(self.available_stages.keys()))
        stage_buttons.addWidget(self.stage_type)
        
        add_btn = QPushButton("Add Stage")
        remove_btn = QPushButton("Remove Stage")
        add_btn.clicked.connect(self.add_stage)
        remove_btn.clicked.connect(self.remove_stage)
        stage_buttons.addWidget(add_btn)
        stage_buttons.addWidget(remove_btn)
        
        # Add up/down buttons
        move_buttons = QHBoxLayout()
        up_btn = QPushButton("↑")
        down_btn = QPushButton("↓")
        up_btn.setFixedWidth(30)
        down_btn.setFixedWidth(30)
        up_btn.clicked.connect(self.move_stage_up)
        down_btn.clicked.connect(self.move_stage_down)
        move_buttons.addWidget(up_btn)
        move_buttons.addWidget(down_btn)
        
        # Add selection arrows
        select_up_btn = QPushButton("↑")
        select_down_btn = QPushButton("↓")
        select_up_btn.setFixedWidth(30)
        select_down_btn.setFixedWidth(30)
        select_up_btn.clicked.connect(self.select_stage_above)
        select_down_btn.clicked.connect(self.select_stage_below)
        move_buttons.addSpacing(20)  # Add some space between move and select buttons
        move_buttons.addWidget(select_up_btn)
        move_buttons.addWidget(select_down_btn)
        move_buttons.addStretch()
        
        stages_layout.addLayout(stage_buttons)
        stages_layout.addLayout(move_buttons)
        
        stages_group.setLayout(stages_layout)
        left_panel.addWidget(stages_group)
        
        # Stage properties below stages tree
        props_group = QGroupBox("Stage Properties")
        props_layout = QVBoxLayout()
        
        # Make properties scrollable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        props_widget = QWidget()
        self.props_container = QVBoxLayout(props_widget)
        self.props_container.addStretch()
        
        scroll.setWidget(props_widget)
        props_layout.addWidget(scroll)
        
        props_group.setLayout(props_layout)
        left_panel.addWidget(props_group)

        # Add timing information section
        timing_group = QGroupBox("Stage Timing")
        timing_layout = QVBoxLayout()
        self.timing_label = QLabel("No timing data available")
        self.timing_label.setStyleSheet("""
            QLabel {
                color: #0066cc;
                font-family: monospace;
                padding: 5px;
            }
        """)
        timing_layout.addWidget(self.timing_label)
        timing_group.setLayout(timing_layout)
        left_panel.addWidget(timing_group)
        
        main_layout.addLayout(left_panel, stretch=1)
        
        # Right side - Preview and Output
        right_panel = QVBoxLayout()
        
        # Preview section
        preview_group = QGroupBox("Stage Preview")
        preview_layout = QVBoxLayout()
        
        # Preview controls
        preview_controls = QHBoxLayout()
        self.live_preview = QCheckBox("Live Preview")
        self.live_preview.stateChanged.connect(self.toggle_live_preview)
        preview_controls.addWidget(self.live_preview)
        
        refresh_btn = QPushButton("Refresh Preview")
        refresh_btn.clicked.connect(self.update_preview)
        preview_controls.addWidget(refresh_btn)
        preview_layout.addLayout(preview_controls)
        
        # Preview image
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.preview_label)
        
        preview_group.setLayout(preview_layout)
        right_panel.addWidget(preview_group)
        
        # Stage Output section
        output_group = QGroupBox("Stage Output")
        output_layout = QVBoxLayout()
        
        # Create terminal-like text display
        self.output_display = QLabel()
        self.output_display.setStyleSheet("""
            QLabel {
                background-color: #1E1E1E;
                color: #00FF00;
                font-family: Consolas, Monaco, monospace;
                padding: 10px;
                border-radius: 5px;
            }
        """)
        self.output_display.setWordWrap(True)
        self.output_display.setMinimumHeight(150)
        self.output_display.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        output_layout.addWidget(self.output_display)
        
        output_group.setLayout(output_layout)
        right_panel.addWidget(output_group)
        
        main_layout.addLayout(right_panel, stretch=2)
        
    def handle_drop(self, event):
        """Handle stage reordering via drag and drop"""
        item = self.tree.currentItem()
        if not item or not self.pipeline:
            return
            
        old_idx = self.tree.indexOfTopLevelItem(item)
        QTreeWidget.dropEvent(self.tree, event)
        new_idx = self.tree.indexOfTopLevelItem(item)
        
        if old_idx != new_idx:
            stage = self.pipeline.stages.pop(old_idx)
            self.pipeline.stages.insert(new_idx, stage)
            self.update_preview()
            
    def browse_pipeline(self):
        """Open file dialog to select pipeline config"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Pipeline Config", "", "XML Files (*.xml)")
        if file_path:
            self.pipeline_path.setText(file_path)
            
    def toggle_live_preview(self, state):
        """Toggle live preview timer"""
        if state == Qt.Checked:
            self.preview_timer.start(1000)  # Update every second
        else:
            self.preview_timer.stop()
            
    def update_preview(self):
        """Update the preview image for current stage"""
        if not self.pipeline:
            return
            
        try:
            # Get selected stage
            item = self.tree.currentItem()
            if not item:
                return
                
            stage_idx = self.tree.indexOfTopLevelItem(item)
            if stage_idx < 0:
                return
                
            # Create a temporary pipeline with stages up to selected stage
            preview_pipeline = Pipeline("preview")
            for stage in self.pipeline.stages[:stage_idx + 1]:
                preview_pipeline.add_stage(stage)
                
            # Process pipeline
            try:
                # For first stage (ImageCapture), pass camera_controller
                # For other stages, pass the frame from previous stage
                if preview_pipeline.stages[0].__class__.__name__ == "ImageCapture":
                    results = preview_pipeline.process(self.camera_controller)
                else:
                    # If not starting with ImageCapture, get a fresh frame
                    frame = self.camera_controller.get_frame(1)
                    if frame is None:
                        raise Exception("Failed to capture frame from camera")
                    results = preview_pipeline.process(frame)
                
                # Update timing information
                selected_stage = preview_pipeline.stages[-1]
                stage_time = preview_pipeline.stage_timings.get(selected_stage.name, 0)
                total_time = sum(preview_pipeline.stage_timings.values())
                
                timing_text = (f"Current Stage Time: {stage_time:.3f} seconds\n"
                             f"Cumulative Time: {total_time:.3f} seconds\n"
                             f"Stage Contribution: {(stage_time/total_time*100 if total_time > 0 else 0):.1f}%")
                self.timing_label.setText(timing_text)
                
                # Get the frame to display
                current_frame = None
                stage_result = results.get(selected_stage.name)
                
                if isinstance(stage_result, dict) and 'frame' in stage_result:
                    current_frame = stage_result['frame']
                elif isinstance(stage_result, np.ndarray):
                    current_frame = stage_result
                    
                if current_frame is None:
                    raise Exception("No preview available for this stage")
                    
                # Convert frame for display
                if len(current_frame.shape) == 2:  # Grayscale
                    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2RGB)
                elif current_frame.shape[2] == 3:  # BGR to RGB
                    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                    
                # Scale frame to fit preview
                height, width = current_frame.shape[:2]
                max_size = (400, 300)
                scale = min(max_size[0]/width, max_size[1]/height)
                new_size = (int(width*scale), int(height*scale))
                current_frame = cv2.resize(current_frame, new_size)
                
                # Convert to Qt image
                height, width = current_frame.shape[:2]
                bytes_per_line = 3 * width
                qt_image = QImage(current_frame.data, width, height, 
                                bytes_per_line, QImage.Format_RGB888)
                self.preview_label.setPixmap(QPixmap.fromImage(qt_image))
                
                # Show stage results in terminal output
                if isinstance(stage_result, dict):
                    # Format the output nicely
                    output_lines = []
                    for k, v in stage_result.items():
                        if k != 'frame' and not isinstance(v, np.ndarray):
                            if isinstance(v, (list, tuple)):
                                # Format lists/tuples nicely
                                output_lines.append(f"{k}:")
                                for item in v:
                                    output_lines.append(f"  - {item}")
                            else:
                                output_lines.append(f"{k}: {v}")
                    
                    if output_lines:
                        self.output_display.setText("\n".join(output_lines))
                    else:
                        self.output_display.setText("No data to display for this stage")
                else:
                    self.output_display.setText("No data to display for this stage")
                    
            except Exception as e:
                error_msg = str(e)
                # Split long error messages into multiple lines
                if len(error_msg) > 50:
                    error_msg = "\n".join(error_msg[i:i+50] for i in range(0, len(error_msg), 50))
                self.preview_label.setText(f"Pipeline Error:\n{error_msg}")
                self.output_display.setText(f"Pipeline Error:\n{error_msg}")
                print(f"Pipeline Error: {str(e)}")  # Debug print
                
        except Exception as e:
            error_msg = str(e)
            # Split long error messages into multiple lines
            if len(error_msg) > 50:
                error_msg = "\n".join(error_msg[i:i+50] for i in range(0, len(error_msg), 50))
            self.preview_label.setText(f"Preview Error:\n{error_msg}")
            self.output_display.setText(f"Preview Error:\n{error_msg}")
            print(f"Preview Error: {str(e)}")  # Debug print
            
    def add_stage(self):
        """Add a new stage to the pipeline"""
        if not self.pipeline:
            self.pipeline = Pipeline("new_pipeline")
            
        stage_type = self.stage_type.currentText()
        if stage_type in self.available_stages:
            stage_class = self.available_stages[stage_type]
            stage = stage_class(f"{stage_type.lower()}_{len(self.pipeline.stages)}")
            self.pipeline.add_stage(stage)
            self.update_tree()
            
    def create_property_widget(self, stage, prop_name: str, value: Any) -> QHBoxLayout:
        """Create appropriate widget for property type"""
        layout = QHBoxLayout()
        
        # Create label with tooltip showing property name
        label = QLabel(f"{prop_name}:")
        label.setToolTip(f"Property: {prop_name}")
        layout.addWidget(label)
        
        if isinstance(value, bool):
            widget = QCheckBox()
            widget.setChecked(value)
            # Connect to stateChanged instead of clicked to catch all state changes
            widget.stateChanged.connect(
                lambda state: (
                    self.update_stage_property(stage, prop_name, bool(state == Qt.Checked)),
                    self.update_preview()  # Force immediate preview update
                )
            )
        elif isinstance(value, int):
            widget = QSpinBox()
            widget.setRange(-999999, 999999)
            widget.setValue(value)
            widget.valueChanged.connect(
                lambda v: self.update_stage_property(stage, prop_name, v))
        elif isinstance(value, float):
            widget = QDoubleSpinBox()
            widget.setRange(-999999, 999999)
            widget.setDecimals(3)
            widget.setValue(value)
            widget.valueChanged.connect(
                lambda v: self.update_stage_property(stage, prop_name, v))
        else:
            widget = QLineEdit(str(value))
            widget.textChanged.connect(
                lambda t: self.update_stage_property(stage, prop_name, t))
            
        # Add tooltip to widget
        widget.setToolTip(f"Type: {type(value).__name__}")
        layout.addWidget(widget)
        return layout
        
    def update_stage_property(self, stage, prop_name: str, value: Any):
        """Update stage property and refresh preview"""
        try:
            setattr(stage, prop_name, value)
            self.update_preview()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to update {prop_name}: {str(e)}")
        
    def stage_selected(self, item: QTreeWidgetItem, column: int):
        """Show properties for selected stage"""
        if not self.pipeline or not item:
            return
            
        # Clear existing properties
        while self.props_container.count():
            child = self.props_container.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                while child.layout().count():
                    subchild = child.layout().takeAt(0)
                    if subchild.widget():
                        subchild.widget().deleteLater()
                        
        # Get stage
        stage_idx = self.tree.indexOfTopLevelItem(item)
        if stage_idx < 0:
            return
        stage = self.pipeline.stages[stage_idx]
        
        # Add properties
        for name, value in vars(stage).items():
            if not name.startswith('_'):  # Skip private attributes
                self.props_container.addLayout(
                    self.create_property_widget(stage, name, value))
                
        self.props_container.addStretch()
        self.update_preview()
        
    def update_tree(self):
        """Update the pipeline stages tree"""
        self.tree.clear()
        if not self.pipeline:
            return
            
        for stage in self.pipeline.stages:
            item = QTreeWidgetItem([stage.name, str(stage.enabled)])
            item.setCheckState(1, Qt.Checked if stage.enabled else Qt.Unchecked)
            self.tree.addTopLevelItem(item)
            
    def load_pipeline(self):
        """Load pipeline from XML file"""
        try:
            # Get the src directory path (one level up from vision)
            src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_dir = os.path.join(src_dir, "vision", "pipelines", "configs")
            
            # Open file dialog
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Pipeline Config", 
                default_dir,
                "XML Files (*.xml)"
            )
            
            if file_path:
                print(f"Loading pipeline from: {file_path}")  # Debug print
                self.pipeline = Pipeline.load_from_xml(file_path)
                self.pipeline_path.setText(file_path)
                self.update_tree()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load pipeline: {str(e)}")
            
    def save_pipeline(self):
        """Save pipeline to XML file"""
        if not self.pipeline:
            QMessageBox.warning(self, "Error", "No pipeline to save")
            return
            
        try:
            # Get the src directory path (one level up from vision)
            src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_dir = os.path.join(src_dir, "vision", "pipelines", "configs")
            
            # Open save file dialog
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Pipeline Config",
                default_dir,
                "XML Files (*.xml)"
            )
            
            if file_path:
                # Ensure .xml extension
                if not file_path.lower().endswith('.xml'):
                    file_path += '.xml'
                    
                # Ensure directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                self.pipeline.save_to_xml(file_path)
                self.pipeline_path.setText(file_path)
                QMessageBox.information(self, "Success", "Pipeline saved successfully")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save pipeline:\n{str(e)}")
            
    def remove_stage(self):
        """Remove selected stage from pipeline"""
        item = self.tree.currentItem()
        if not item or not self.pipeline:
            return
            
        stage_idx = self.tree.indexOfTopLevelItem(item)
        if stage_idx >= 0:
            self.pipeline.stages.pop(stage_idx)
            self.update_tree() 
            
    def move_stage_up(self):
        """Move selected stage up in the pipeline"""
        if not self.pipeline:
            return
            
        item = self.tree.currentItem()
        if not item:
            return
            
        current_idx = self.tree.indexOfTopLevelItem(item)
        if current_idx <= 0:  # Already at top
            return
            
        # Move in pipeline
        stage = self.pipeline.stages.pop(current_idx)
        self.pipeline.stages.insert(current_idx - 1, stage)
        
        # Update UI
        self.update_tree()
        self.tree.setCurrentItem(self.tree.topLevelItem(current_idx - 1))
        self.update_preview()
        
    def move_stage_down(self):
        """Move selected stage down in the pipeline"""
        if not self.pipeline:
            return
            
        item = self.tree.currentItem()
        if not item:
            return
            
        current_idx = self.tree.indexOfTopLevelItem(item)
        if current_idx >= len(self.pipeline.stages) - 1:  # Already at bottom
            return
            
        # Move in pipeline
        stage = self.pipeline.stages.pop(current_idx)
        self.pipeline.stages.insert(current_idx + 1, stage)
        
        # Update UI
        self.update_tree()
        self.tree.setCurrentItem(self.tree.topLevelItem(current_idx + 1))
        self.update_preview()
        
    def select_stage_above(self):
        """Select the stage above the current selection"""
        if not self.pipeline:
            return
            
        current_item = self.tree.currentItem()
        if not current_item:
            # If no selection, select the last item
            if self.tree.topLevelItemCount() > 0:
                last_item = self.tree.topLevelItem(self.tree.topLevelItemCount() - 1)
                self.tree.setCurrentItem(last_item)
                self.stage_selected(last_item, 0)  # Trigger stage selected action
            return
            
        current_idx = self.tree.indexOfTopLevelItem(current_item)
        if current_idx > 0:
            new_item = self.tree.topLevelItem(current_idx - 1)
            self.tree.setCurrentItem(new_item)
            self.stage_selected(new_item, 0)  # Trigger stage selected action
        else:
            # If at top, select the bottom item
            last_item = self.tree.topLevelItem(self.tree.topLevelItemCount() - 1)
            self.tree.setCurrentItem(last_item)
            self.stage_selected(last_item, 0)  # Trigger stage selected action
            
    def select_stage_below(self):
        """Select the stage below the current selection"""
        if not self.pipeline:
            return
            
        current_item = self.tree.currentItem()
        if not current_item:
            # If no selection, select the first item
            if self.tree.topLevelItemCount() > 0:
                first_item = self.tree.topLevelItem(0)
                self.tree.setCurrentItem(first_item)
                self.stage_selected(first_item, 0)  # Trigger stage selected action
            return
            
        current_idx = self.tree.indexOfTopLevelItem(current_item)
        if current_idx < self.tree.topLevelItemCount() - 1:
            new_item = self.tree.topLevelItem(current_idx + 1)
            self.tree.setCurrentItem(new_item)
            self.stage_selected(new_item, 0)  # Trigger stage selected action
        else:
            # If at bottom, select the top item
            first_item = self.tree.topLevelItem(0)
            self.tree.setCurrentItem(first_item)
            self.stage_selected(first_item, 0)  # Trigger stage selected action 