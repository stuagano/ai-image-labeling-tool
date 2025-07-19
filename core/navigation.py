"""
Core Navigation Module for Image Labeling Application
Centralizes navigation functionality to eliminate DRY violations
"""

import streamlit as st
from typing import Optional, Callable
from streamlit_img_label.manage import ImageDirManager


class NavigationController:
    """Centralized navigation controller for image labeling applications"""
    
    def __init__(self, image_dir_manager: ImageDirManager):
        """
        Initialize navigation controller
        
        Args:
            image_dir_manager: ImageDirManager instance for handling file operations
        """
        self.idm = image_dir_manager
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables for navigation"""
        if "files" not in st.session_state:
            st.session_state["files"] = self.idm.get_all_files()
            st.session_state["annotation_files"] = self.idm.get_exist_annotation_files()
            st.session_state["image_index"] = 0
        else:
            self.idm.set_all_files(st.session_state["files"])
            self.idm.set_annotation_files(st.session_state["annotation_files"])
    
    def refresh(self):
        """Refresh file lists and reset image index"""
        st.session_state["files"] = self.idm.get_all_files()
        st.session_state["annotation_files"] = self.idm.get_exist_annotation_files()
        st.session_state["image_index"] = 0
    
    def next_image(self):
        """Navigate to next image"""
        image_index = st.session_state["image_index"]
        if image_index < len(st.session_state["files"]) - 1:
            st.session_state["image_index"] += 1
        else:
            st.warning('This is the last image.')
    
    def previous_image(self):
        """Navigate to previous image"""
        image_index = st.session_state["image_index"]
        if image_index > 0:
            st.session_state["image_index"] -= 1
        else:
            st.warning('This is the first image.')
    
    def next_annotate_file(self):
        """Navigate to next image that needs annotation"""
        image_index = st.session_state["image_index"]
        next_image_index = self.idm.get_next_annotation_image(image_index)
        if next_image_index is not None:
            st.session_state["image_index"] = next_image_index
        else:
            st.warning("All images are annotated.")
            self.next_image()
    
    def go_to_image(self):
        """Go to specific image based on file selection"""
        if "file" in st.session_state and st.session_state["file"] in st.session_state["files"]:
            file_index = st.session_state["files"].index(st.session_state["file"])
            st.session_state["image_index"] = file_index
    
    def get_current_image_info(self) -> dict:
        """Get information about current image and navigation state"""
        n_files = len(st.session_state["files"])
        n_annotate_files = len(st.session_state["annotation_files"])
        
        return {
            "total_files": n_files,
            "annotated_files": n_annotate_files,
            "remaining_files": n_files - n_annotate_files,
            "current_index": st.session_state["image_index"],
            "current_file": st.session_state["files"][st.session_state["image_index"]] if st.session_state["files"] else None
        }
    
    def render_navigation_sidebar(self):
        """Render navigation controls in sidebar"""
        info = self.get_current_image_info()
        
        # Show status
        st.sidebar.write("Total files:", info["total_files"])
        st.sidebar.write("Total annotate files:", info["annotated_files"])
        st.sidebar.write("Remaining files:", info["remaining_files"])
        
        # File selector
        if st.session_state["files"]:
            st.sidebar.selectbox(
                "Files",
                st.session_state["files"],
                index=st.session_state["image_index"],
                on_change=self.go_to_image,
                key="file",
            )
        
        # Navigation buttons
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.button(label="Previous image", on_click=self.previous_image)
        with col2:
            st.button(label="Next image", on_click=self.next_image)
        
        st.sidebar.button(label="Next need annotate", on_click=self.next_annotate_file)
        st.sidebar.button(label="Refresh", on_click=self.refresh)


def create_navigation_controller(image_dir_manager: ImageDirManager) -> NavigationController:
    """Factory function to create a navigation controller"""
    return NavigationController(image_dir_manager)