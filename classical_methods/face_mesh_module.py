import cv2
import mediapipe as mp
import time
import numpy as np
from typing import Tuple, Union
import math


class FaceMeshDetector():
    def __init__(self, static_image_mode: bool = False, max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5) -> None:
        """
        Initialize class variables.

        Args:
            static_image_mode (bool): Whether to treat input images as static images or a continuous video stream.
                False: Treat input images as a continuous video stream (a.k.a. video mode). This mode will try to detect
                    faces in the first input image, and upon a successful detection, subsequent detections will be
                    made by attempting to track the face from the previous frame. If tracking is successful, computation
                    for the frames after the first one should be faster than running face detection on each individual
                    input image. Use this mode when you want to track faces across images in a video stream or for
                    live, real-time face recognition.
                True: Treat input images as static images (a.k.a. image mode). This mode will treat each input image as
                    an independent image and will not try to detect or track faces across images. Use this mode when you
                    want to run face detection/face landmark detection on a set of non-continuous, unrelated
                    input images.
            max_num_faces (int): Maximum number of faces to detect.
            min_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for face detection to be considered
                successful.
            min_tracking_confidence (float): Minimum confidence value ([0.0, 1.0]) for face tracking to be considered
                successful.
        """
        self.mp_draw = mp.solutions.drawing_utils # type: ignore
        self.mp_face_mesh = mp.solutions.face_mesh # type: ignore
        # Store initialization parameters for lazy loading
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.face_mesh = None  # Initialize lazily to avoid resource conflicts
        self.drawing_spec_landmark = self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)
        self.drawing_spec_connection = self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
    
    def _initialize_face_mesh(self):
        """Lazy initialization of face mesh to avoid resource conflicts in multiprocessing."""
        if self.face_mesh is None:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=self.static_image_mode,
                max_num_faces=self.max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
    
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except:
                pass  # Ignore cleanup errors

    def find_face_mesh(self, image: np.ndarray, draw: bool = False) -> Tuple[bool, np.ndarray]:
        """
        Detect face mesh and return facial landmarks for the given image.

        Args:
            image (np.ndarray): The input image in BGR format.
            draw (bool): Whether to draw the face mesh on the image.

        Returns:
            tuple[bool, np.ndarray]: A tuple containing:
                - face_detected (bool): Indicates whether a face was detected in the image.
                - landmarks_pixels (np.ndarray): An array of shape (478, 2) representing the pixel coordinates (x, y)
                  for each of the 478 total face landmarks detected. The i-th row corresponds to the i-th landmark
                  (zero-indexed, so row 0 is landmark 1).

        Note:
            The input image should be in BGR format, as OpenCV loads images/videos in BGR format by default.
            The returned landmarks_pixels array contains the pixel coordinates (x, y) for each face landmark.

        References:
            - Face Landmarks Key: https://github.com/tensorflow/tfjs-models/blob/master/face-landmarks-detection/mesh_map.jpg
        """
        # Initialize face mesh lazily to avoid resource conflicts
        self._initialize_face_mesh()
        
        # Get image dimensions
        image_height, image_width, image_channels = image.shape

        # CV2 loads images/videos in BGR format, convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            results = self.face_mesh.process(image_rgb)
        except Exception as e:
            print(f"Warning: Face mesh processing failed: {e}")
            return False, np.zeros((478, 2), dtype="int")
        # print(results)

        # landmarks_pixels is an array of shape (478, 2) with x, y coordinates (as pixels) for each landmark
        # landmarks_pixels = np.zeros((478, 2), dtype="int")
        landmarks_pixels = np.zeros((478, 2), dtype="int")

        face_detected = False

        if results.multi_face_landmarks:
            # Face(s) were detected
            # Loop through each face detected

            face_detected = True

            for face_landmarks in results.multi_face_landmarks:
                # face_landmarks is a list of landmarks for face

                if draw:
                    self.mp_draw.draw_landmarks(image, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS, self.drawing_spec_landmark,
                                                self.drawing_spec_connection)

                # Loop through each landmark
                for id, landmark in enumerate(face_landmarks.landmark):
                    # There are 478 landmarks in total, with x, y, z normalized coordinates
                    # print(id, landmark)

                    # Convert normalized coordinates to pixel coordinates (NOTE: z is currently unused)
                    x, y = int(landmark.x * image_width), int(landmark.y * image_height)
                    # x, y = self._normalized_to_pixel_coordinates(landmark.x, landmark.y, image_width, image_height)
                    # print(id, x, y)

                    # Store pixel coordinates in array
                    landmarks_pixels[id] = (x, y)

                    # if draw:
                    #     # Display the landmark index on the visualization
                    #     cv2.putText(image, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        return face_detected, landmarks_pixels
    

    def _normalized_to_pixel_coordinates(self, normalized_x: float, normalized_y: float, image_width: int, image_height: int) -> Tuple[int, int]:
        """
        Converts normalized value pair to pixel coordinates.
        """

        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            print(f"WARNING: Normalized value pair ({normalized_x}, {normalized_y}) is not in valid range!")

            return -1, -1
        
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)

        return x_px, y_px
    

    def _draw_ROI_bounding_box(self, bounding_box_pixels: np.ndarray, frame_grayscale_rgb: np.ndarray, roi_name: str) -> None:
        """
        Draws the bounding box for an ROI on the frame image along with the ROI name.

        Args:
            bounding_box_pixels (np.ndarray): An array of shape (n, 2), where n is the number of landmarks
                that form the bounding box for the requested ROI. Each row represents the (x, y)
                coordinates of a landmark pixel.
            frame_grayscale_rgb (np.ndarray): The frame image in grayscale RGB format.
            roi_name (str): The name of the requested ROI. This is the text that will be displayed
                above the bounding box.

        Returns:
            None
        
        NOTE: After drawing the bounding box and displaying the image, the function will wait for
        the user to press any key before continuing.

        NOTE: The usage of this function requires multi-threading to be disabled (aka num_threads=1).
        """

        # Draw the bounding box on the frame image
        cv2.polylines(frame_grayscale_rgb, [bounding_box_pixels], isClosed=True, color=(0, 0, 255), thickness=2)

        # Add roi_name as text above the bounding box
        text_position = (bounding_box_pixels[0, 0], bounding_box_pixels[0, 1] - 10)
        cv2.putText(frame_grayscale_rgb, roi_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the image
        cv2.imshow("ROI Bounding Boxes", frame_grayscale_rgb)
        cv2.waitKey(0)
    
        return
    

    def _get_ROI_bounding_box_pixels(self, landmarks_pixels: np.ndarray, roi_name: str, landmark_indices) -> np.ndarray:
        """
        Takes in the pixel coordinates of all face landmarks and returns the pixel coordinates
        of the face landmarks that represent the bounding box for the requested ROI.

        Args:
            landmarks_pixels (np.ndarray): An array of shape (468, 2) representing the pixel coordinates (x, y)
                for each of the 468 total face landmarks detected. The i-th row corresponds to the i-th landmark
                (zero-indexed, so row 0 is landmark 1).
            roi_name (str): The name of the requested ROI. Must be one of the predefined ROIs in
                `self.face_roi_definitions`.

        Returns:
            np.ndarray: An array of shape (n, 2), where n is the number of landmarks
                that form the bounding box for the requested ROI. Each row represents the (x, y)
                coordinates of a landmark pixel.

        Raises:
            KeyError: If the provided roi_name does not match any of the predefined ROIs.

        Note:
            The returned bounding_box_pixels is in the same format as the input landmarks_pixels,
            with each array of shape (2,) representing the (x, y) coordinates of a landmark pixel.
        """
        bounding_box_pixels = np.array([])
        
        bounding_box_pixels = landmarks_pixels[landmark_indices]

        return bounding_box_pixels


def main():
    input_video = "learning_mediapipe/videos/test_video.mp4"
    # input_video = "learning_mediapipe/videos/test_video_2.mp4"
    video = cv2.VideoCapture(input_video)
    if not video.isOpened():
        print("Error opening video file:", input_video)
        exit(1)

    previous_time = 0
    start_time = time.time()

    detector = FaceMeshDetector()

    # Loop through video frames
    while video.isOpened():
        # Get a frame of video
        ret, frame = video.read()
        if not ret:
            break

        # Get pixel locations of all face landmarks
        face_detected, landmarks_pixels = detector.find_face_mesh(image=frame, draw=True)
        
        # if face_detected:
        #     # Do something with the landmarks
        #     roi_cheek_n_nose_landmarks = np.array([31, 228, 229, 230, 231, 232, 233, 245, 465, 453, 452, 451, 450, 449, 448, 340, 345, 352, 376, 411, 427, 426, 294, 278, 360, 363, 281, 5, 51, 134, 131, 102, 203, 206, 207, 187, 147, 123, 116, 111])

        # Calculate and overlay FPS

        current_time = time.time()
        # FPS = (# frames processed (1)) / (# seconds taken to process those frames)
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # Display frame

        cv2.imshow("Image", frame)
        cv2.waitKey(1)
    
    end_time = time.time()
    average_fps = (video.get(cv2.CAP_PROP_FRAME_COUNT) - 1) / (end_time - start_time)
    print(f"Average FPS: {average_fps}")

    video.release()


if __name__ == "__main__":
    main()