import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np

# Global variables that will be initialized in main()
frames = None
is_recording = False
video_writer = None
left_frame_idx = 0
right_frame_idx = 0
fig = None
ax = None
output_video_path = "output.mp4"
skeleton_connections = [
    ("NOSE", "LEFT_EYE_INNER"), ("LEFT_EYE_INNER", "LEFT_EYE"), ("LEFT_EYE", "LEFT_EYE_OUTER"),
    ("NOSE", "RIGHT_EYE_INNER"), ("RIGHT_EYE_INNER", "RIGHT_EYE"), ("RIGHT_EYE", "RIGHT_EYE_OUTER"),
    ("NOSE", "LEFT_SHOULDER"), ("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"),
    ("NOSE", "RIGHT_SHOULDER"), ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"), ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"), ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE")
]

def adjust_orientation(keypoints):
    adjusted_keypoints = {}
    for part, (x, y) in keypoints.items():
        adjusted_keypoints[part] = (x, -y)
    return adjusted_keypoints

def extract_keypoints(frame_keypoints):
    keypoints = {}
    for part, coords in frame_keypoints.items():
        keypoints[part] = (coords['x'], coords['y'])
    return adjust_orientation(keypoints)

def calculate_bounding_box(keypoints):
    coords = np.array(list(keypoints.values()))
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    return x_min, x_max, y_min, y_max

def draw_head(ax, head_center, head_radius=0.02, color='blue'):
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi, 10)
    x = head_radius * np.outer(np.cos(u), np.sin(v)) + head_center[0]
    y = head_radius * np.outer(np.sin(u), np.sin(v)) + head_center[1]
    # Plotting a surface to approximate a sphere
    # ax.plot_surface(x, y, np.zeros_like(x), color=color)

def plot_human_figure(ax, keypoints, color='blue', label="Frame"):
    keypoints_coords = {part: (kp[0], kp[1]) for part, kp in keypoints.items()}
    # Plot skeleton connections
    for connection in skeleton_connections:
        part_a, part_b = connection
        if part_a in keypoints_coords and part_b in keypoints_coords:
            ax.plot(
                [keypoints_coords[part_a][0], keypoints_coords[part_b][0]],
                [keypoints_coords[part_a][1], keypoints_coords[part_b][1]],
                color=color
            )
    # Draw the head (using the NOSE as a center point)
    if "NOSE" in keypoints_coords:
        draw_head(ax, keypoints_coords["NOSE"], color=color)

def update_plot(frame_idx1, frame_idx2, preserve_view=False):
    global ax, fig
    if preserve_view:
        current_elev = ax.elev
        current_azim = ax.azim

    ax.clear()
    keypoints1 = extract_keypoints(frames[frame_idx1])
    keypoints2 = extract_keypoints(frames[frame_idx2])

    plot_human_figure(ax, keypoints1, color='blue', label=f"Frame {frame_idx1}")
    plot_human_figure(ax, keypoints2, color='red', label=f"Frame {frame_idx2}")

    x_min_1, x_max_1, y_min_1, y_max_1 = calculate_bounding_box(keypoints1)
    x_min_2, x_max_2, y_min_2, y_max_2 = calculate_bounding_box(keypoints2)

    x_min, x_max = min(x_min_1, x_min_2), max(x_max_1, x_max_2)
    y_min, y_max = min(y_min_1, y_min_2), max(y_max_1, y_max_2)

    ax.set_xlim(x_min - 0.1, x_max + 0.1)
    ax.set_ylim(y_min - 0.1, y_max + 0.1)

    ax.set_title(f"Comparison of Frame {frame_idx1} (blue) and Frame {frame_idx2} (red)")

    if preserve_view:
        ax.view_init(elev=current_elev, azim=current_azim)

    fig.canvas.draw()

def start_stop_recording():
    global is_recording, video_writer, fig, output_video_path
    if is_recording:
        is_recording = False
        video_writer.release()
        video_writer = None
        print("Recording stopped.")
    else:
        is_recording = True
        width, height = fig.canvas.get_width_height()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))
        print("Recording started.")

def record_frame():
    global video_writer, fig
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    img = img.reshape(height, width, 3)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    video_writer.write(img_bgr)

def on_trackbar_left(val):
    global left_frame_idx, right_frame_idx
    left_frame_idx = val
    update_plot(left_frame_idx, right_frame_idx)
    if is_recording:
        record_frame()

def on_trackbar_right(val):
    global left_frame_idx, right_frame_idx
    right_frame_idx = val
    update_plot(left_frame_idx, right_frame_idx)
    if is_recording:
        record_frame()

def on_record_button(val):
    if val == 1:
        start_stop_recording()
        cv2.setTrackbarPos('Record', 'Trackbar Window', 0)

def main():
    global frames, fig, ax, left_frame_idx, right_frame_idx

    # Load keypoints data from JSON file
    json_file = "/Users/danielfonseca/repos/baseball_pose/pose_jsons/acura_1.json"
    with open(json_file, 'r') as f:
        keypoints_data = json.load(f)

    # Extract frames (each frame's keypoints)
    frames = [frame_data['keypoints'] for frame_data in keypoints_data]

    # Create the Matplotlib figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Create the OpenCV window and add trackbars
    cv2.namedWindow('Trackbar Window')
    cv2.createTrackbar('Left Frame', 'Trackbar Window', 0, len(frames) - 1, on_trackbar_left)
    cv2.createTrackbar('Right Frame', 'Trackbar Window', 0, len(frames) - 1, on_trackbar_right)
    cv2.createTrackbar('Record', 'Trackbar Window', 0, 1, on_record_button)

    # Initial plot of frame 0 for both left and right
    update_plot(0, 0, preserve_view=False)
    plt.show()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
