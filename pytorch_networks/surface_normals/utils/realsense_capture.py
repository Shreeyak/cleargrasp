# On macOS, you'll need to build the wrapper and export PYTHONPATH=$PYTHONPATH:/path/to/librealsense/build/wrappers/python/Debug

COLOR_SAVE = 'np_color_data.npy'
DEPTH_SAVE = 'np_depth_data.npy'


import pyrealsense2 as rs
import numpy as np

# Give time for auto-exposure to kick in
CAPTURES_BEFORE_SAVING = 10

# Create a context object. This object owns the handles to all connected realsense devices
rs_pipeline = rs.pipeline()
#Create a config and configure the pipeline to stream
rs_config = rs.config()
rs_config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
rs_config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
rs_profile = rs_pipeline.start(rs_config)
rs_depth_sensor = rs_profile.get_device().first_depth_sensor()
rs_depth_scale = rs_depth_sensor.get_depth_scale()
print("Depth Scale is: " , rs_depth_scale)
align_to = rs.stream.color
align = rs.align(align_to)

np_depth_data = None
np_color_data = None
images_captured = 0
while True:
    # Create a pipeline object. This object configures the streaming camera and owns it's handle
    frames = rs_pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()
    if not aligned_depth_frame or not color_frame: # doesn't count as a capture
        continue
    elif images_captured < CAPTURES_BEFORE_SAVING:
        images_captured = images_captured + 1
    else:
        raw_depth_data = np.asanyarray(aligned_depth_frame.get_data())
        np_depth_data = (raw_depth_data * rs_depth_scale).astype(np.float32)
        np_color_data = np.asanyarray(color_frame.get_data())
        break

rs_pipeline.stop()

print("Image captured.  Saving.")

print("np_depth_data.shape =", np_depth_data.shape,
      "; np_depth_data.dtype =", np_depth_data.dtype,
      "; avg(np_depth_data) =", np.average(np_depth_data),
      "; median(np_depth_data) =", np.median(np_depth_data),
#      "; mode(np_depth_data) =", np.mode(np_depth_data),
      "; range(np_depth_data) =", np.amin(np_depth_data), "-", np.amax(np_depth_data)
)
print("np_color_data.shape =", np_color_data.shape,
      "; np_color_data.dtype =", np_color_data.dtype,
      "; avg(np_color_data) =", np.average(np_color_data),
      "; median(np_color_data) =", np.median(np_color_data),
#      "; mode(np_color_data) =", np.mode(np_color_data),
      "; range(np_color_data) =", np.amin(np_color_data), "-", np.amax(np_color_data)
)
np.save(COLOR_SAVE, np_color_data)
np.save(DEPTH_SAVE, np_depth_data)
