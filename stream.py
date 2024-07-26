import time
import subprocess
import mujoco
import mujoco.viewer

# FFmpeg command to capture the screen and stream to local RTMP server without audio
ffmpeg_command = [
    '../ffmpeg/ffmpeg',
    '-f', 'x11grab',  # Capture the X11 screen
    '-s', '1000x1000',  # Set the resolution (adjust to your screen resolution)
    '-i', '172.22.144.1:0',  # Capture from the default display
    # '-c:v', 'libx264',  # Use the H.264 codec
    # '-preset', 'veryfast',  # Set the encoding speed
    '-b:v', '3000k',  # Set the video bitrate
    '-pix_fmt', 'yuv420p',  # Ensure the pixel format is correct
    '-g', '60',  # GOP size (adjust based on your needs)
    '-f', 'flv',  # Output format
    'rtmp://localhost/live/stream'  # Local RTMP URL
]

# Start the FFmpeg process
ffmpeg_process = subprocess.Popen(ffmpeg_command)

# Load the MuJoCo model
m = mujoco.MjModel.from_xml_path('customant.xml')
d = mujoco.MjData(m)

# Launch the MuJoCo viewer
with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after 500 seconds.
    start = time.time()
    while viewer.is_running() and time.time() - start < 500:
        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(m, d)

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# Terminate the FFmpeg process after the viewer is closed
ffmpeg_process.terminate()
