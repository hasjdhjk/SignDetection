import paramiko
import pandas as pd
import keyboard
import time

# Raspberry Pi credentials
RASPBERRY_PI_IP = "192.168.137.1"  # Replace with your Pi's IP in USB gadget mode
PI_USERNAME = "pi"
PI_PASSWORD = "raspberry"

# Load labels from the Excel sheet
excel_file = "labels.xlsx"  # Replace with the path to your Excel file
labels_df = pd.read_excel(excel_file)
labels = labels_df['Label'].tolist()  # Assuming the column is named 'Label'

# Establish SSH connection
def connect_to_pi():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(RASPBERRY_PI_IP, username=PI_USERNAME, password=PI_PASSWORD)
    return ssh

# Send a command to the Raspberry Pi
def send_command(ssh, command):
    stdin, stdout, stderr = ssh.exec_command(command)
    stdout.channel.recv_exit_status()  # Wait for the command to finish
    return stdout.read().decode('utf-8'), stderr.read().decode('utf-8')

# Main script
try:
    ssh = connect_to_pi()
    print("Connected to Raspberry Pi.")
    print("Press Spacebar to start/stop recording. Press 'q' to quit.")

    recording = False
    current_label_index = 0

    while True:
        if keyboard.is_pressed('space'):
            if not recording:
                # Check if labels are available
                if current_label_index >= len(labels):
                    print("No more labels available in the Excel sheet.")
                    break

                # Start recording on Raspberry Pi
                label = labels[current_label_index]
                command = f"python3 ~/start_recording.py start {label}"
                _, err = send_command(ssh, command)
                if err:
                    print(f"Error starting recording: {err}")
                else:
                    recording = True
                    print(f"Started recording: {label}")
                time.sleep(0.5)  # Prevent immediate re-trigger
            else:
                # Stop recording on Raspberry Pi
                command = "python3 ~/start_recording.py stop"
                _, err = send_command(ssh, command)
                if err:
                    print(f"Error stopping recording: {err}")
                else:
                    recording = False
                    print(f"Stopped recording.")
                    current_label_index += 1
                time.sleep(0.5)  # Prevent immediate re-trigger

        if keyboard.is_pressed('q'):
            print("Exiting...")
            break
finally:
    ssh.close()
    print("Disconnected from Raspberry Pi.")
