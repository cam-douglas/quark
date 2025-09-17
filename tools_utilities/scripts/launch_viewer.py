"""
Launcher for the Quark Brain Simulation with MuJoCo Viewer.

This script starts the main brain simulation as a detached subprocess,
ensuring that the MuJoCo viewer can launch without being interrupted
by the parent process exiting.
"""
import subprocess
import os
import sys
import shlex

def main():
    """Launches the brain_main.py script in a new terminal window."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Define the command to run - use mjpython for MuJoCo viewer support on macOS
    python_executable = "mjpython"  # Required for MuJoCo viewer on macOS
    main_script = "-m"
    script_path = "brain.brain_main"

    # Set the model XML environment variable
    model_xml_path = os.environ.get(
        "QUARK_MODEL_XML",
        os.path.join(project_root, "brain/architecture/embodiment/humanoid.xml")
    )

    if not os.path.exists(model_xml_path):
        print(f"❌ ERROR: MuJoCo model file not found at: {model_xml_path}")
        print("Please ensure the file exists or set the QUARK_MODEL_XML environment variable.")
        sys.exit(1)

    # Construct the full command string to be executed in the new terminal
    # We must activate the virtual environment and then run the script.
    venv_activate_path = os.path.join(project_root, ".venv", "bin", "activate")

    # Ensure the python executable path is correctly quoted if it contains spaces
    safe_python_executable = shlex.quote(python_executable)

    command_to_run = (
        f"source {shlex.quote(venv_activate_path)}; "
        f"export QUARK_MODEL_XML={shlex.quote(model_xml_path)}; "
        f"cd {shlex.quote(project_root)}; "
        f"{safe_python_executable} {main_script} {script_path} --viewer"
    )

    print("🚀 Launching Quark Brain Simulation with MuJoCo Viewer...")
    print(f"   Model: {model_xml_path}")
    print("   Using mjpython for MuJoCo viewer support on macOS")
    print(f"   Executing command:\n   {command_to_run}")

    # For macOS, use `open` to launch a new terminal instance.
    # This ensures the GUI application has the correct environment.
    try:
        subprocess.run(
            ['open', '-a', 'Terminal', '-n', '--args', 'bash', '-c', command_to_run],
            check=True
        )
        print("\n✅ A new terminal window has been opened for the brain simulation.")
        print("   🖥️  The MuJoCo viewer should appear showing the 3D humanoid model")
        print("   🧠 The full Quark brain simulation will be running")
        print("   🗣️  Press Ctrl+C in that terminal to pause and chat with Quark!")
    except FileNotFoundError:
        print("❌ ERROR: `open` command not found. This launcher is designed for macOS.")
        print("   Please run the following command manually in a new terminal:")
        print(f"   {command_to_run}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to launch new terminal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
