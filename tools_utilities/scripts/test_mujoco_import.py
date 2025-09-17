"""A simple script to test if MuJoCo can be imported correctly."""
import sys

def test_mujoco_import():
    """Attempts to import mujoco and mujoco.viewer."""
    print("--- Starting MuJoCo Import Test ---")
    try:
        import mujoco
        print("✅ SUCCESS: `mujoco` module imported.")

        import mujoco.viewer
        print("✅ SUCCESS: `mujoco.viewer` submodule imported.")

        print("\nCONCLUSION: MuJoCo installation appears to be working correctly.")
        return 0

    except ImportError as e:
        print(f"❌ FAILURE: Could not import mujoco. Error: {e}")
        print("\nCONCLUSION: The MuJoCo library is likely missing or not installed correctly.")
        print("RECOMMENDATION: Try reinstalling with `pip install mujoco`.")
        return 1

    except Exception as e:
        print(f"❌ FAILURE: An unexpected error occurred during import: {e}")
        print("\nCONCLUSION: There is a deeper issue with the MuJoCo installation or its dependencies.")
        return 1

if __name__ == "__main__":
    sys.exit(test_mujoco_import())
