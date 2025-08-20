import time
from optimizer import optimize_model
from safety_check import safety_passed

def main_loop():
    while True:
        print("🔁 Starting brain simulation cycle...")
        if safety_passed():
            optimize_model()
        else:
            print("⚠️ Safety check failed. Skipping optimization.")
        time.sleep(300)  # 5 minutes between cycles

if __name__ == "__main__":
    main_loop()