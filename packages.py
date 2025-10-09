import os
import subprocess
import sys

# --- AUTO-INSTALL SELENIUM DEPENDENCIES ---
def ensure_package(package):
    try:
        __import__(package)
    except ImportError:
        print(f"📦 Installing missing package: {package} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure key packages
for pkg in ["selenium", "webdriver_manager", "chromium-browser"]:
    ensure_package(pkg)

# Ensure Chrome is available (for Linux environments)
if not os.path.exists("/usr/bin/google-chrome") and not os.path.exists("/usr/bin/chromium-browser"):
    try:
        print("⚙️ Installing Google Chrome ...")
        subprocess.run([
            "apt-get", "update", "-y"
        ], check=True)
        subprocess.run([
            "apt-get", "install", "-y", "google-chrome-stable"
        ], check=True)
    except Exception as e:
        print(f"⚠️ Could not auto-install Chrome: {e}")
