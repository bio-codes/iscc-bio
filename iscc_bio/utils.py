import os
import shutil
import platform


def find_java_home():
    """Dynamically find Java installation directory."""
    # Check if JAVA_HOME is already set
    java_home = os.environ.get("JAVA_HOME")
    if java_home and os.path.exists(java_home):
        return java_home

    # Try to find Java using shutil.which (cross-platform)
    java_exe = shutil.which("java")
    if java_exe:
        # Resolve symlinks to get the actual Java location
        java_exe = os.path.realpath(java_exe)

        # Navigate up from java executable to find JAVA_HOME
        # Typically: .../Java/jdk-XX/bin/java[.exe] or .../Java/jre-XX/bin/java[.exe]
        java_bin = os.path.dirname(java_exe)
        java_home = os.path.dirname(java_bin)

        # Verify this looks like a valid Java home
        java_binary = "java.exe" if platform.system() == "Windows" else "java"
        if os.path.exists(os.path.join(java_home, "bin", java_binary)):
            return java_home

    # Try common installation paths as fallback
    if platform.system() == "Windows":
        common_paths = [
            r"C:\Program Files\Java",
            r"C:\Program Files (x86)\Java",
            r"C:\ProgramData\Oracle\Java",
        ]

        for base_path in common_paths:
            if os.path.exists(base_path):
                # Look for JDK/JRE folders
                for folder in os.listdir(base_path):
                    java_path = os.path.join(base_path, folder)
                    if os.path.exists(os.path.join(java_path, "bin", "java.exe")):
                        return java_path

    return None
