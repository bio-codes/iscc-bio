import os
from .utils import find_java_home
from loguru import logger as log

if not os.environ.get("JAVA_HOME"):
    java_home = find_java_home()
    if java_home:
        os.environ["JAVA_HOME"] = java_home
        log.debug(f"Auto-detected JAVA_HOME: {java_home}")
    else:
        log.warning(
            "Warning: Could not auto-detect Java installation. bioio-bioformats may not work."
        )
        log.warning("Please install Java and/or set JAVA_HOME environment variable.")

__version__ = "0.1.0"
