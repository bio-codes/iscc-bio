import os
import sys
from .utils import find_java_home
from loguru import logger

# Configure loguru format to show module:line instead of full path
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{module}:{line}</cyan> - <level>{message}</level>",
)

# Suppress noisy logging from third-party libraries using loguru
logger.disable("omero")
logger.disable("omero.gateway")

if not os.environ.get("JAVA_HOME"):
    java_home = find_java_home()
    if java_home:
        os.environ["JAVA_HOME"] = java_home
        logger.debug(f"Auto-detected JAVA_HOME: {java_home}")
    else:
        logger.warning(
            "Warning: Could not auto-detect Java installation. bioio-bioformats may not work."
        )
        logger.warning("Please install Java and/or set JAVA_HOME environment variable.")


def _silence_bioformats_logging():
    """Silence Java logging from bioformats loci package.

    Called lazily on first bioformats usage to avoid JVM initialization
    during module import.
    """
    try:
        import scyjava
        import jpype

        # Only start if not already started
        if not jpype.isJVMStarted():
            scyjava.config.endpoints.append("ome:formats-gpl:6.7.0")
            scyjava.start_jvm()
            loci = jpype.JPackage("loci")
            loci.common.DebugTools.setRootLevel("OFF")
    except Exception:
        # Silently fail if bioformats is not available or already initialized
        pass


# Don't initialize JVM at import time - let bioformats do it lazily
# _silence_bioformats_logging()

__version__ = "0.2.0"
