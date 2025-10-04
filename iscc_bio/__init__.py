import os
from .utils import find_java_home
from loguru import logger

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
    """Silence Java logging from bioformats loci package."""
    try:
        import scyjava
        import jpype

        scyjava.config.endpoints.append("ome:formats-gpl:6.7.0")
        scyjava.start_jvm()
        loci = jpype.JPackage("loci")
        loci.common.DebugTools.setRootLevel("OFF")
    except Exception:
        # Silently fail if bioformats is not available or already initialized
        pass


_silence_bioformats_logging()

__version__ = "0.1.0"
