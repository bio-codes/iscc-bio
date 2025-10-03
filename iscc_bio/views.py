"""Intelligent bioimage view extraction for perceptual hashing."""

import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
from PIL import Image
from bioio import BioImage
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ViewInfo:
    """Information about an extracted view."""

    data: np.ndarray
    view_type: str  # 'mip', 'best_focus', 'representative', 'composite'
    scene: Optional[int] = None
    timepoint: Optional[int] = None
    z_plane: Optional[int] = None
    channels: Optional[List[int]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ImageAccessor:
    """Unified interface for accessing bioimage data from various sources."""

    def __init__(self, source: Union[Path, str, BioImage, Any]):
        """Initialize accessor with various source types.

        Args:
            source: Path to file, OMERO image ID, BioImage instance, or BlitzGateway image
        """
        self.source = source
        self._bioimage = None
        self._omero_image = None
        self._metadata_cache = {}

        # Detect source type
        if isinstance(source, BioImage):
            self._bioimage = source
        elif isinstance(source, (Path, str)) and Path(source).exists():
            # Local file
            self._bioimage = BioImage(source, reader=None)
        else:
            # Assume OMERO if we have an image-like object with getId method
            if hasattr(source, "getId"):
                self._omero_image = source
            else:
                self._bioimage = BioImage(source, reader=None)

    @property
    def is_omero(self) -> bool:
        """Check if this is an OMERO image."""
        return self._omero_image is not None

    @property
    def dims(self) -> Dict[str, int]:
        """Get dimension sizes."""
        if self.is_omero:
            img = self._omero_image
            return {
                "T": img.getSizeT(),
                "Z": img.getSizeZ(),
                "C": img.getSizeC(),
                "Y": img.getSizeY(),
                "X": img.getSizeX(),
            }
        else:
            dims = self._bioimage.dims
            return {
                "T": dims.T if hasattr(dims, "T") else 1,
                "Z": dims.Z if hasattr(dims, "Z") else 1,
                "C": dims.C if hasattr(dims, "C") else 1,
                "Y": dims.Y if hasattr(dims, "Y") else dims.shape[-2],
                "X": dims.X if hasattr(dims, "X") else dims.shape[-1],
            }

    @property
    def resolution_levels(self) -> List[int]:
        """Get available resolution levels."""
        if self.is_omero:
            # OMERO pyramids - simplified approach
            # Most OMERO images don't have pyramids unless very large
            # We'll just use resolution level 0 for now
            return [0]
        else:
            # BioIO resolution levels
            if hasattr(self._bioimage, "resolution_levels"):
                return self._bioimage.resolution_levels
            return [0]

    def get_plane(
        self, z: int = 0, c: int = 0, t: int = 0, resolution_level: int = -1
    ) -> np.ndarray:
        """Get a single 2D plane efficiently.

        Args:
            z: Z index
            c: Channel index
            t: Timepoint index
            resolution_level: Resolution level (-1 for lowest/fastest)

        Returns:
            2D numpy array
        """
        if self.is_omero:
            # OMERO efficient plane access
            pixels = self._omero_image.getPrimaryPixels()

            # Get plane directly (OMERO handles resolution internally)
            plane = pixels.getPlane(z, c, t)
            return np.array(plane)
        else:
            # BioIO plane access
            if resolution_level == -1 and len(self.resolution_levels) > 1:
                self._bioimage.set_resolution_level(self.resolution_levels[-1])

            # Use get_image_data for specific plane
            kwargs = {"T": t, "Z": z, "C": c}
            plane = self._bioimage.get_image_data("YX", **kwargs)
            return plane

    def get_z_stack(
        self, c: int = 0, t: int = 0, resolution_level: int = -1
    ) -> np.ndarray:
        """Get all Z planes for a channel/timepoint.

        Args:
            c: Channel index
            t: Timepoint index
            resolution_level: Resolution level (-1 for lowest)

        Returns:
            3D array (Z, Y, X)
        """
        dims = self.dims
        if dims["Z"] == 1:
            return self.get_plane(0, c, t, resolution_level)[np.newaxis, ...]

        if self.is_omero:
            # Fetch Z planes individually for OMERO
            planes = []
            for z in range(dims["Z"]):
                plane = self.get_plane(z, c, t, resolution_level)
                planes.append(plane)
            return np.stack(planes)
        else:
            # BioIO can get entire Z-stack at once
            if resolution_level == -1 and len(self.resolution_levels) > 1:
                self._bioimage.set_resolution_level(self.resolution_levels[-1])

            kwargs = {"T": t, "C": c}
            z_stack = self._bioimage.get_image_data("ZYX", **kwargs)
            return z_stack

    def get_thumbnail(self, max_size: int = 128) -> np.ndarray:
        """Get a quick thumbnail of the image.

        Args:
            max_size: Maximum dimension size

        Returns:
            2D thumbnail array
        """
        # Use middle Z, first C, first T at lowest resolution
        dims = self.dims
        z = dims["Z"] // 2 if dims["Z"] > 1 else 0

        # Get plane at lowest resolution
        plane = self.get_plane(z=z, c=0, t=0, resolution_level=-1)

        # Downsample if still too large
        if max(plane.shape) > max_size:
            pil_img = Image.fromarray(self._normalize_to_uint8(plane))
            pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            plane = np.array(pil_img)

        return plane

    def _normalize_to_uint8(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to uint8 range."""
        if data.dtype == np.uint8:
            return data

        # Adaptive percentile normalization for biological data
        p2, p98 = np.percentile(data, [2, 98])
        if p98 > p2:
            normalized = np.clip((data - p2) / (p98 - p2), 0, 1)
            # Square root scaling for better contrast
            normalized = np.sqrt(normalized)
            return (normalized * 255).astype(np.uint8)
        else:
            return np.zeros_like(data, dtype=np.uint8)


def calculate_focus_score(plane: np.ndarray) -> float:
    """Calculate focus score using variance of Laplacian.

    Args:
        plane: 2D image array

    Returns:
        Focus score (higher is sharper)
    """
    if plane.size == 0:
        return 0.0

    # Compute Laplacian
    laplacian = np.gradient(np.gradient(plane, axis=0), axis=1)
    return float(np.var(laplacian))


def calculate_entropy(plane: np.ndarray) -> float:
    """Calculate Shannon entropy of image.

    Args:
        plane: 2D image array

    Returns:
        Entropy score
    """
    if plane.size == 0:
        return 0.0

    # Convert to uint8 if needed
    if plane.dtype != np.uint8:
        p_min, p_max = plane.min(), plane.max()
        if p_max > p_min:
            plane = ((plane - p_min) / (p_max - p_min) * 255).astype(np.uint8)
        else:
            return 0.0

    # Calculate histogram
    hist, _ = np.histogram(plane, bins=256, range=(0, 256))
    hist = hist[hist > 0]  # Remove zero bins

    # Calculate entropy
    probs = hist / hist.sum()
    entropy = -np.sum(probs * np.log2(probs))

    return float(entropy)


def extract_views(
    image_source: Union[Path, str, BioImage, Any],
    max_views: int = 8,
    strategies: List[str] = None,
    min_quality_score: float = 0.1,
) -> List[ViewInfo]:
    """Extract intelligent views from bioimage for perceptual hashing.

    This function extracts up to 8 representative 2D views from complex
    multi-dimensional bioimage data. It works efficiently with both local
    files (via BioIO) and remote OMERO instances (via BlitzGateway).

    Args:
        image_source: Path to file, BioImage instance, or OMERO image object
        max_views: Maximum number of views to extract (default 8)
        strategies: List of strategies to use. Options:
                   'mip' - Maximum intensity projections
                   'best_focus' - Best focus Z-planes
                   'representative' - Representative sampling
                   'composite' - Multi-channel composites
                   Default: ['mip', 'best_focus', 'representative']
        min_quality_score: Minimum quality score for view inclusion (0-1)

    Returns:
        List of ViewInfo objects containing extracted views and metadata
    """
    start_time = time.time()

    if strategies is None:
        strategies = ["mip", "best_focus", "representative"]

    # Create unified accessor
    accessor = ImageAccessor(image_source)
    dims = accessor.dims

    logger.info(f"Extracting views from image with dimensions: {dims}")
    logger.info(f"Using strategies: {strategies}")

    views = []

    # Sample timepoints
    if dims["T"] > 1:
        t_indices = (
            [0, dims["T"] // 2, dims["T"] - 1] if dims["T"] > 2 else [0, dims["T"] - 1]
        )
    else:
        t_indices = [0]

    # MIP strategy
    if "mip" in strategies and dims["Z"] > 1:
        logger.info("Extracting maximum intensity projections")

        for t_idx in t_indices[:2]:  # Limit MIPs to save views
            for c_idx in range(min(2, dims["C"])):  # Max 2 channels for MIPs
                # Get Z-stack at low resolution for MIP
                z_stack = accessor.get_z_stack(c=c_idx, t=t_idx, resolution_level=-1)

                # Compute MIP
                mip = np.max(z_stack, axis=0)

                # Check quality using entropy
                quality = calculate_entropy(mip)
                # Use a more reasonable threshold for entropy (typically 2-8 for meaningful images)
                if quality >= max(min_quality_score, 2.0):
                    views.append(
                        ViewInfo(
                            data=mip,
                            view_type="mip",
                            timepoint=t_idx,
                            channels=[c_idx],
                            metadata={"quality_score": quality},
                        )
                    )

                if len(views) >= max_views:
                    break

            if len(views) >= max_views:
                break

    # Best focus strategy
    if "best_focus" in strategies and dims["Z"] > 3 and len(views) < max_views:
        logger.info("Finding best focus planes")

        for t_idx in t_indices[:1]:  # One timepoint for focus
            for c_idx in range(min(2, dims["C"])):
                # Evaluate focus scores across Z
                best_z = 0
                best_score = 0.0

                # Sample Z planes for focus detection
                z_sample_indices = np.linspace(
                    0, dims["Z"] - 1, min(10, dims["Z"]), dtype=int
                )

                for z_idx in z_sample_indices:
                    plane = accessor.get_plane(
                        z=z_idx, c=c_idx, t=t_idx, resolution_level=-1
                    )
                    score = calculate_focus_score(plane)

                    if score > best_score:
                        best_score = score
                        best_z = z_idx

                # Get best focus plane at higher resolution
                if best_score > min_quality_score:
                    plane = accessor.get_plane(
                        z=best_z, c=c_idx, t=t_idx, resolution_level=0
                    )

                    # Double-check entropy to ensure it has content
                    entropy = calculate_entropy(plane)
                    if entropy >= max(min_quality_score, 2.0):
                        views.append(
                            ViewInfo(
                                data=plane,
                                view_type="best_focus",
                                timepoint=t_idx,
                                z_plane=best_z,
                                channels=[c_idx],
                                metadata={
                                    "focus_score": best_score,
                                    "entropy_score": entropy,
                                },
                            )
                        )

                if len(views) >= max_views:
                    break

            if len(views) >= max_views:
                break

    # Representative sampling strategy - fast entropy-based selection with diversity
    if "representative" in strategies and len(views) < max_views:
        logger.info("Extracting representative samples with fast entropy selection")

        remaining = max_views - len(views)

        # Quick scan: sample fewer planes initially for speed
        n_samples = min(20, dims["Z"])  # Cap at 20 for speed
        z_samples = np.linspace(0, dims["Z"] - 1, n_samples, dtype=int)

        # Track best Z per channel to ensure diversity
        best_per_channel = {}  # {channel: [(z, entropy), ...]}

        for c_idx in range(min(dims["C"], 3)):  # Max 3 channels
            channel_candidates = []

            for z_idx in z_samples:
                # Quick entropy check on downsampled plane
                plane = accessor.get_plane(z=z_idx, c=c_idx, t=0, resolution_level=-1)

                # Fast entropy approximation: just check variance as proxy
                variance = np.var(plane)
                if variance > 100:  # Quick threshold
                    # Calculate real entropy only for promising planes
                    entropy = calculate_entropy(plane)
                    if entropy > 2.0:
                        channel_candidates.append((z_idx, entropy))

            if channel_candidates:
                # Sort and keep diverse Z values (at least 10 Z-planes apart)
                channel_candidates.sort(key=lambda x: x[1], reverse=True)
                selected = []
                for z, ent in channel_candidates:
                    if not selected or all(abs(z - s[0]) > 10 for s in selected):
                        selected.append((z, ent))
                        if len(selected) >= max(2, remaining // max(1, dims["C"])):
                            break
                best_per_channel[c_idx] = selected

        # Collect final views from diverse selections
        all_candidates = []
        for c_idx, z_list in best_per_channel.items():
            for z_idx, entropy in z_list:
                all_candidates.append((entropy, z_idx, c_idx))

        # Sort by entropy and add views
        all_candidates.sort(reverse=True)

        for entropy, z_idx, c_idx in all_candidates[:remaining]:
            plane = accessor.get_plane(z=z_idx, c=c_idx, t=0, resolution_level=-1)
            views.append(
                ViewInfo(
                    data=plane,
                    view_type="representative",
                    timepoint=0,
                    z_plane=z_idx,
                    channels=[c_idx],
                    metadata={"entropy_score": entropy},
                )
            )

            if len(views) >= max_views:
                break

    # Composite strategy (RGB from first 3 channels)
    if "composite" in strategies and dims["C"] >= 3 and len(views) < max_views:
        logger.info("Creating multi-channel composite")

        z_idx = dims["Z"] // 2 if dims["Z"] > 1 else 0

        # Get first 3 channels
        channels = []
        for c_idx in range(3):
            plane = accessor.get_plane(z=z_idx, c=c_idx, t=0, resolution_level=-1)
            channels.append(plane)

        # Stack as RGB
        composite = np.stack(channels, axis=-1)

        views.append(
            ViewInfo(
                data=composite,
                view_type="composite",
                timepoint=0,
                z_plane=z_idx,
                channels=[0, 1, 2],
                metadata={"composite_type": "rgb"},
            )
        )

    # Ensure we have at least one view
    if not views:
        logger.warning("No views passed quality threshold, extracting default view")
        plane = accessor.get_thumbnail()
        views.append(
            ViewInfo(
                data=plane,
                view_type="representative",
                timepoint=0,
                z_plane=0,
                channels=[0],
            )
        )

    elapsed = time.time() - start_time
    logger.info(f"Extracted {len(views)} views in {elapsed:.2f} seconds")

    return views


def views_to_thumbnails(
    views: List[ViewInfo],
    max_size: Tuple[int, int] = (512, 512),
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Convert extracted views to thumbnail images.

    Args:
        views: List of ViewInfo objects from extract_views
        max_size: Maximum thumbnail size (width, height)
        output_dir: Directory to save thumbnails (optional)

    Returns:
        List of paths to saved thumbnails (if output_dir provided)
    """
    thumbnails = []

    for i, view in enumerate(views):
        # Normalize data
        data = view.data
        if data.dtype != np.uint8:
            p2, p98 = np.percentile(data, [2, 98])
            if p98 > p2:
                data = np.clip((data - p2) / (p98 - p2), 0, 1)
                data = np.sqrt(data)  # Better contrast
                data = (data * 255).astype(np.uint8)
            else:
                data = np.zeros_like(data, dtype=np.uint8)

        # Create PIL image
        if data.ndim == 2:
            pil_img = Image.fromarray(data, mode="L")
        elif data.ndim == 3 and data.shape[-1] == 3:
            pil_img = Image.fromarray(data, mode="RGB")
        else:
            # Fallback to grayscale
            if data.ndim == 3:
                data = data[..., 0]
            pil_img = Image.fromarray(data, mode="L")

        # Resize
        pil_img.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Save if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)

            # Generate filename
            filename = f"view_{i:02d}_{view.view_type}"
            if view.timepoint is not None:
                filename += f"_t{view.timepoint}"
            if view.z_plane is not None:
                filename += f"_z{view.z_plane}"
            if view.channels:
                filename += f"_c{''.join(map(str, view.channels))}"
            filename += ".png"

            output_path = output_dir / filename
            pil_img.save(output_path)
            thumbnails.append(output_path)

            logger.info(f"Saved thumbnail: {output_path}")

    return thumbnails
