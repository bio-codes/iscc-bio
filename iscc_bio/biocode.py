"""Bioimage fingerprinting combining ISCC-SUM and ISCC-IMAGE codes.

This module implements a sophisticated bioimage fingerprinting system that:
1. Generates ISCC-SUM hash over normalized pixel content during incremental processing
2. Extracts ~5 representative views per scene during the same pass
3. Generates ISCC-IMAGE codes for each view
4. Combines view codes into a global ISCC-MIXED descriptor
"""

import iscc_sum
import iscc_core as ic
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class CandidateView:
    """Candidate view during incremental processing."""

    data: np.ndarray  # 384x384 cached view
    z: int
    c: int
    t: int
    quality_score: float
    view_type: str  # 'focus', 'entropy', 'middle'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneFingerprint:
    """Complete fingerprint for a bioimage scene."""

    scene_id: str
    iscc_sum: str
    views: List[Dict[str, Any]]  # Selected views with ISCC-IMAGE codes
    iscc_mixed: Optional[str] = None


def _plane_to_canonical_bytes(plane: np.ndarray) -> bytes:
    """Convert a 2D plane to canonical byte representation.

    Uses big-endian byte order for compatibility with OMERO.

    Args:
        plane: 2D NumPy array representing a single plane

    Returns:
        Bytes in big-endian format
    """
    if plane.ndim != 2:
        raise ValueError(f"Expected 2D plane, got {plane.ndim}D")

    flat = plane.flatten(order="C")

    if flat.dtype.byteorder == ">" or (
        flat.dtype.byteorder == "=" and np.little_endian
    ):
        canonical_bytes = flat.astype(f">{flat.dtype.char}", copy=False).tobytes()
    else:
        canonical_bytes = flat.astype(f">{flat.dtype.char}").tobytes()

    return canonical_bytes


def _calculate_entropy(plane: np.ndarray) -> float:
    """Calculate Shannon entropy of a plane."""
    if plane.size == 0:
        return 0.0

    if plane.dtype != np.uint8:
        p_min, p_max = plane.min(), plane.max()
        if p_max > p_min:
            plane = ((plane - p_min) / (p_max - p_min) * 255).astype(np.uint8)
        else:
            return 0.0

    hist, _ = np.histogram(plane, bins=256, range=(0, 256))
    hist = hist[hist > 0]

    if len(hist) == 0:
        return 0.0

    probs = hist / hist.sum()
    entropy = -np.sum(probs * np.log2(probs))

    return float(entropy)


def _calculate_focus_score(plane: np.ndarray) -> float:
    """Calculate focus score using variance of Laplacian."""
    if plane.size == 0:
        return 0.0

    laplacian = np.gradient(np.gradient(plane, axis=0), axis=1)
    return float(np.var(laplacian))


def _resize_to_cache(plane: np.ndarray, target_size: int = 384) -> np.ndarray:
    """Resize plane to cache size for later view extraction."""
    if plane.dtype != np.uint8:
        p_min, p_max = plane.min(), plane.max()
        if p_max > p_min:
            plane_norm = ((plane - p_min) / (p_max - p_min) * 255).astype(np.uint8)
        else:
            plane_norm = np.zeros_like(plane, dtype=np.uint8)
    else:
        plane_norm = plane

    pil_img = Image.fromarray(plane_norm, mode="L")
    pil_img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

    return np.array(pil_img)


def _prepare_for_iscc(image: np.ndarray) -> np.ndarray:
    """Prepare image for ISCC Image-Code generation.

    Args:
        image: 2D grayscale image

    Returns:
        1024-element uint8 array (32x32 flattened)
    """
    if image.dtype != np.uint8:
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            image = np.zeros_like(image, dtype=np.uint8)

    pil_img = Image.fromarray(image, mode="L")
    pil_resized = pil_img.resize((32, 32), Image.Resampling.BILINEAR)

    return np.array(pil_resized).flatten()


class ViewSelector:
    """Selects representative views during incremental processing."""

    def __init__(self, max_views: int = 5, cache_size: int = 384):
        self.max_views = max_views
        self.cache_size = cache_size
        self.candidates: List[CandidateView] = []
        self.best_focus_per_channel: Dict[int, Tuple[float, CandidateView]] = {}
        self.best_entropy_per_channel: Dict[int, Tuple[float, CandidateView]] = {}
        self.middle_z_cached = False

    def process_plane(
        self, plane: np.ndarray, z: int, c: int, t: int, dims: Dict[str, int]
    ) -> None:
        """Process a plane during incremental scan.

        Args:
            plane: 2D plane data
            z: Z index
            c: Channel index
            t: Timepoint index
            dims: Dimension sizes
        """
        # Only process first timepoint for now
        if t > 0:
            return

        # Calculate quality metrics
        entropy = _calculate_entropy(plane)
        focus_score = _calculate_focus_score(plane)

        # Cache middle Z plane
        middle_z = dims["Z"] // 2
        if z == middle_z and not self.middle_z_cached:
            cached = _resize_to_cache(plane, self.cache_size)
            candidate = CandidateView(
                data=cached,
                z=z,
                c=c,
                t=t,
                quality_score=entropy,
                view_type="middle",
                metadata={"entropy": entropy, "focus": focus_score},
            )
            self.candidates.append(candidate)
            self.middle_z_cached = True

        # Track best focus per channel
        if (
            c not in self.best_focus_per_channel
            or focus_score > self.best_focus_per_channel[c][0]
        ):
            cached = _resize_to_cache(plane, self.cache_size)
            candidate = CandidateView(
                data=cached,
                z=z,
                c=c,
                t=t,
                quality_score=entropy,
                view_type="focus",
                metadata={"entropy": entropy, "focus": focus_score},
            )
            self.best_focus_per_channel[c] = (focus_score, candidate)

        # Track best entropy per channel
        if entropy > 3.0:  # Minimum threshold
            if (
                c not in self.best_entropy_per_channel
                or entropy > self.best_entropy_per_channel[c][0]
            ):
                cached = _resize_to_cache(plane, self.cache_size)
                candidate = CandidateView(
                    data=cached,
                    z=z,
                    c=c,
                    t=t,
                    quality_score=entropy,
                    view_type="entropy",
                    metadata={"entropy": entropy, "focus": focus_score},
                )
                self.best_entropy_per_channel[c] = (entropy, candidate)

    def select_final_views(self) -> List[CandidateView]:
        """Select final representative views from candidates."""
        selected = []

        # Add best focus views
        for _, (_, candidate) in sorted(self.best_focus_per_channel.items()):
            if len(selected) < self.max_views:
                selected.append(candidate)

        # Add best entropy views if different
        for c, (_, candidate) in sorted(self.best_entropy_per_channel.items()):
            if len(selected) < self.max_views:
                # Check if not too similar to existing views
                is_duplicate = False
                for existing in selected:
                    if existing.c == candidate.c and abs(existing.z - candidate.z) < 3:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    selected.append(candidate)

        # Add middle views if needed
        for candidate in self.candidates:
            if candidate.view_type == "middle" and len(selected) < self.max_views:
                is_duplicate = any(
                    ex.c == candidate.c and abs(ex.z - candidate.z) < 3
                    for ex in selected
                )
                if not is_duplicate:
                    selected.append(candidate)

        # Ensure we have at least 2 views for ISCC-MIXED
        if len(selected) < 2:
            # Add all candidates if not enough selected
            selected_ids = {id(v) for v in selected}
            for candidate in self.candidates:
                if id(candidate) not in selected_ids and len(selected) < self.max_views:
                    selected.append(candidate)
                    selected_ids.add(id(candidate))

            # If still not enough, add any remaining from tracking dicts
            if len(selected) < 2:
                for _, (_, candidate) in self.best_focus_per_channel.items():
                    if (
                        id(candidate) not in selected_ids
                        and len(selected) < self.max_views
                    ):
                        selected.append(candidate)
                        selected_ids.add(id(candidate))

        # Sort by channel then quality
        selected.sort(key=lambda v: (v.c, -v.quality_score))

        return selected[: self.max_views]


def generate_biocode(
    image_path: str, output_dir: Optional[str] = None, max_views: int = 5
) -> List[SceneFingerprint]:
    """Generate biocode fingerprints for all scenes in a bioimage.

    Args:
        image_path: Path to the bioimage file
        output_dir: Optional directory to save view PNGs
        max_views: Maximum views per scene (default: 5)

    Returns:
        List of SceneFingerprint objects
    """
    from bioio import BioImage

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    fingerprints = []
    img = BioImage(image_path)
    num_scenes = len(img.scenes)

    logger.info(f"Processing {num_scenes} scene(s) from {image_path.name}")

    for scene_idx in range(num_scenes):
        if num_scenes > 1:
            img.set_scene(scene_idx)
            scene_id = img.scenes[scene_idx]
            logger.info(f"Processing scene {scene_idx}: {scene_id}")
        else:
            scene_id = image_path.stem

        # Get dimensions
        dims = img.dims
        shape = dims.shape
        dim_order = dims.order

        t_idx = dim_order.index("T") if "T" in dim_order else None
        c_idx = dim_order.index("C") if "C" in dim_order else None
        z_idx = dim_order.index("Z") if "Z" in dim_order else None

        size_t = shape[t_idx] if t_idx is not None else 1
        size_c = shape[c_idx] if c_idx is not None else 1
        size_z = shape[z_idx] if z_idx is not None else 1
        size_y = shape[dims.order.index("Y")]
        size_x = shape[dims.order.index("X")]

        dim_info = {"T": size_t, "C": size_c, "Z": size_z, "Y": size_y, "X": size_x}

        logger.debug(
            f"Scene {scene_idx} dimensions: T={size_t}, C={size_c}, Z={size_z}, Y={size_y}, X={size_x}"
        )

        # Initialize hasher and view selector
        hasher = iscc_sum.IsccSumProcessor()
        selector = ViewSelector(max_views=max_views)

        # Process planes in Z→C→T order
        for z in range(size_z):
            for c in range(size_c):
                for t in range(size_t):
                    kwargs = {}
                    if z_idx is not None:
                        kwargs["Z"] = z
                    if c_idx is not None:
                        kwargs["C"] = c
                    if t_idx is not None:
                        kwargs["T"] = t

                    plane = img.get_image_data("YX", **kwargs)

                    # Update pixel hash
                    canonical_bytes = _plane_to_canonical_bytes(plane)
                    hasher.update(canonical_bytes)

                    # Process for view selection
                    selector.process_plane(plane, z, c, t, dim_info)

        # Finalize ISCC-SUM
        iscc_sum_result = hasher.result(wide=True, add_units=False)
        iscc_sum_code = iscc_sum_result.iscc

        # Select final views
        selected_views = selector.select_final_views()

        # Generate ISCC-IMAGE codes for each view
        view_data = []
        image_codes = []

        for view_idx, view in enumerate(selected_views):
            # Prepare for ISCC
            pixels = _prepare_for_iscc(view.data)

            # Generate ISCC-IMAGE code
            img_code_result = ic.gen_image_code_v0(pixels.tolist(), bits=256)
            img_code = img_code_result["iscc"]
            image_codes.append(img_code)

            # Create view ID
            view_id = f"{image_path.stem}_s{scene_idx}_c{view.c}_z{view.z}_t{view.t}_{view.view_type}"

            # Save view if output directory specified
            if output_dir:
                view_path = output_dir / f"{view_id}.png"
                pil_img = Image.fromarray(view.data, mode="L")
                pil_img.save(view_path)
                logger.debug(f"Saved view: {view_path}")

            view_data.append(
                {
                    "view_id": view_id,
                    "iscc_image": img_code,
                    "z": view.z,
                    "c": view.c,
                    "t": view.t,
                    "type": view.view_type,
                    "metadata": view.metadata,
                }
            )

        # Generate ISCC-MIXED code from all view codes (requires at least 2 codes)
        if len(image_codes) >= 2:
            mixed_result = ic.gen_mixed_code_v0(image_codes, bits=256)
            iscc_mixed_code = mixed_result["iscc"]
        else:
            iscc_mixed_code = None
            if image_codes:
                logger.warning(
                    f"Scene {scene_idx}: Only {len(image_codes)} view(s) extracted, "
                    "ISCC-MIXED requires at least 2 codes"
                )

        # Create fingerprint
        fingerprint = SceneFingerprint(
            scene_id=scene_id,
            iscc_sum=iscc_sum_code,
            views=view_data,
            iscc_mixed=iscc_mixed_code,
        )

        fingerprints.append(fingerprint)

    return fingerprints


def format_output(fingerprints: List[SceneFingerprint], filename: str) -> str:
    """Format fingerprints as human-readable output.

    Args:
        fingerprints: List of SceneFingerprint objects
        filename: Original filename

    Returns:
        Formatted string output
    """
    lines = [f"Processing File: {filename}", ""]

    for fp in fingerprints:
        lines.append(f"Scene ID: {fp.scene_id}")
        lines.append(f"ISCC-SUM: {fp.iscc_sum}")

        for view in fp.views:
            lines.append(f"    View ID: {view['view_id']}.png")
            lines.append(f"    ISCC-IMAGE: {view['iscc_image']}")

        if fp.iscc_mixed:
            lines.append(f"ISCC-MIXED: {fp.iscc_mixed}")

        lines.append("")

    return "\n".join(lines)
