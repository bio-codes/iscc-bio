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


def _calculate_coverage(plane: np.ndarray) -> float:
    """Calculate coverage (proportion of non-background pixels)."""
    if plane.size == 0:
        return 0.0

    # Determine threshold based on dtype
    if plane.dtype == np.uint8:
        threshold = 10
    else:
        threshold = np.percentile(plane, 5) if plane.size > 0 else 0

    return float(np.mean(plane > threshold))


def _resize_to_cache(plane: np.ndarray, target_size: int = 384) -> np.ndarray:
    """Resize plane to cache size with contrast enhancement."""
    # Apply percentile-based contrast enhancement
    p2, p98 = np.percentile(plane, [2, 98])

    if p98 > p2:
        # Stretch to full range
        plane_enhanced = np.clip((plane - p2) / (p98 - p2), 0, 1)
        # Apply gamma correction for better contrast
        plane_enhanced = np.power(plane_enhanced, 0.8)
        plane_norm = (plane_enhanced * 255).astype(np.uint8)
    else:
        # Fallback for uniform images
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
        self.channel_planes: Dict[
            int, np.ndarray
        ] = {}  # Cache for composite generation

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
        coverage = _calculate_coverage(plane)

        # Cache middle Z plane as fallback for multi-Z stacks
        # Skip for single-Z images to avoid duplicates with focus views
        if dims["Z"] > 1:
            middle_z = dims["Z"] // 2
            if z == middle_z and not self.middle_z_cached and middle_z > 0:
                cached = _resize_to_cache(plane, self.cache_size)
                candidate = CandidateView(
                    data=cached,
                    z=z,
                    c=c,
                    t=t,
                    quality_score=entropy,
                    view_type="middle",
                    metadata={
                        "entropy": entropy,
                        "focus": focus_score,
                        "coverage": coverage,
                    },
                )
                self.candidates.append(candidate)
                self.middle_z_cached = True

        # Quality thresholds for other views
        min_entropy = 3.5
        min_coverage = 0.1

        # Skip very low quality planes UNLESS we have no views yet (ensure at least one)
        if (entropy < min_entropy or coverage < min_coverage) and len(
            self.best_focus_per_channel
        ) > 0:
            return

        # Track best focus per channel (also serves as fallback for single-Z images)
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
                quality_score=max(entropy, 1.0),  # Minimum quality for fallback
                view_type="focus",
                metadata={
                    "entropy": entropy,
                    "focus": focus_score,
                    "coverage": coverage,
                },
            )
            self.best_focus_per_channel[c] = (focus_score, candidate)
            # Cache plane for composite generation
            if c not in self.channel_planes:
                self.channel_planes[c] = cached

        # Track best entropy per channel
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
                metadata={
                    "entropy": entropy,
                    "focus": focus_score,
                    "coverage": coverage,
                },
            )
            self.best_entropy_per_channel[c] = (entropy, candidate)

    def _generate_iscc(self, view_data: np.ndarray) -> Optional[str]:
        """Generate ISCC code for deduplication."""
        try:
            pixels = _prepare_for_iscc(view_data)
            result = ic.gen_image_code_v0(pixels.tolist(), bits=256)
            return result["iscc"]
        except Exception:
            return None

    def _is_similar(self, code1: Optional[str], code2: Optional[str]) -> bool:
        """Check if two ISCC codes are similar (Hamming distance <= 8)."""
        if not code1 or not code2:
            return False
        try:
            distance = ic.iscc_distance(code1, code2)
            return distance <= 8
        except Exception:
            return False

    def _create_composite(self) -> Optional[CandidateView]:
        """Create RGB composite from best channels."""
        if len(self.channel_planes) < 2:
            return None

        # Get up to 3 best channels
        channels_sorted = sorted(
            self.channel_planes.items(),
            key=lambda x: self.best_focus_per_channel.get(x[0], (0, None))[0],
            reverse=True,
        )[:3]

        if not channels_sorted:
            return None

        # Normalize and stack channels
        normalized = []
        for c_idx, plane in channels_sorted:
            # Percentile normalization
            p2, p98 = np.percentile(plane, [2, 98])
            if p98 > p2:
                norm = np.clip((plane - p2) / (p98 - p2), 0, 1)
            else:
                norm = np.zeros_like(plane, dtype=np.float32)
            normalized.append((norm * 255).astype(np.uint8))

        # Pad to 3 channels if needed
        while len(normalized) < 3:
            normalized.append(np.zeros_like(normalized[0]))

        # Stack as RGB
        composite = np.stack(normalized[:3], axis=-1)

        # Calculate quality from grayscale version
        gray = np.mean(composite, axis=-1).astype(np.uint8)
        entropy = _calculate_entropy(gray)
        coverage = _calculate_coverage(gray)

        return CandidateView(
            data=composite,
            z=-1,  # Composite has no single Z
            c=-1,  # Composite spans channels
            t=0,
            quality_score=entropy,
            view_type="composite",
            metadata={
                "entropy": entropy,
                "coverage": coverage,
                "channels": [c for c, _ in channels_sorted],
            },
        )

    def select_final_views(self) -> List[CandidateView]:
        """Select final representative views with ISCC-based deduplication."""
        candidates_pool = []

        # Create composite view first (highest priority)
        composite = self._create_composite()
        if (
            composite and composite.quality_score >= 3.5
        ):  # Lower threshold for composites
            candidates_pool.append(composite)
            logger.debug(
                f"Added composite view (entropy={composite.quality_score:.2f})"
            )

        # Add best focus views per channel
        for c, (score, candidate) in sorted(self.best_focus_per_channel.items()):
            candidates_pool.append(candidate)
            logger.debug(
                f"Added focus view C{c} Z{candidate.z} (entropy={candidate.quality_score:.2f})"
            )

        # Add best entropy views if different Z from focus
        for c, (score, candidate) in sorted(self.best_entropy_per_channel.items()):
            # Check if different from focus view of same channel
            focus_view = self.best_focus_per_channel.get(c, (0, None))[1]
            if not focus_view or abs(candidate.z - focus_view.z) >= 5:
                candidates_pool.append(candidate)
                logger.debug(
                    f"Added entropy view C{c} Z{candidate.z} (entropy={candidate.quality_score:.2f})"
                )

        # Add middle plane fallback candidates
        for candidate in self.candidates:
            if candidate.view_type == "middle":
                # Check if not already in pool
                if not any(id(c) == id(candidate) for c in candidates_pool):
                    candidates_pool.append(candidate)
                    logger.debug(
                        f"Added middle fallback view (entropy={candidate.quality_score:.2f})"
                    )

        # ISCC-based deduplication
        selected = []
        selected_iscc = []

        for candidate in candidates_pool:
            if len(selected) >= self.max_views:
                break

            # Generate ISCC for this candidate
            if candidate.data.ndim == 3:
                # Composite - use grayscale for comparison
                gray = np.mean(candidate.data, axis=-1).astype(np.uint8)
                iscc_code = self._generate_iscc(gray)
            else:
                iscc_code = self._generate_iscc(candidate.data)

            # Check for similarity with already selected views
            is_duplicate = False
            for existing_iscc in selected_iscc:
                if self._is_similar(iscc_code, existing_iscc):
                    is_duplicate = True
                    logger.debug(
                        f"Skipped duplicate view: {candidate.view_type} C{candidate.c} Z{candidate.z}"
                    )
                    break

            if not is_duplicate:
                selected.append(candidate)
                selected_iscc.append(iscc_code)

        # Ensure minimum of 2 views - relax ISCC filtering if needed
        if len(selected) < 2:
            logger.warning(
                f"Only {len(selected)} unique view(s) after ISCC deduplication, relaxing constraints"
            )
            for candidate in candidates_pool:
                if len(selected) >= self.max_views or len(selected) >= 2:
                    break
                if all(id(candidate) != id(s) for s in selected):
                    selected.append(candidate)
                    logger.debug(
                        f"Added fallback view: {candidate.view_type} C{candidate.c} Z{candidate.z}"
                    )

        logger.info(
            f"Selected {len(selected)} final views from {len(candidates_pool)} candidates"
        )

        return selected


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
            # Prepare for ISCC (handle both grayscale and RGB)
            if view.data.ndim == 3:
                # Composite RGB - convert to grayscale for ISCC
                gray = np.mean(view.data, axis=-1).astype(np.uint8)
                pixels = _prepare_for_iscc(gray)
            else:
                pixels = _prepare_for_iscc(view.data)

            # Generate ISCC-IMAGE code
            img_code_result = ic.gen_image_code_v0(pixels.tolist(), bits=256)
            img_code = img_code_result["iscc"]
            image_codes.append(img_code)

            # Create view ID
            if view.view_type == "composite":
                channels_str = "_".join(
                    str(c) for c in view.metadata.get("channels", [])
                )
                view_id = f"{image_path.stem}_s{scene_idx}_composite_c{channels_str}_t{view.t}"
            else:
                view_id = f"{image_path.stem}_s{scene_idx}_c{view.c}_z{view.z}_t{view.t}_{view.view_type}"

            # Save view if output directory specified
            if output_dir:
                view_path = output_dir / f"{view_id}.png"
                if view.data.ndim == 3:
                    # RGB composite
                    pil_img = Image.fromarray(view.data, mode="RGB")
                else:
                    # Grayscale
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
        elif len(image_codes) == 1:
            # Duplicate the single code to generate ISCC-MIXED
            logger.info(
                f"Scene {scene_idx}: Only 1 view extracted, duplicating code for ISCC-MIXED"
            )
            mixed_result = ic.gen_mixed_code_v0(
                [image_codes[0], image_codes[0]], bits=256
            )
            iscc_mixed_code = mixed_result["iscc"]
        else:
            iscc_mixed_code = None
            logger.warning(
                f"Scene {scene_idx}: No views extracted, cannot generate ISCC-MIXED"
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
