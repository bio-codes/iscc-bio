"""Quick test to verify SHA256 fix for OMERO compatibility."""

import hashlib
from pathlib import Path
from bioio import BioImage
from iscc_bio.canonical import plane_to_canonical_bytes, calculate_pixel_sha256_bioio

def test_sha256_calculation():
    """Test SHA256 calculation with a local bioimage file."""

    # Test with any available bioimage file
    test_files = [
        Path(r"E:\biocodes\bioimages\xyc_tiles.czi"),
        # Add other test files as needed
    ]

    for test_file in test_files:
        if test_file.exists():
            print(f"\nTesting with: {test_file.name}")
            print("=" * 60)

            # Calculate using our new SHA256 function
            sha256_hash = calculate_pixel_sha256_bioio(test_file, scene_index=0)
            print(f"SHA256 hash: {sha256_hash}")

            # Also calculate a quick SHA1 for comparison
            img = BioImage(test_file)
            img.set_scene(0)

            # Get first plane for quick test
            plane = img.get_image_data("YX", C=0, T=0, Z=0)
            plane_bytes = plane_to_canonical_bytes(plane)

            sha1_single = hashlib.sha1(plane_bytes).hexdigest()
            sha256_single = hashlib.sha256(plane_bytes).hexdigest()

            print(f"\nFirst plane hashes:")
            print(f"  SHA1:   {sha1_single}")
            print(f"  SHA256: {sha256_single}")

            print("\nThe complete SHA256 hash above should match OMERO's")
            print("calculateMessageDigest() output when the same image")
            print("is imported into OMERO!")

            break
    else:
        print("No test files found!")

if __name__ == "__main__":
    test_sha256_calculation()