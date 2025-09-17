# iscc-bio - ISCC Processing for Bioimages

This python library and CLI tool explores the posibilities of generating ISO 24138:2024 International Standard 
Content Codes (ISCC) for bio image formats supported by bioio. Bioimage formats can be very different from 
standard image formats containing multiple layers and views that require special handling.

For documentation of bioio see https://deepwiki.com/bioio-devs/bioio
For documentation of ISCC see https://deepwiki.com/iscc/iscc-core

## ISCC Mixed-Code Approach

ISO 24138:2024 supports ISCC Image-Codes which are perceptual hashes of 2D images. To support bioimages, we 
need a format agnostic and deterministic approach to segment any bioimage into individual 2D images.
We could than create individual Image-Codes per 2D image and create a global descriptor of the collection of 
images using the ISCC Mixed-Code.

