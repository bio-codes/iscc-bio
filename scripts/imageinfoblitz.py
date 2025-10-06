"""Print core metadata from all images on OMERO server.

Connects to OMERO server using Blitz Gateway and iterates through all images,
printing their core metadata fields.
"""

import sys
from omero.gateway import BlitzGateway


def print_image_metadata(conn):
    # type: (BlitzGateway) -> None
    """Print metadata for all images on OMERO server.

    :param conn: Connected BlitzGateway instance
    """
    # Get all images
    images = conn.getObjects("Image")

    image_count = 0
    for image in images:
        image_count += 1
        print(f"\n{'=' * 70}")
        print(f"Image #{image_count}")
        print(f"{'=' * 70}")

        # Core metadata
        print(f"ID:              {image.getId()}")
        print(f"Name:            {image.getName()}")
        print(f"Description:     {image.getDescription() or 'N/A'}")
        print(f"OwnerFullName:   {image.getOwnerFullName()}")
        print(f"Author:          {image.getAuthor()}")
        print(f"Project:         {image.getProject().getName()}")
        print(f"Env:             {image.getImagingEnvironment()}")
        # print(f"Dataset:          {image.getDataset().getName()}")

        # Acquisition date
        acq_date = image.getDate()
        print(f"Acquisition:     {acq_date if acq_date else 'N/A'}")

        # Dimensions
        print(f"Size (X×Y):      {image.getSizeX()} × {image.getSizeY()}")
        print(f"Size Z:          {image.getSizeZ()}")
        print(f"Size C:          {image.getSizeC()}")
        print(f"Size T:          {image.getSizeT()}")

        # Pixel information
        pixels = image.getPrimaryPixels()
        if pixels:
            pixel_type = pixels.getPixelsType()
            print(f"Pixel Type:      {pixel_type.getValue() if pixel_type else 'N/A'}")
            print(f"Physical Size X: {pixels.getPhysicalSizeX() or 'N/A'}")
            print(f"Physical Size Y: {pixels.getPhysicalSizeY() or 'N/A'}")
            print(f"Physical Size Z: {pixels.getPhysicalSizeZ() or 'N/A'}")

            # SHA1 hash if available
            sha1 = pixels.getSha1()
            print(f"SHA1 Hash:       {sha1 if sha1 else 'N/A'}")

        # Dataset/Project information
        datasets = []
        for ds in image.listParents():
            datasets.append(ds.getName())

        if datasets:
            print(f"Datasets:        {', '.join(datasets)}")

        # File format from original file
        fileset = image.getFileset()
        if fileset:
            orig_files = fileset.listFiles()
            if orig_files:
                first_file = list(orig_files)[0]
                print(f"Original File:   {first_file.getName()}")
                print(f"File Path:       {first_file.getPath()}")

    print(f"\n{'=' * 70}")
    print(f"Total images: {image_count}")
    print(f"{'=' * 70}")


def main():
    """Connect to OMERO server and print image metadata."""
    # Connection parameters
    SERVER = "omero.iscc.id"
    PORT = 4064
    USERNAME = "root"
    PASSWORD = "omero"

    print(f"Connecting to OMERO server: {SERVER}:{PORT}")
    print(f"Username: {USERNAME}")

    # Create connection
    conn = BlitzGateway(USERNAME, PASSWORD, host=SERVER, port=PORT)

    try:
        if not conn.connect():
            print(f"ERROR: Failed to connect to {SERVER}:{PORT}", file=sys.stderr)
            sys.exit(1)

        print(f"Successfully connected to {SERVER}")
        print(f"User: {conn.getUser().getName()}")
        print()

        # Print image metadata
        print_image_metadata(conn)

    finally:
        conn.close()
        print("\nConnection closed.")


if __name__ == "__main__":
    main()
