"""High-level Python API for biocode generation.

Provides a single entry point for generating ISCC-SUM biocodes from any supported
source: local files (via BioIO), OME-NGFF/Zarr archives, or OMERO servers.
"""

from importlib.metadata import version
from pathlib import Path

from loguru import logger

from iscc_bio.biocode import generate_biocode

GENERATOR = f"iscc-bio - v{version('iscc-bio')}"


def biocode(
    source=None,
    *,
    simprints=False,
    bits=256,
    source_type="auto",
    host=None,
    port=4064,
    username=None,
    password=None,
    iid=None,
    fid=None,
    conn=None,
):
    """Generate biocode (ISCC-SUM) for bioimage data.

    Produces a top-level raw-byte ISCC-SUM over the original source (a single
    file, or a directory tree such as an OME-Zarr store traversed in
    TREEWALK-ISCC order) with the per-scene IMAGEWALK content codes nested as
    ``parts``. The top-level code is decode-independent (still produced if image
    reading fails) and, for a single file, identical to the standard ``iscc-sum``
    tool. Each part matches the IsccEntry schema of the ISCC search API
    (https://search.iscc.id).

    Supports three data sources:

    - **Local files** via BioIO (OME-TIFF, CZI, ND2, LIF, etc.)
    - **OME-NGFF/Zarr** archives (local or remote)
    - **OMERO** servers via Blitz gateway

    :param source: Path to bioimage file or zarr directory (for local sources)
    :param simprints: Generate per-plane data-code simprints (DATA_NONE_V0)
    :param bits: Bit length for ISCC codes (default: 256)
    :param source_type: Source type hint — ``"auto"``, ``"bioio"``, or ``"zarr"``
    :param host: OMERO server hostname (e.g., ``"omero.server.com"``)
    :param port: OMERO server port (default: 4064)
    :param username: OMERO username (required with ``host``)
    :param password: OMERO password (required with ``host``)
    :param iid: OMERO image ID
    :param fid: OMERO fileset ID
    :param conn: Pre-connected BlitzGateway instance (alternative to host/username/password)
    :return: Dict with the top-level ``iscc_code``, ``units``, ``datahash``,
        ``filesize``, ``generator``, and ``parts`` (the per-scene entries). For
        OMERO sources the top-level ``iscc_code`` is ``None`` (raw originals are
        not yet fetched); ``parts`` still carries the per-scene codes.
    :raises FileNotFoundError: If source path does not exist
    :raises ValueError: If parameters are invalid or incomplete
    :raises ConnectionError: If OMERO connection fails
    :raises ImportError: If required optional dependency is missing

    **Examples**::

        >>> from iscc_bio.api import biocode

        # Local bioimage file (auto-detects format via BioIO)
        >>> result = biocode("image.ome.tiff")
        >>> result["iscc_code"]              # raw-byte top-level ISCC-SUM
        'ISCC:...'
        >>> result["parts"][0]["iscc_code"]  # per-scene content code
        'ISCC:...'

        # OME-NGFF/Zarr archive (top-level code is the tree SUM over the store)
        >>> result = biocode("dataset.zarr")

        # With per-plane simprints for granular similarity search
        >>> result = biocode("image.czi", simprints=True)
        >>> result["parts"][0]["simprints"]["DATA_NONE_V0"][0]
        {'simprint': '...', 'offset': 0, 'size': ...}

        # OMERO server — single image
        >>> results = biocode(host="omero.server.com", username="user",
        ...                   password="pass", iid=123)

        # OMERO server — entire fileset
        >>> results = biocode(host="omero.server.com", username="user",
        ...                   password="pass", fid=456)

        # OMERO with pre-connected BlitzGateway
        >>> from omero.gateway import BlitzGateway
        >>> conn = BlitzGateway("user", "pass", host="omero.server.com")
        >>> conn.connect()
        >>> results = biocode(conn=conn, iid=123)
        >>> conn.close()
    """
    is_omero = (
        host is not None or conn is not None or iid is not None or fid is not None
    )
    is_local = source is not None

    if is_omero and is_local:
        raise ValueError(
            "Cannot specify both 'source' and OMERO parameters (host/conn/iid/fid)"
        )

    if is_omero:
        return _biocode_omero(
            host=host,
            port=port,
            username=username,
            password=password,
            iid=iid,
            fid=fid,
            conn=conn,
            simprints=simprints,
            bits=bits,
        )

    if is_local:
        return _biocode_local(
            source,
            source_type=source_type,
            simprints=simprints,
            bits=bits,
        )

    raise ValueError(
        "Provide 'source' for local files or OMERO parameters (host/conn + iid/fid)"
    )


def _biocode_local(source, *, source_type, simprints, bits):
    """Generate biocode from a local file or zarr directory.

    Computes the raw-byte top-level ISCC-SUM first (so it is produced even if
    pixel decoding fails), then nests the per-scene IMAGEWALK codes as ``parts``.
    """
    from iscc_bio.imagewalk import iter_planes_bioio, iter_planes_ngff
    from iscc_bio.rawsum import raw_sum

    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    if source_type == "auto":
        if source.suffix == ".zarr" or (
            source.is_dir() and (source / ".zattrs").exists()
        ):
            source_type = "zarr"
        else:
            source_type = "bioio"

    if source_type not in ("zarr", "bioio"):
        raise ValueError(
            f"Unknown source_type: {source_type!r} (expected 'auto', 'bioio', or 'zarr')"
        )

    top = raw_sum(source, bits=bits)

    if source_type == "zarr":
        planes = iter_planes_ngff(str(source))
    else:
        planes = iter_planes_bioio(str(source))

    try:
        parts = generate_biocode(planes, simprints=simprints, bits=bits)
    except Exception as exc:
        # Graceful degradation: the raw-byte top-level identity is already
        # computed, so a decode/read failure must not discard it. Emit a warning
        # and return the top-level code with no parts.
        logger.warning(
            f"Pixel decoding failed for {source}; returning top-level code only ({exc})"
        )
        parts = []

    return {**top, "generator": GENERATOR, "parts": parts}


def _biocode_omero(*, host, port, username, password, iid, fid, conn, simprints, bits):
    """Generate biocode from an OMERO server."""
    from iscc_bio.imagewalk import iter_planes_blitz_image, iter_planes_blitz_fileset

    if not iid and not fid:
        raise ValueError("OMERO mode requires 'iid' (image ID) or 'fid' (fileset ID)")

    own_conn = conn is None
    if own_conn:
        if not host:
            raise ValueError(
                "Provide 'host' or a pre-connected 'conn' for OMERO access"
            )
        if not username or not password:
            raise ValueError("OMERO connection requires 'username' and 'password'")

        try:
            from omero.gateway import BlitzGateway
        except ImportError:
            raise ImportError(
                "OMERO support requires omero-py. Install with: pip install -r requirements-omero.txt"
            ) from None

        conn = BlitzGateway(username, password, host=host, port=port)
        if not conn.connect():
            raise ConnectionError(f"Failed to connect to OMERO server: {host}:{port}")

    try:
        if fid:
            fileset = conn.getObject("Fileset", fid)
            if not fileset:
                raise ValueError(f"Fileset {fid} not found on OMERO server")
            planes = iter_planes_blitz_fileset(conn, fileset)
        else:
            image = conn.getObject("Image", iid)
            if not image:
                raise ValueError(f"Image {iid} not found on OMERO server")
            planes = iter_planes_blitz_image(conn, image)

        parts = generate_biocode(planes, simprints=simprints, bits=bits)
        # OMERO raw originals (RawFileStore) are not yet fetched, so there is no
        # raw-byte top-level code; the per-scene parts still carry full identity.
        return {"iscc_code": None, "generator": GENERATOR, "parts": parts}
    finally:
        if own_conn:
            conn.close()
