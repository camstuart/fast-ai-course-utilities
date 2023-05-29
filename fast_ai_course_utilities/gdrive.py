import IPython.display
import requests


def _get_gdrive_object(file_id: str) -> bytes:
    """Get a file object from Google Drive by ID"""
    r = requests.get(f"https://drive.google.com/uc?id={file_id}", timeout=5)
    return bytes(r.content)


def gdrive_audio_display(file_id: str) -> IPython.display.Audio:
    """Get audio file from a Google Drive by ID and return a display"""
    return IPython.display.Audio(_get_gdrive_object(file_id))


def gdrive_file_download(file_id: str, output_filename: str) -> None:
    """Download a file from Google Drive by ID and save locally"""
    file_data = _get_gdrive_object(file_id)
    with open(output_filename, "wb") as fh:
        fh.write(file_data)
