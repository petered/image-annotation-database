import os

IMAGE_FILE_EXTENSIONS = '.jpg', '.jpeg', '.png', '.tiff'
VIDEO_FILE_EXTENSIONS = '.mp4', '.mov', '.avi', '.mpg', '.mpeg', '.m4v', '.mkv', '.wmv', '.flv', '.webm', '.gif'
MEDIA_FILE_EXTENSIONS = IMAGE_FILE_EXTENSIONS + VIDEO_FILE_EXTENSIONS


def is_hidden(path: str) -> bool:
    # Hidden files like "._DJI_20230526115257_0002_W.jpeg" are corrupt or incomplete
    return os.path.basename(path).startswith('.')


def is_image_path(path: str, allow_hidden=False) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_FILE_EXTENSIONS and (allow_hidden or not is_hidden(path))


def is_video_path(path: str, allow_hidden=False) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_FILE_EXTENSIONS and (allow_hidden or not is_hidden(path))


def is_media_path(path: str, allow_hidden=False) -> bool:
    return is_video_path(path) or is_image_path(path) and (allow_hidden or not is_hidden(path))
