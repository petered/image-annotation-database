import dataclasses
import os
from dataclasses import dataclass, fields
from typing import Optional, Tuple, Dict, Any

from image_annotation.basic_utils import is_image_path


class VideoChannels:
    VISUAL = "Visual"
    THERMAL = "Thermal"


@dataclass
class RecordQuery:
    case: Optional[str] = None
    record_id: Optional[str] = None
    nickname: Optional[str] = None
    detector_name: Optional[str] = None
    file: Optional[str] = None  # Path to an annotated file in the record - can be used to search for records matching this file


@dataclass
class PlayStateQuery:
    index: int = 0  # Index of frame in video or image sequence
    ij: Optional[Tuple[int, int]] = None  # (row, col) of center pixel in view
    zoom: Optional[float] = None  # Zoom level of view (number of display pixels per image pixel)
    source: str = VideoChannels.VISUAL  # Which channel to display (e.g. visual, thermal, etc.)


@dataclass
class GotoQuery(PlayStateQuery, RecordQuery):
    """ This is a desired destination to go to.  You can specify it in a number of ways. """
    verified: bool = False  # Whether this query has been verified to point to an existing location
    # file: Optional[str] = None
    # index: int = 0
    # ij: Optional[Tuple[int, int]] = None
    # zoom: Optional[float] = None
    # record_id: Optional[str] = None
    # detector_name: Optional[str] = None
    # case: Optional[str] = None

    @classmethod
    def from_record_and_play_state_queries(cls, record_query: Optional[RecordQuery], play_state_query: Optional[PlayStateQuery]) -> 'GotoQuery':
        """ Create a GotoQuery from a RecordQuery and a PlayStateQuery """
        return cls(**dataclasses.asdict(record_query) if record_query else {}, **dataclasses.asdict(play_state_query) if play_state_query else {})

    @classmethod
    def from_query_string(cls, query_string: str):
        """ Parse a query string into a Destination object.
        e.g. case=My_Case_Name&file=/path/to/file.mp4&frame=1234&xy=123,456&zoom=2
        If it's not resolveable, raise a ValueError
        """
        # Remove the prefix, but don't require it
        query_string = query_string.strip()
        if query_string.startswith('goto:'):
            query_string = query_string[len('goto:'):].strip()

        try:
            decoded_dict: Dict[str, Any()] = {k: v for k, v in (part.split('=') for part in query_string.split('&'))}
        except ValueError:
            raise ValueError(f"Could not parse place identifier string: {query_string}.\n\n Make sure it's in a format like: case=My_Case_Name&file=/path/to/file.mp4&frame=1234&xy=123,456&zoom=2")

        if 'file' in decoded_dict and ':' in os.path.basename(decoded_dict['file']):
            decoded_dict['file'], decoded_dict['index'] = source_identifier_to_path_and_index(decoded_dict['file'])

        return cls(
            case=decoded_dict.get('case', None),
            file=decoded_dict.get('file', None),
            record_id=decoded_dict.get('record_id', None),
            source=decoded_dict.get('source', VideoChannels.VISUAL),
            index=int(decoded_dict.get('index', 0)) if decoded_dict.get('index', '') else None,
            ij=tuple(int(i) for i in decoded_dict.get('ij', '').strip('()').split(',')) if decoded_dict.get('ij', '') else None,
            zoom=float(decoded_dict.get('zoom', 1)) if decoded_dict.get('zoom', '') else None,
        )

    def to_query_string(self) -> str:

        """ Convert this to a query string (&-separated key=value pairs) """
        # Go through all fields, and only include them if they differ from the default value
        return 'goto:'+'&'.join(f'{f.name}={v}' for f in fields(self) if (v:=getattr(self, f.name)) != f.default or f.name=='index')


@dataclass
class ActualLocation:
    """ A concrete location to go to in the view frame.  It assumes the existence of a record """
    case_name: str
    record_id: str
    index: int
    ij: Optional[Tuple[int, int]] = None
    zoom: Optional[float] = None


def source_identifier_to_path_and_index(source_identifier: str) -> Tuple[str, int]:

    source_id_filename = os.path.basename(source_identifier)
    if ':' in source_id_filename:
        actual_basename, frame_ix = source_id_filename.split(':')
        path = os.path.join(os.path.dirname(source_identifier), actual_basename)
        frame_ix = int(frame_ix)
    else:
        path, frame_ix = source_identifier, 0
        assert is_image_path(path), f"You specified your source identifier as '{source_identifier}.  Since no index was specified, we expect an image."
    return path, frame_ix

