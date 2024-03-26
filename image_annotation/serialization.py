import json
import os
import shutil
from _py_abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeVar, Any, Tuple

import msgpack_numpy
import numpy as np

from artemis.general.utils_for_testing import hold_tempfile
from dataclasses_serialization.serializer_base import Serializer, dict_serialization, dict_deserialization, noop_serialization, noop_deserialization, list_deserialization, dict_to_dataclass
from dataclasses_serialization.serializer_base.tuple import tuple_deserialization


class ICustomPreserializer(metaclass=ABCMeta):

    def serialize(self, obj: Any) -> Any:
        """ Turn the object into a bytestring """
        return {DataClassWithNumpyPreSerializer.serialize(k): DataClassWithNumpyPreSerializer.serialize(v) for k, v in obj.__dict__.items()}

    @classmethod
    def deserialize(cls: type, data: Any) -> Any:
        """ Load the object from a bytestring """
        # if not isinstance(data, dict):
        #     # DeserializationError gets caught in union_deserialization - important for Optional-types
        #     raise DeserializationError(f"Cannot deserialize {data} to type {cls}")
        # return cls(**data)
        return dict_to_dataclass(cls, data, deserialization_func=DataClassWithNumpyPreSerializer.deserialize)


class ISerializeableWithTransients(ICustomPreserializer):

    @classmethod
    @abstractmethod
    def get_transient_fields(cls: type) -> Tuple[str, ...]:
        raise NotImplementedError()

    def serialize(self, obj: Any) -> Any:
        """ Turn the object into a bytestring """
        return {DataClassWithNumpyPreSerializer.serialize(k): DataClassWithNumpyPreSerializer.serialize(v) for k, v in obj.__dict__.items()
                if k not in self.get_transient_fields()}


DataClassWithNumpyPreSerializer = Serializer(
    serialization_functions={
        dict: lambda dct: dict_serialization(dct, key_serialization_func=DataClassWithNumpyPreSerializer.serialize, value_serialization_func=DataClassWithNumpyPreSerializer.serialize),
        list: lambda lst: list(map(DataClassWithNumpyPreSerializer.serialize, lst)),
        tuple: lambda lst: list(map(DataClassWithNumpyPreSerializer.serialize, lst)),
        (str, int, float, bool, type(None)): noop_serialization,
        np.float64: float,
        np.float32: float,
        np.float16: float,
        # np.ndarray: msgpack_numpy.packb,
        np.ndarray: noop_serialization,
        np.int32: int,
        np.int64: int,
        np.uint8: int,
        Enum: lambda item: item.value,
        ICustomPreserializer: lambda item: item.serialize(item)
    },
    deserialization_functions={
        dict: lambda cls, dct: dict_deserialization(cls, dct, key_deserialization_func=DataClassWithNumpyPreSerializer.deserialize,
                                                    value_deserialization_func=DataClassWithNumpyPreSerializer.deserialize),
        list: lambda cls, lst: list_deserialization(cls, lst, deserialization_func=DataClassWithNumpyPreSerializer.deserialize),
        (str, int, float, bool, type(None)): noop_deserialization,
        np.ndarray: noop_deserialization,
        # tuple: lambda cls, lst: tuple(DataClassWithNumpyPreSerializer.deserialize(nested_cls, v) for nested_cls, v in zip_equal(get_args(cls), lst)),  # For now...
        tuple: lambda cls, lst: tuple_deserialization(cls, lst, deserialization_func=DataClassWithNumpyPreSerializer.deserialize),  # For now...
        # np.ndarray: lambda cls, data: msgpack_numpy.unpackb(data),  # ERROR WAS HERE...
        Enum: lambda cls, data: cls(data),
        ICustomPreserializer: lambda cls, data: cls.deserialize(data)
    }
)


def obj_to_msgpack_bytes(obj: Any) -> bytes:
    primitive = DataClassWithNumpyPreSerializer.serialize(obj)
    return msgpack_numpy.dumps(primitive)


def msgpack_bytes_to_obj(cls: type, bytestr: bytes) -> Any:
    primitive = msgpack_numpy.loads(bytestr, strict_map_key=False)
    return DataClassWithNumpyPreSerializer.deserialize(cls, primitive)



ObjType = TypeVar('ObjType')



@dataclass
class ISerializer(Generic[ObjType], metaclass=ABCMeta):
    cls: type(ObjType)

    @abstractmethod
    def dumps(self, obj: ObjType) -> bytes:
        """ Turn the object into a bytestring """

    @abstractmethod
    def loads(self, bytestr: bytes) -> ObjType:
        """ Load the object from a bytestring """


@dataclass
class MsgPackSerializer(ISerializer):
    cls: type(ObjType)

    def dumps(self, obj: ObjType) -> bytes:
        return obj_to_msgpack_bytes(obj)

    def loads(self, bytestr: bytes) -> ObjType:
        return msgpack_bytes_to_obj(self.cls, bytestr)


@dataclass
class JSONSerializer(ISerializer):
    cls: type(ObjType)

    def dumps(self, obj: ObjType) -> str:
        primitive = DataClassWithNumpyPreSerializer.serialize(obj)
        return json.dumps(primitive)

    def loads(self, bytestr: str) -> ObjType:
        primitive = json.loads(bytestr)
        return DataClassWithNumpyPreSerializer.deserialize(self.cls, primitive)



#
# MsgPackSerializer = Serializer(
#     serialization_functions={object: obj_to_msgpack_bytes},
#     deserialization_functions={object: msgpack_bytes_to_obj}
# )


@dataclass
class FileSerializer(Generic[ObjType]):
    path: str

    # klass: type
    serializer: ISerializer
    binary_format: bool = True

    @classmethod
    def from_msgpack(cls, path: str, klass: type) -> 'FileSerializer':
        return FileSerializer(path=path, serializer=MsgPackSerializer(klass), binary_format=True)

    @classmethod
    def from_json(cls, path: str, klass: type) -> 'FileSerializer':
        return FileSerializer(path=path, serializer=JSONSerializer(klass), binary_format=False)

    def dump(self, obj: ObjType):
        parent_dir, _ = os.path.split(self.path)
        os.makedirs(parent_dir, exist_ok=True)
        try:
            with hold_tempfile() as ftemp:
                with open(ftemp, 'wb' if self.binary_format else 'w') as f:
                    byte_rep = self.serializer.dumps(obj)
                    f.write(byte_rep)
                shutil.move(ftemp, self.path)  # Safer to move after in case there is an error during write
        except:
            if os.path.exists(self.path):
                os.remove(self.path)
            raise

    def load(self) -> ObjType:
        with open(self.path, 'rb' if self.binary_format else 'r') as f:
            byte_rep = f.read()
            return self.serializer.loads(byte_rep)


MetaDataType = TypeVar('MetaDataType')
DataType = TypeVar('DataType')


@dataclass
class FileWithMetaDataSer(Generic[MetaDataType, DataType]):
    """ This is handy if you have metadata that you want to be able to load without loading the entire file """
    preserializer: Serializer = field(default_factory=lambda: DataClassWithNumpyPreSerializer)

    def dump(self, metadata: MetaDataType, data: DataType, path: str, mkdirs: bool = True):
        path = os.path.expanduser(path)
        if mkdirs:
            parent, _ = os.path.split(path)
            os.makedirs(parent, exist_ok=True)

        metaser = self.preserializer.serialize(metadata)
        videoser = self.preserializer.serialize(data)

        with hold_tempfile(ext='.mpk') as fname:
            with open(fname, 'wb') as f:
                packer = msgpack_numpy.Packer(f)
                f.write(packer.pack(metaser))
                f.write(packer.pack(videoser))
            shutil.move(fname, path)

    def load_metadata(self, cls: type(MetaDataType), path: str) -> MetaDataType:
        with open(path, 'rb') as f:
            unpacker = msgpack_numpy.Unpacker(f, raw=False, strict_map_key=False)
            remetaser = unpacker.unpack()
        return self.preserializer.deserialize(cls, remetaser)

    def load_data(self, cls: type(DataType), path: str) -> DataType:
        with open(path, 'rb') as f:
            unpacker = msgpack_numpy.Unpacker(f, raw=False, strict_map_key=False)
            unpacker.unpack()
            data_ser = unpacker.unpack()
        return self.preserializer.deserialize(cls, data_ser)


FILE_WITH_METADATA_SERIALIZER = FileWithMetaDataSer()


# @dataclass
# class MsgPackFileSerializer(FileSerializer):
#     serializer: Serializer


# @dataclass
# class JSONFileSerializer(FileSerializer):
#     serializer: Serializer = JSONStrSerializer
#     binary_format: bool = False


#
# MsgPackFileSerializer = Serializer(
#     serialization_functions={
#         object: lambda obj: msgpack.dumps(DataClassWithNumpyPreSerializer.serialize(obj))
#     },
#     deserialization_functions={
#         object: lambda cls, serialized_obj: DataClassWithNumpyPreSerializer.deserialize(cls, msgpack.loads(serialized_obj, strict_map_key=False))
#     }
# )
#

