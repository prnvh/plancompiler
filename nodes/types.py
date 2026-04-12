from enum import Enum

class NodeType(str, Enum):
    FILE_PATH = "FilePath"
    DATA_FRAME = "DataFrame"
    SERIES = "Series"
    SCALAR = "Scalar"
    DB_HANDLE = "DBHandle"
    HTTP_RESPONSE = "HTTPResponse"
    ANY = "Any"
