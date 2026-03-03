from pydantic import BaseModel
from nodes.types import NodeType

class Node(BaseModel):
    name: str
    description: str
    input_type: NodeType
    output_type: NodeType
    template_path: str
    required_params: list[str] = []
    function_name: str

NODE_REGISTRY = {

    # -------------------------
    # INGESTION
    # -------------------------

    "CSVParser": Node(
        name="CSVParser",
        description="Reads a CSV file from disk and returns a DataFrame",
        input_type=NodeType.FILE_PATH,
        output_type=NodeType.DATA_FRAME,
        template_path="nodes/templates/csv_parser.py",
        required_params=["file_path"],
        function_name="csv_parser"
    ),

    "JSONParser": Node(
        name="JSONParser",
        description="Reads a JSON file from disk and returns a DataFrame",
        input_type=NodeType.FILE_PATH,
        output_type=NodeType.DATA_FRAME,
        template_path="nodes/templates/json_parser.py",
        required_params=["file_path"],
        function_name="json_parser"
    ),

    "ExcelParser": Node(
        name="ExcelParser",
        description="Reads an Excel file from disk and returns a DataFrame",
        input_type=NodeType.FILE_PATH,
        output_type=NodeType.DATA_FRAME,
        template_path="nodes/templates/excel_parser.py",
        required_params=["file_path"],
        function_name="excel_parser"
    ),

    # -------------------------
    # DATAFRAME PROCESSING
    # -------------------------

    "SchemaValidator": Node(
        name="SchemaValidator",
        description="Validates DataFrame columns and types against expected schema",
        input_type=NodeType.DATA_FRAME,
        output_type=NodeType.DATA_FRAME,
        template_path="nodes/templates/schema_validator.py",
        required_params=[],
        function_name="schema_validator"
    ),

    "DataTransformer": Node(
        name="DataTransformer",
        description="Applies transformations to a DataFrame (rename, filter, cast)",
        input_type=NodeType.DATA_FRAME,
        output_type=NodeType.DATA_FRAME,
        template_path="nodes/templates/data_transformer.py",
        required_params=[],
        function_name="data_transformer"
    ),

    "DataFilter": Node(
        name="DataFilter",
        description="Filters rows using pandas query syntax",
        input_type=NodeType.DATA_FRAME,
        output_type=NodeType.DATA_FRAME,
        template_path="nodes/templates/data_filter.py",
        required_params=["condition"],
        function_name="data_filter"
    ),

    "ColumnSelector": Node(
        name="ColumnSelector",
        description="Selects specific columns from a DataFrame",
        input_type=NodeType.DATA_FRAME,
        output_type=NodeType.DATA_FRAME,
        template_path="nodes/templates/column_selector.py",
        required_params=["columns"],
        function_name="column_selector"
    ),

    "NullHandler": Node(
        name="NullHandler",
        description="Handles null values in a DataFrame (drop or fill)",
        input_type=NodeType.DATA_FRAME,
        output_type=NodeType.DATA_FRAME,
        template_path="nodes/templates/null_handler.py",
        required_params=["strategy"],
        function_name="null_handler"
    ),

    "DataSorter": Node(
        name="DataSorter",
        description="Sorts a DataFrame by a specified column",
        input_type=NodeType.DATA_FRAME,
        output_type=NodeType.DATA_FRAME,
        template_path="nodes/templates/data_sorter.py",
        required_params=["by", "ascending"],
        function_name="data_sorter"
    ),

    "TypeCaster": Node(
        name="TypeCaster",
        description="Casts DataFrame columns to specified types",
        input_type=NodeType.DATA_FRAME,
        output_type=NodeType.DATA_FRAME,
        template_path="nodes/templates/type_caster.py",
        required_params=["mapping"],
        function_name="type_caster"
    ),

    "DataFrameJoin": Node(
        name="DataFrameJoin",
        description="Joins two DataFrames on a key",
        input_type=NodeType.DATA_FRAME,  # you will allow multi-input validation
        output_type=NodeType.DATA_FRAME,
        template_path="nodes/templates/dataframe_join.py",
        required_params=["on", "how"],
        function_name="dataframe_join"
    ),

    "StatsSummary": Node(
        name="StatsSummary",
        description="Generates descriptive statistics for a DataFrame",
        input_type=NodeType.DATA_FRAME,
        output_type=NodeType.DATA_FRAME,
        template_path="nodes/templates/stats_summary.py",
        required_params=[],
        function_name="stats_summary"
    ),

    "DataDeduplicator": Node(
        name="DataDeduplicator",
        description="Removes duplicate rows from a DataFrame",
        input_type=NodeType.DATA_FRAME,
        output_type=NodeType.DATA_FRAME,
        template_path="nodes/templates/data_deduplicator.py",
        required_params=[],
        function_name="data_deduplicator"
    ),

    "Aggregator": Node(
        name="Aggregator",
        description="Aggregates a DataFrame — group by, sum, count, mean",
        input_type=NodeType.DATA_FRAME,
        output_type=NodeType.DATA_FRAME,
        template_path="nodes/templates/aggregator.py",
        required_params=["group_by", "agg_func"],
        function_name="aggregator"
    ),

    # -------------------------
    # STORAGE
    # -------------------------

    "SQLiteConnector": Node(
        name="SQLiteConnector",
        description="Stores a DataFrame into a SQLite database table",
        input_type=NodeType.DATA_FRAME,
        output_type=NodeType.DB_HANDLE,
        template_path="nodes/templates/sqlite_connector.py",
        required_params=["db_path", "table_name"],
        function_name="sqlite_connector"
    ),

    "SQLiteReader": Node(
        name="SQLiteReader",
        description="Connects to an existing SQLite database and returns a DB handle",
        input_type=NodeType.FILE_PATH,
        output_type=NodeType.DB_HANDLE,
        template_path="nodes/templates/sqlite_reader.py",
        required_params=["db_path"],
        function_name="sqlite_reader"
    ),

    "PostgresConnector": Node(
        name="PostgresConnector",
        description="Stores a DataFrame into a PostgreSQL database table",
        input_type=NodeType.DATA_FRAME,
        output_type=NodeType.DB_HANDLE,
        template_path="nodes/templates/postgres_connector.py",
        required_params=["connection_string", "table_name"],
        function_name="postgres_connector"
    ),

    "QueryEngine": Node(
        name="QueryEngine",
        description="Runs SQL queries against a DBHandle and returns a DataFrame",
        input_type=NodeType.DB_HANDLE,
        output_type=NodeType.DATA_FRAME,
        template_path="nodes/templates/query_engine.py",
        required_params=["query"],
        function_name="query_engine"
    ),

    # -------------------------
    # EXPORTERS
    # -------------------------

    "CSVExporter": Node(
        name="CSVExporter",
        description="Exports a DataFrame to a CSV file",
        input_type=NodeType.DATA_FRAME,
        output_type=NodeType.FILE_PATH,
        template_path="nodes/templates/csv_exporter.py",
        required_params=["output_path"],
        function_name="csv_exporter"
    ),

    "JSONExporter": Node(
        name="JSONExporter",
        description="Exports a DataFrame to a JSON file",
        input_type=NodeType.DATA_FRAME,
        output_type=NodeType.FILE_PATH,
        template_path="nodes/templates/json_exporter.py",
        required_params=["output_path"],
        function_name="json_exporter"
    ),

    # -------------------------
    # API / HTTP
    # -------------------------

    "RESTEndpoint": Node(
        name="RESTEndpoint",
        description="Exposes a DBHandle as a REST API endpoint using Flask",
        input_type=NodeType.DB_HANDLE,
        output_type=NodeType.HTTP_RESPONSE,
        template_path="nodes/templates/rest_endpoint.py",
        required_params=["route", "port"],
        function_name="rest_endpoint"
    ),

    "AuthMiddleware": Node(
        name="AuthMiddleware",
        description="Adds API key authentication to an HTTP endpoint",
        input_type=NodeType.HTTP_RESPONSE,
        output_type=NodeType.HTTP_RESPONSE,
        template_path="nodes/templates/auth_middleware.py",
        required_params=["api_key_env_var"],
        function_name="auth_middleware"
    ),

    "ErrorHandler": Node(
        name="ErrorHandler",
        description="Wraps any output in structured error handling",
        input_type=NodeType.ANY,
        output_type=NodeType.HTTP_RESPONSE,
        template_path="nodes/templates/error_handler.py",
        required_params=[],
        function_name="error_handler"
    ),

    # -------------------------
    # OBSERVABILITY
    # -------------------------

    "Logger": Node(
        name="Logger",
        description="Logs intermediate data without modifying it",
        input_type=NodeType.ANY,
        output_type=NodeType.ANY,
        template_path="nodes/templates/logger.py",
        required_params=[],
        function_name="logger"
    ),
    # -------------------------
    # ADAPTERS / TYPE BRIDGES
    # -------------------------
    
    "HTTPToDataFrame": Node(
        name="HTTPToDataFrame",
        description="Converts HTTP JSON response into a DataFrame",
        input_type=NodeType.HTTP_RESPONSE,
        output_type=NodeType.DATA_FRAME,
        template_path="nodes/templates/http_to_dataframe.py",
        required_params=[],
        function_name="http_to_dataframe"
    ),
}