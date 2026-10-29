from types import ModuleType, TracebackType
from typing import Union, Dict, Any

def error_message_details(error, error_detail: Any):
    # Read the traceback for the exception currently being handled.
    _, _, exe_tb = error_detail.exc_info()
    # The traceback points to the file and line where the error occurred.
    file_name = (
        exe_tb.tb_frame.f_code.co_filename
        if exe_tb is not None
        else "Unknown"
    )

    line_number = exe_tb.tb_lineno if exe_tb is not None else "Unknown"
    # Build one readable message containing the location and original error.
    error_message = "Error occured in Pyhton Script Name [{0}] Line Number [{1}] Error Message [{2}]".format(
        file_name, line_number, str(error)
    )

    return error_message

# Define explicitly what your error handling system permits:
ErrorDetailType = Union[ModuleType, TracebackType, Dict[str, Any], str, None]

class CustomException(Exception):
    def __init__(self, error_message, error_detail: ErrorDetailType):
        # Initialize Exception so normal exception behavior is preserved.
        super().__init__(error_message)

        # Keep the original error context for callers that need it.
        self.error_detail = error_detail

        # Store the formatted message returned by the traceback helper.
        self.error_message = error_message_details(error_message, error_detail = error_detail)

    def __str__(self):
        # Display the formatted message when the exception is printed.
        return self.error_message