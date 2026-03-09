import sys

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = "Error occurred in script [{0}] line [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail=sys):
        super().__init__(error_message)
        self.error_message = str(error_message)

        _, _, exc_tb = error_detail.exc_info()

        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            self.error_message = f"Error occurred in script [{file_name}] line [{line_number}] error message [{error_message}]"

    def __str__(self):
        return self.error_message