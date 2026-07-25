

class CustomException(Exception):
    def __init__(self, e):
        super().__init__(str(e))

        self.e = e
        self.tb = getattr(e, "__traceback__", None)
        if not self.tb:
            import sys
            _, _, self.tb = sys.exc_info()

    def __str__(self):
        if self.tb:
            return (
                f"e occurred in file "
                f"{self.tb.tb_frame.f_code.co_filename} "
                f"at line {self.tb.tb_lineno}: {self.e}"
            )
        return str(self.e)