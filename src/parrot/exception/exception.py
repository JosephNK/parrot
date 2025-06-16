from enum import Enum


class TranslationErrorCode(Enum):
    TRANSLATION_ERROR = 1000


class TranslationError(Exception):
    """TranslationError 클래스"""

    def __init__(self, message="An error occurred during translation", error_code=None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

    def __str__(self):
        if self.error_code:
            return f"[ErrorCode {self.error_code}] {self.message}"
        return self.message
