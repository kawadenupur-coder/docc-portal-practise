import sys
import traceback
from logger.custom_logger import CustomLogger

logger = CustomLogger().get_logger(__file__)

class DocumentPortalException(Exception):
    """Custom exception for Document Portal"""
    
    def __init__(self, error_message, error_details: sys):
        """
        error_details: exc_info()
        """
        self.file_name = error_details.exc_info()[2].tb_frame.f_code.co_filename
        self.lineno = error_details.exc_info()[2].tb_lineno
        self.error_message = str(error_message)
        self.traceback_str = ''.join(traceback.format_exception(*error_details.exc_info()))
        
        # Log the error
        logger.error(
            "DocumentPortalException raised",
            file=self.file_name,
            line=self.lineno,
            error_message=self.error_message,
            traceback=self.traceback_str
        )
        
        super().__init__(self.__str__())
    
    def __str__(self):
        return f"""
Error in [{self.file_name}] at line [{self.lineno}]
Message: {self.error_message}
Traceback:
{self.traceback_str}
"""

if __name__ == "__main__":
    try:
        # Simulate an error
        a = 1 / 0
    except Exception as e:
        print(e)
    except Exception as e:
        exp_exc = DocumentPortalException(e, sys)
        logger.error(exp_exc)
        raise exp_exc