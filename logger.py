import logging

NIKOSIS_LOG_LEVEL = 21
logging.addLevelName(NIKOSIS_LOG_LEVEL, "NIKO_LOG")

class NikoLogFormatter(logging.Formatter):
    def __init__(self, standard_fmt, clean_fmt):
        super().__init__()
        self.standard_formatter = logging.Formatter(standard_fmt)
        self.clean_formatter = logging.Formatter(clean_fmt)

    def format(self, record):
        if record.levelno == NIKOSIS_LOG_LEVEL:
            return self.clean_formatter.format(record)
        return self.standard_formatter.format(record)

# Setup logger
niko_logger = logging.getLogger("Nikosis Nodes")
niko_logger.setLevel(logging.INFO)

# Remove existing handlers
if niko_logger.handlers:
    niko_logger.handlers.clear()

# Create handler with dual formatter
console_handler = logging.StreamHandler()
formatter = NikoLogFormatter(
    standard_fmt='\033[92m%(asctime)s - %(levelname)s - %(message)s\033[0m',
    clean_fmt='\033[36m%(message)s\033[0m'  # Cyan for clean messages
)
console_handler.setFormatter(formatter)
niko_logger.addHandler(console_handler)
niko_logger.propagate = False

# Add custom log method
def niko_log(self, message, *args, **kwargs):
    if self.isEnabledFor(NIKOSIS_LOG_LEVEL):
        self._log(NIKOSIS_LOG_LEVEL, message, args, **kwargs)

niko_logger.niko_log = niko_log.__get__(niko_logger, logging.Logger)
