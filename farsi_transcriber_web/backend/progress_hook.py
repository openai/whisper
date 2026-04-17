import threading
import tqdm

# Thread-local storage to track which job is running in the current thread
_thread_local = threading.local()

def set_current_job(job_id):
    """Set the current job ID for this thread."""
    _thread_local.job_id = job_id

def get_current_job():
    """Get the current job ID for this thread."""
    return getattr(_thread_local, 'job_id', None)

class ProgressTqdm(tqdm.tqdm):
    """
    Custom tqdm class that updates the job progress in the global jobs dictionary.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._job_id = get_current_job()
        
    def update(self, n=1):
        super().update(n)
        if self._job_id:
            # We need to access the jobs dictionary. 
            # Since we can't easily pass it in __init__ (signature mismatch),
            # we will rely on a callback or global access.
            # For simplicity, we'll use a callback set at module level.
            if _progress_callback:
                # Calculate percentage
                # total might be None if unknown
                if self.total:
                    percentage = (self.n / self.total) * 100
                    _progress_callback(self._job_id, percentage)

_progress_callback = None

def install_hook(callback):
    """
    Install the tqdm hook.
    callback: function(job_id, percentage)
    """
    global _progress_callback
    _progress_callback = callback
    
    # Monkey-patch tqdm.tqdm
    # We patch the class in the tqdm module so anyone importing it gets our version
    tqdm.tqdm = ProgressTqdm
    
    # Also try to patch it in whisper.transcribe if possible, 
    # in case it was already imported using 'from tqdm import tqdm'
    try:
        import whisper.transcribe
        if hasattr(whisper.transcribe, 'tqdm'):
            # If it's a class
            if isinstance(whisper.transcribe.tqdm, type):
                whisper.transcribe.tqdm = ProgressTqdm
            # If it's the module
            elif hasattr(whisper.transcribe.tqdm, 'tqdm'):
                whisper.transcribe.tqdm.tqdm = ProgressTqdm
    except ImportError:
        pass
