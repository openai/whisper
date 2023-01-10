import os
import argparse


def cli():
    from . import _MODELS, _download, available_models

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default="small", choices=available_models(), help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisper by default")

    args = parser.parse_args().__dict__

    download_root = args["model_dir"]
    if download_root is None:
        download_root = os.getenv(
            "XDG_CACHE_HOME", 
            os.path.join(os.path.expanduser("~"), ".cache", "whisper")
        )

    _download(_MODELS[args["model"]], download_root, False)
