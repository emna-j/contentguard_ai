
import multiprocessing
import sys
from pathlib import Path

if __name__ == '__main__':
    multiprocessing.freeze_support()


    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    current_dir = Path(__file__).parent
    frontend_dir = current_dir / 'frontend'
    sys.path.insert(0, str(current_dir))

    from app import app, logger

    port = 5001
    logger.info("=" * 60)
    logger.info("ContentGuard AI Server Starting...")
    logger.info(f"Server URL: http://localhost:{port}")
    logger.info("=" * 60)

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)