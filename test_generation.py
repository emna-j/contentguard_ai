
import multiprocessing
import sys
from pathlib import Path
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))

    print("=" * 60)
    print("ContentGuard AI - Generation System Test")
    print("=" * 60)

    print("\n[1/5] Testing module imports...")
    try:
        from generation.generator_engine import get_generator
        from generation.models import PriorityLevel

        print(" Modules imported successfully")
    except Exception as e:
        print(f" Import error: {e}")
        sys.exit(1)

    print("\n[2/5] Initializing generator engine...")
    try:
        generator = get_generator()
        print(" Generator initialized")
        print(f"   - Ray workers: {len(generator.ray_workers)}")
        print(f"   - Thread pool: active")
        print(f"   - Process pool: active")
        print(f"   - Consumer threads: {len(generator.consumer_threads)}")
    except Exception as e:
        print(f" Initialization error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n[3/5] Submitting test task...")
    try:
        task_id = generator.submit(
            prompt="A beautiful sunset over mountains",
            style="realistic",
            quality="high",
            priority=PriorityLevel.HIGH
        )
        print(f" Task submitted: {task_id}")
    except Exception as e:
        print(f" Task submission error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n[4/5] Monitoring task status...")
    max_wait = 120  # 2 minutes max
    start_time = time.time()
    last_status = None

    try:
        while time.time() - start_time < max_wait:
            status = generator.get_status(task_id)
            if status:
                current_status = status.get('status')
                progress = status.get('progress', 0)

                if current_status != last_status:
                    print(f"   Status: {current_status} ({progress}%)")
                    last_status = current_status

                if current_status == "completed":
                    print(f" Task completed!")
                    print(f"   Filename: {status.get('filename')}")
                    print(f"   Time: {time.time() - start_time:.1f}s")
                    break
                elif current_status == "failed":
                    print(f" Task failed: {status.get('error')}")
                    break

            time.sleep(2)
        else:
            print("  Task timeout (2 minutes)")
    except KeyboardInterrupt:
        print("\n  Test interrupted by user")
    except Exception as e:
        print(f" Monitoring error: {e}")
        import traceback

        traceback.print_exc()

    print("\n[5/5] Shutting down...")
    try:
        generator.shutdown()
        print(" Clean shutdown completed")
    except Exception as e:
        print(f"⚠  Shutdown warning: {e}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)