import runpod
import os
import logging
import requests
import tempfile
import shutil # For rmtree
from runpod.serverless.utils import rp_cleanup # For potential cleanup later

# Attempt to import the inference function
try:
    from hymm_sp.inference_runner import process_video_avatar_job
except ImportError as e:
    # This might happen if PYTHONPATH is not set correctly in the environment
    # or if there's an issue within inference_runner.py itself.
    logging.critical(f"Failed to import process_video_avatar_job: {e}. Ensure PYTHONPATH is set and inference_runner.py is correct.")
    # We can't proceed if this fails, so raise an error or define a dummy for syntax checking
    def process_video_avatar_job(*args, **kwargs):
        raise RuntimeError("process_video_avatar_job could not be imported.")

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log expected environment variables (inference_runner.py will use these)
MODEL_BASE_ENV = os.environ.get("MODEL_BASE", "./weights")
CPU_OFFLOAD_ENV = os.environ.get("CPU_OFFLOAD", "1")
# These are just for logging in the handler; inference_runner directly uses os.environ.get
logger.info(f"Handler starting. Expected MODEL_BASE: {MODEL_BASE_ENV} (used by inference_runner)")
logger.info(f"Handler starting. Expected CPU_OFFLOAD: {CPU_OFFLOAD_ENV} (used by inference_runner)")


def download_file(url, prefix):
    logger.info(f"Attempting to download file from URL: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        file_extension = os.path.splitext(url)[1]
        if not file_extension:
            file_extension = '.tmp'

        # Create temp file in a specific directory if needed, default is fine for now
        temp_file = tempfile.NamedTemporaryFile(delete=False, prefix=prefix, suffix=file_extension, mode='wb')
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        temp_file.close()
        logger.info(f"Successfully downloaded {url} to {temp_file.name}")
        return temp_file.name
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        raise


def handler(job):
    job_id = job.get('id', 'local_test_job_id') # Use a default if id is not present
    logger.info(f"Received job: {job_id}")

    job_input = job.get('input', {})
    image_url = job_input.get('image_url')
    audio_url = job_input.get('audio_url')

    logger.info(f"Image URL: {image_url}")
    logger.info(f"Audio URL: {audio_url}")

    if not image_url or not audio_url:
        logger.warning(f"Job {job_id}: Missing image_url or audio_url in input.")
        return {"error": "Missing image_url or audio_url in input."}

    temp_image_path = None
    temp_audio_path = None
    output_dir = None # Define output_dir here for broader scope in finally/except

    try:
        logger.info(f"Job {job_id}: Downloading image...")
        temp_image_path = download_file(image_url, f"image_{job_id}_")
        logger.info(f"Job {job_id}: Image downloaded to: {temp_image_path}")

        logger.info(f"Job {job_id}: Downloading audio...")
        temp_audio_path = download_file(audio_url, f"audio_{job_id}_")
        logger.info(f"Job {job_id}: Audio downloaded to: {temp_audio_path}")

        # Create a temporary directory for the output of this specific job
        output_dir = tempfile.mkdtemp(prefix=f"hva_output_{job_id}_")
        logger.info(f"Job {job_id}: Created temporary output directory: {output_dir}")

        # Ensure inference_runner can find MODEL_BASE and CPU_OFFLOAD from the environment
        # No need to set them via os.environ here if they are already set in the Docker env.
        # inference_runner.py reads them directly.

        logger.info(f"Job {job_id}: Starting video generation process...")
        generated_video_path = process_video_avatar_job(
            image_path=temp_image_path,
            audio_path=temp_audio_path,
            output_base_path=output_dir, # Pass the created directory
            job_id=job_id
        )
        logger.info(f"Job {job_id}: Video generation complete. Output video at: {generated_video_path}")

        # In a real scenario, you might upload generated_video_path to S3
        # and then return the S3 URL. The file itself could then be cleaned up.
        # For now, returning the path. RunPod might offer a way to serve/access this.
        return {
            "message": "Video generated successfully.",
            "video_path": generated_video_path,
            # "debug_input_image_path": temp_image_path, # Keep for debugging if needed
            # "debug_input_audio_path": temp_audio_path  # Keep for debugging if needed
        }

    except Exception as e:
        logger.error(f"Job {job_id}: An error occurred during processing: {str(e)}", exc_info=True)
        # Cleanup output directory if it was created and an error occurred
        if output_dir and os.path.exists(output_dir):
            logger.info(f"Job {job_id}: Cleaning up output directory due to error: {output_dir}")
            try:
                shutil.rmtree(output_dir)
            except OSError as rm_error:
                logger.error(f"Job {job_id}: Error removing output directory {output_dir}: {rm_error}")
        return {
            "error": f"Failed to process video for job {job_id}: {str(e)}"
        }
    finally:
        # This block executes regardless of success or failure, for cleanup of inputs.
        # The generated video in output_dir is intentionally NOT cleaned up here if successful,
        # as its path is returned. If an error occurred, output_dir is cleaned in the except block.
        logger.info(f"Job {job_id}: Cleaning up temporary input files...")
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                logger.info(f"Job {job_id}: Removed temporary image file: {temp_image_path}")
            except OSError as e_rm_img:
                logger.warning(f"Job {job_id}: Could not remove temporary image file {temp_image_path}: {e_rm_img}")

        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                logger.info(f"Job {job_id}: Removed temporary audio file: {temp_audio_path}")
            except OSError as e_rm_aud:
                logger.warning(f"Job {job_id}: Could not remove temporary audio file {temp_audio_path}: {e_rm_aud}")


if __name__ == "__main__":
    # This block is for local testing. RunPod calls `handler` directly.
    logger.info("Starting RunPod handler for local testing...")

    # Example of how you might test locally (requires actual URLs and models):
    # Ensure MODEL_BASE and CPU_OFFLOAD are set in your local environment if testing this way.
    # print(f"To test locally, ensure MODEL_BASE is set (currently: {MODEL_BASE_ENV})")
    # print(f"and necessary model files are present.")

    # test_job_payload = {
    #     "id": "local_test_001",
    #     "input": {
    #          # Use very small, publicly accessible image and audio files for testing download
    #         "image_url": "https://raw.githubusercontent.com/runpod/runpod-python/main/docs/images/runpod-logo-gradient.png",
    #         "audio_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3" # Example small mp3
    #     }
    # }
    # result = handler(test_job_payload)
    # print(f"Local test handler result: {result}")

    # For RunPod, the serverless worker starts the handler.
    runpod.serverless.start({"handler": handler})
