# ==============================================================================
# Attribution
# ------------------------------------------------------------------------------
# This file is derived from RoboChallengeInference:
#   https://github.com/RoboChallenge/RoboChallengeInference.git
#
# We vendor this code for RoboChallenge integration.
# Local modifications: none.
# ==============================================================================

import time
import logging

def process_job(client, gpu_client, job_id, robot_id, image_size, image_type, action_type, duration, max_wait=600):
    """
    Handles the processing of a single job, including status checking, robot starting,
    state polling, inference, and action posting.

    Args:
        client: An instance of the InterfaceClient for interacting with the job system.
        gpu_client: An inference client that wraps the policy/model for decision-making.
        job_id (str): The unique identifier for the job to process.
        robot_id (str): The unique identifier for the robot associated with the job.
        image_size (list): The size of images to request from the robot (e.g., [224, 224]).
        image_type (list): The types of images to request (e.g., ["high", "left_hand", "right_hand"]).
        action_type (str): The type of action to perform (e.g., "joint").
        duration (float): The duration for each action command.
        max_wait (int, optional): Maximum time to wait for job completion in seconds. Defaults to 600.

    Notes:
        - This function should not be modified by users.
        - It performs the main job loop for a single job, including error handling and logging.
        - For more details about parameters, see README.md.
    """
    try:
        device, status = client.get_job_status(job_id)
        logging.info(f"Processing job_id: {job_id}, status: {status}")
        if status == "ready":
            client.update_job_info(job_id, robot_id)
            r = client.start_robot(job_id)
            logging.info(f"Started robot: {r.content}")
            if r.status_code == 200:
                start_time = time.time()
                while True:
                    device, status = client.get_job_status(job_id)
                    if status != "running":
                        break
                    state = client.get_state(image_size, image_type, action_type)
                    if not state:
                        time.sleep(0.5)
                        continue
                    if state['state'] == "size_none":
                        client.post_size()
                        time.sleep(0.5)
                        continue
                    if state['state'] != "normal" or state['pending_actions'] != 0:
                        time.sleep(0.5)
                        continue
                    logging.info("get_robot_state time: %.2f", time.time() - state['timestamp'])
                    # `executor.infer()` requires `state["job_id"]` (used for per-job output/logging),
                    # but the robot state payload may not include it, so we inject it here.
                    if isinstance(state, dict) and "job_id" not in state:
                        state["job_id"] = job_id
                    result = gpu_client.infer(state)
                    logging.info(f"Inference result: {result}")
                    client.post_actions(result, duration, action_type)
                    if time.time() - start_time > max_wait:
                        logging.warning(f"Job {job_id} exceeded max wait time.")
                        break
    except Exception as e:
        logging.error(f"Error processing job {job_id}: {e}")


def job_loop(client, gpu_client, job_collection_id, image_size, image_type, action_type, duration):
    """
    Main loop for polling and processing all jobs in a job collection.

    Args:
        client: An instance of the InterfaceClient for interacting with the job system.
        gpu_client: An inference client that wraps the policy/model for decision-making.
        job_collection_id (str): The unique identifier for the job collection to monitor.
        image_size (list): The size of images to request from the robot.
        image_type (list): The types of images to request.
        action_type (str): The type of action to perform.
        duration (float): The duration for each action command.

    Notes:
        - This function repeatedly polls the job collection for active jobs.
        - It processes jobs with status "ready" by calling process_job.
        - The loop exits if no active jobs are found after several consecutive polls.
        - This function should not be modified by users.
        - For more details about parameters, see README.md.
    """
    ACTIVE_STATES = ["assigned", "prepare", "ready", "running"]
    MAX_EMPTY_POLLS = 10
    empty_poll_count = 0

    while True:
        job_collection = client.get_all_jobs(job_collection_id)
        jobs = job_collection["jobs"]

        has_active_job = False
        exit_code = 0
        for job in jobs:
            status = job["status"]
            if status in ACTIVE_STATES:
                has_active_job = True
                break
            elif status in ["finished", "cancelled", "failed"]:
                exit_code += 1

        if not has_active_job and exit_code == len(jobs):
            empty_poll_count += 1
            logging.info(f"No active jobs, poll count: {empty_poll_count}")
            if empty_poll_count >= MAX_EMPTY_POLLS:
                logging.info("No new jobs after multiple checks, exiting.")
                break
            time.sleep(1)
            continue
        else:
            empty_poll_count = 0

        for job in jobs:
            job_id = job["job_id"]
            robot_id = job["device"]["robot_id"]
            status = job["status"]
            logging.info(f"Job id: {job_id}, status: {status}, remaining jobs: {len(jobs)}")
            if status == "ready":
                process_job(client, gpu_client, job_id, robot_id, image_size, image_type, action_type, duration)

        time.sleep(1)
