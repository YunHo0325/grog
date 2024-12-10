import argparse
import sys
import requests
import os
import time
from urllib.parse import urlparse
import subprocess
import socket
import re
from utils.gradio_helpers import (
    build_gradio_inputs,
    build_gradio_outputs_replicate,
    create_dynamic_gradio_app,
    create_gradio_app_script,
    detect_file_type,
)
from prance import ResolvingParser
from tempfile import NamedTemporaryFile
from datetime import datetime
import shutil
from slugify import slugify


def check_nvidia_gpu():
    try:
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False


def sort_properties_by_order(properties):
    ordered_properties = sorted(
        properties.items(), key=lambda x: x[1].get("x-order", float("inf"))
    )
    return ordered_properties


def parse_api_specs(schema_url):
    schema_response = requests.get(schema_url)
    openapi_spec = schema_response.content

    with NamedTemporaryFile(mode="wb") as tmpfile:
        tmpfile.write(openapi_spec)
        tmpfile.flush()  # Ensure all data is written to the file

        parser = ResolvingParser(tmpfile.name)
        api_spec = parser.specification

    return api_spec


def parse_docker_image_data(docker_uri):
    pattern = r"/([^/]+)/([^@]+)"
    match = re.search(pattern, docker_uri)
    if match:
        first_part = match.group(1)
        second_part = match.group(2)
        return first_part, second_part
    else:
        return None, None


def wait_util_cog_ready(hostname, docker_port):
    # check http://0.0.0.0:5000/health-check {"status": "READY"} or {"status":"STARTING","setup":null}
    counter = 0
    while True:
        try:
            response = requests.get(f"http://{hostname}:{docker_port}/health-check")
            response.raise_for_status()
            if response.json()["status"] == "READY":
                print("Cog Server is ready.")
                break
            else:
                print(f"Waiting for cog server (models loading) {docker_port}...")
                counter += 1
                time.sleep(5)
                if counter >= 250:
                    raise Exception("Docker image timeout")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP Error occurred: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching data: {e}")


def wait_until_docker(hostname, docker_port):
    counter = 0
    while True:
        try:
            with socket.create_connection(
                (hostname, int(docker_port)), timeout=1
            ) as sock:
                print("Cog Docker is ready.")
                break  # Exit the loop when the server is up
        except (socket.timeout, ConnectionRefusedError):
            print(f"Waiting for cog docker to start on port {docker_port}...")
            counter += 1
            time.sleep(5)
            if counter >= 250:
                raise Exception("Docker image timeout")


def run_docker_container(docker_image, hostname, local_port):
    docker_command = [
        "docker",
        "run",
        "-d",
        "-h",
        hostname,
        "--net",
        "host",
        "-p",
        f"{local_port}:5000",
    ]
    is_nvidia_gpu_available = check_nvidia_gpu()
    print("=======is_nvidia_gpu_available=========")
    print(is_nvidia_gpu_available)
    print("================")
    if is_nvidia_gpu_available:
        docker_command.append("--gpus=all")
    #docker run -d -h 0.0.0.0 --net host -p 7250:7250 --gpus=all r8.im/fofr/face-to-sticker
    docker_command.append(docker_image)
    process = subprocess.Popen(docker_command)

    wait_until_docker(hostname, local_port)


def process_replicate_model_data(model_id):
    import requests
    from bs4 import BeautifulSoup
    import json

    try:
        url = f"https://replicate.com/{model_id}?input=docker&output=json"
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text
    except requests.exceptions.HTTPError as e:
        raise Exception(f"HTTP Error occurred: {e}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching data: {e}")

    try:
        soup = BeautifulSoup(html_content, "html.parser")
        script_tags = soup.find_all("script", {"type": "application/json"})
        data = None
        for script_tag in script_tags:
          if("initialPrediction" in script_tag.string):
            json_str = script_tag.string
            data = json.loads(json_str)
            # print(data)
            break
        if data is None:
          raise ValueError("Data with 'initialPrediction' not found in the HTML content.")
    except Exception as e:
        raise Exception(f"Failed to process model data: {str(e)}")
    
    if data["initialPrediction"] is not None:
        if isinstance(data["initialPrediction"]["output"], str):
            output_types = [detect_file_type(data["initialPrediction"]["output"])]
        elif isinstance(data["initialPrediction"]["output"], dict):
            output_types = ["json"]
        else:
            output_types = [
                detect_file_type(output)
                for output in data["initialPrediction"]["output"]
            ]
        if all(x == "list" for x in output_types):
            output_types = ["json"]
    else:
        output_types = None

    with open("data.txt", "w") as f:
        f.write(str(data["version"]))

    result = {
        # "docker_image_url": data["version"]["_extras"]["docker_image_name"],
        # "docker_image_url": "r8.im/fofr/face-to-sticker:latest",
        # "docker_image_url": "playground-v2.5-1024px-aesthetic",
        # "docker_image_url": "r8.im/playgroundai/playground-v2.5-1024px-aesthetic@sha256:a45f82a1382bed5c7aeb861dac7c7d191b0fdf74d8d57c4a0e6ed7d4d0bf7d24",
        "docker_image_url" : "r8.im/tencent/hunyuan-video@sha256:847dfa8b01e739637fc76f480ede0c1d76408e1d694b830b5dfb8e547bf98405",
        "output_types": output_types,
        "ordered_input_schema": sort_properties_by_order(
            data["version"]["_extras"]["dereferenced_openapi_schema"]["components"][
                "schemas"
            ]["Input"]["properties"]
        ),
        "example_inputs": (
            data["initialPrediction"]["input"]
            if data["initialPrediction"] is not None
            else None
        ),
        "model_name": (
            None
            if "version" not in data
            else data["version"]["_extras"]["model"]["name"]
        ),
        "model_author": (
            None
            if "version" not in data
            else data["version"]["_extras"]["model"]["owner"]
        ),
        "model_description": (
            None
            if "version" not in data
            else data["version"]["_extras"]["model"]["description"]
        ),
        "api_id": (
            None
            if "version" not in data
            else data["version"]["_extras"]["model"]["latest_version"][
                "id"
            ]
        ),
    }

    return result


def create_parser():
    # Create the parser
    parser = argparse.ArgumentParser(description="CLI tool for processing inputs.")

    # Add arguments
    parser.add_argument("--cog_url", type=str, help="The URL to process.", default=None)
    parser.add_argument(
        "--replicate_model_id", type=str, help="The Replicate model ID.", default=None
    )
    parser.add_argument(
        "--run_type",
        type=str,
        choices=["replicate_api", "local", "huggingface_spaces"],
        help="The type of run to execute.",
        default="local",
    )
    parser.add_argument(
        "--gradio_type",
        type=str,
        choices=["static", "dynamic"],
        help="The type of Gradio interface to use.",
        default="dynamic",
    )
    parser.add_argument(
        "--replicate_token", type=str, help="The Replicate API token.", default=None
    )
    parser.add_argument(
        "--huggingface_token",
        type=str,
        help="The Hugging Face API token.",
        default=None,
    )
    parser.add_argument(
        "--hostname",
        type=str,
        help="The hostname to mount the docker application (0.0.0.0).",
        default="0.0.0.0",
    )
    parser.add_argument(
        "--docker_port",
        type=int,
        help="The port to mount the docker application (default 5000).",
        default=5000,
    )
    parser.add_argument(
        "--space_hardware",
        type=str,
        help="The Hugging Face Space Hardware type.",
        default="cpu-basic",
    )
    parser.add_argument(
        "--space_repo",
        type=str,
        help="If you want a repo for your Hugging Face Space different than the name of the cog model",
        default=None,
    )
    return parser


def check_conditional_args(args):
    # Check for the conditional requirement of replicate_token or huggingface_token
    if args.run_type == "replicate_api" and not args.replicate_token:
        sys.exit(
            "Error: --replicate_token is required when run_type is 'replicate_api'"
        )
    elif args.run_type == "huggingface_spaces" and not args.huggingface_token:
        sys.exit(
            "Error: --huggingface_token is required when run_type is 'huggingface_spaces'"
        )

    # Ensure either cog_url or replicate_model_id is provided
    if not args.cog_url and not args.replicate_model_id:
        sys.exit("Error: Either --cog_url or --replicate_model_id must be provided.")

    if args.cog_url and not args.replicate_model_id:
        sys.exit(
            "Error: cog image URL isn't implemented yet. Please provide a replicate model id"
        )

    if args.run_type == "replicate_api" and not args.replicate_model_id:
        sys.exit(
            "Error: You need to use a --replicate_model_id to use the --replicate_api"
        )


def main():
    parser = create_parser()
    args = parser.parse_args()
    check_conditional_args(args)

    docker_port = str(args.docker_port)
    hostname = args.hostname
    api_id = None


    print(f"{args.gradio_type} {args.run_type}")
    print(f"{args.replicate_model_id}")

    if args.replicate_model_id:
        data = process_replicate_model_data(args.replicate_model_id)
        print(data)
        inputs, inputs_string, names = build_gradio_inputs(
            data["ordered_input_schema"], data["example_inputs"]
        )
        outputs, outputs_string = build_gradio_outputs_replicate(data["output_types"])
        model_name = data["model_name"]
        model_author = data["model_author"]
        title = f"Demo for {model_name} cog image by {data['model_author']}"
        docker_image = data["docker_image_url"]
        model_description = data["model_description"]
    # TODO for args.cog_url
    # else:
    #    docker_image = args.cog_url
    #    model_name, model_author = parse_docker_image_data(docker_image)
    #    title = f"Demo for {model_name} cog image by {model_author}"
    #    model_description = ""

    if args.run_type == "replicate_api":
        api_url = "https://api.replicate.com/v1/predictions"
        api_id = data["api_id"]
    else:
        if args.run_type == "local" and args.gradio_type == "dynamic":
            api_url = f"http://{hostname}:{docker_port}/predictions"

            if args.gradio_type == "dynamic" and not (args.run_type == "huggingface_spaces"):
                app = create_dynamic_gradio_app(
                    inputs,
                    outputs,
                    api_url=api_url,
                    api_id=api_id,
                    replicate_token=args.replicate_token,
                    title=title,
                    model_description=model_description,
                    names=names,
                    hostname=hostname,
                )
                app.launch(share=True)
                
            run_docker_container(docker_image, hostname, docker_port)
            wait_util_cog_ready(hostname, docker_port)
            # TODO for args.cog_url
            # if (args.cog_url) and not args.replicate_model_id:
            #    api_spec = parse_api_specs(
            #        f"http://localhost:{docker_port}/openapi.json"
            #    )
            #    ordered_input_schema = sort_properties_by_order(
            #        api_spec["components"]["schemas"]["Input"]["properties"]
            #    )
            #    inputs, inputs_string, names = build_gradio_inputs(ordered_input_schema)
            #    outputs TODO
        else:
            api_url = f"http://{hostname}:5000/predictions"

    if args.gradio_type == "dynamic" and not (args.run_type == "huggingface_spaces"):
        app = create_dynamic_gradio_app(
            inputs,
            outputs,
            api_url=api_url,
            api_id=api_id,
            replicate_token=args.replicate_token,
            title=title,
            model_description=model_description,
            names=names,
            hostname=hostname,
        )
        app.launch(share=True)
    else:
        app_string = create_gradio_app_script(
            inputs_string,
            outputs_string,
            api_url=api_url,
            api_id=api_id,
            replicate_token=args.replicate_token,
            title=title,
            model_description=model_description,
            local_base=(
                True
                if (args.run_type == "local" and args.gradio_type == "dynamic")
                or args.run_type == "huggingface_spaces"
                else False
            ),
            hostname=hostname,
        )

        if args.run_type == "local" or args.run_type == "huggingface_spaces":
            app_file = "app.py"
            dir_name = f"docker_{model_name}_{int(datetime.now().timestamp())}"
            os.makedirs(f"{dir_name}/utils")
            with open("docker_helpers/Dockerfile", "r") as file:
                docker_file_content = file.read()
            with open(f"{dir_name}/Dockerfile", "w") as file:
                dockerfile_image_data = f"FROM {docker_image}\n"
                file.write(dockerfile_image_data + docker_file_content)
            shutil.copy(
                "docker_helpers/requirements.txt", f"{dir_name}/requirements.txt"
            )
            shutil.copy("docker_helpers/run.sh", f"{dir_name}/run.sh")
            shutil.copy(
                "utils/gradio_helpers.py", f"{dir_name}/utils/gradio_helpers.py"
            )
            # Opening the file in write mode and writing the string
            with open(f"{dir_name}/{app_file}", "w") as file:
                file.write(app_string)
            print(
                f"Folder {dir_name} created. You can build your Dockerfile or modify the Gradio app.py"
            )

            if args.run_type == "huggingface_spaces":
                print("Uploading to Hugging Face...")
                from huggingface_hub import HfApi

                api = HfApi(token=args.huggingface_token)
                try:
                    space_id = api.create_repo(
                        repo_id=(
                            args.space_repo if args.space_repo else slugify(model_name)
                        ),
                        repo_type="space",
                        exist_ok=True,
                        space_sdk="docker",
                        space_hardware=args.space_hardware,
                        private=True,
                    )
                except:
                    raise Exception("Something went wrong with HF repo creation")
                parts = space_id.split("/")
                space_nicename = "/".join(parts[-2:])
                print(space_nicename)
                try:
                    api.upload_folder(
                        repo_id=space_nicename,
                        folder_path=f"{dir_name}",
                        repo_type="space",
                    )
                except:
                    raise Exception("Something went wrong with HF repo uploading")
                print(f"Uploaded to Hugging Face. Access it at {space_id}")
                shutil.rmtree(dir_name)

        elif args.run_type == "replicate_api":
            app_file = f"app_{model_name}_{int(datetime.now().timestamp())}.py"
            with open(app_file, "w") as file:
                file.write(app_string)
            print(
                f"\n{app_file} created. Use it with\n\npython {app_file}\n\nBe careful, your replicate API is in this file in plain text!\n"
            )


if __name__ == "__main__":
    main()
