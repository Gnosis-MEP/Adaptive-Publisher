# Adaptive Publisher
A publisher for Gnosis with an early filtering system that in the future could listen for adaptation plans from the Adaptation Engine in Gnosis. Additionally it also uses the internal protocol of message communication from Gnosis


# Installation

## Configure .env
Copy the `example.env` file to `.env`, and inside the necessary variables.

## Installing Dependencies


### Using pip

To install from the `requirements.txt` file, run the following command:
```
$ pip install -r requirements.txt
```

# Running
Enter project python environment (virtualenv or conda environment)

**ps**: It's required to have the .env variables loaded into the shell so that the project can run properly. An easy way of doing this is using `pipenv shell` to start the python environment with the `.env` file loaded or using the `source load_env.sh` command inside your preferable python environment (eg: conda).

Then, run the service with:
```
$ ./adaptive_publisher/run.py
```

# Testing
Run the script `run_tests.sh`, it will run all tests defined in the **tests** and **non_mocked_tests** directory.
Run the script `perf_run_tests.sh`, it will run all tests defined in the **perf_test** directory (used to evaluate the performance of the transformers).


# Docker
## Manual Build
Build the docker image using: `docker-compose build`

**ps**: It's required to have the .env variables loaded into the shell so that the container can build properly. An easy way of doing this is using `pipenv shell` to start the python environment with the `.env` file loaded or using the `source load_env.sh` command inside your preferable python environment (eg: conda).

## Run
Use `docker-compose run --rm service` to run the docker image


# License
This projet is distributed under the AGPL license, see License file for more details.