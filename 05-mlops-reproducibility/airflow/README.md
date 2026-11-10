# Airflow up

This folder is the Day 4 clinic. The `docker-compose.yaml` file is the official Apache Airflow 3.4.0 file. It was downloaded from:

https://airflow.apache.org/docs/apache-airflow/3.4.0/docker-compose.yaml

The page that links that file is:

https://airflow.apache.org/docs/apache-airflow/3.4.0/howto/docker-compose/index.html

The file in this folder is that download. Do not edit it. The clinic changes are the `.env` file and the mounted `dags`, `logs`, `plugins`, and `config` folders, which the official file already mounts.

The official init script warns when Docker is below these floors:

- Memory, at least 4 GB. The Airflow Docker page also says to give Docker Desktop 4 GB, and ideally 8 GB, or the api server keeps restarting.
- CPUs, at least 2.
- Disk, at least 10 GB.

On Windows, install Docker Desktop and use the WSL2 backend. Give that virtual machine the same memory floor. On macOS, raise the memory in Docker Desktop's Resources settings.

Do not `pip install` Apache Airflow on the laptop. The containers already have it.

## Before the first start

From this folder (`05-mlops-reproducibility/airflow`):

```sh
mkdir -p dags logs plugins config
echo "AIRFLOW_UID=$(id -u)" > .env
```

On macOS or Windows the docs say you can ignore a missing `AIRFLOW_UID`, or copy `.env.example` to `.env`.

Pull the images once, while the network is good:

```sh
docker compose pull
```

The compose file uses these images:

- `apache/airflow:3.4.0`
- `postgres:16`
- `redis:7.2-bookworm`

To carry the images to a laptop with no network, on a machine that already pulled them:

```sh
docker save -o airflow-3.4.0-images.tar apache/airflow:3.4.0 postgres:16 redis:7.2-bookworm
```

On the other laptop:

```sh
docker load -i airflow-3.4.0-images.tar
```

## Start

```sh
docker compose up airflow-init
docker compose up
```

Wait until `docker compose ps` shows `airflow-apiserver` as healthy. The official health check is:

```sh
curl --fail http://localhost:8080/api/v2/monitor/health
```

Open http://localhost:8080. Sign in as `airflow` / `airflow`.

The official file sets `AIRFLOW__CORE__LOAD_EXAMPLES` to true. Open the Dags page, pick an example dag, unpause it, and press Trigger. Watch the grid until the tasks are green.

Stop the clinic with Ctrl-C, then:

```sh
docker compose down
```

`docker compose down --volumes` deletes the clinic database. Use that only when you want a clean clinic.

## If Docker will not start

Some laptops cannot give Docker 4 GB. That is expected. Day 5 has a plain Python runner that calls the same task functions in order, with no Airflow install. Use that runner when this clinic is not available. One working Docker machine per team is enough for the UI. Write that down if your pair splits the work.
