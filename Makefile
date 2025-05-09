IMAGE_NAME := akkadeeemikk/human_attention
CONTAINER_NAME := hattention

build:
	docker build -f docker/Dockerfile -t $(IMAGE_NAME) .

stop:
	docker stop $(CONTAINER_NAME)

jupyter:
	jupyter lab --allow-root --ip=0.0.0.0 --port=8892 --no-browser --NotebookApp.token=heatmaps

run_docker:
	docker run -it --rm \
		--ipc=host \
		--network=host \
		--gpus=all \
		-v ./:/workspace/ \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME) bash
	
enter:
	docker exec -it $(CONTAINER_NAME) bash

tuning:
	python vlm_injector_heat_tune.py --config config/qwen2.5_heat_tune.yaml