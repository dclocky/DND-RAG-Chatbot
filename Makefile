.PHONY: run evaluate qa docker

run:
	python main.py

evaluate:
	python -m src.cli evaluate

qa:
	python -m src.cli qa

docker:
	docker build -t dnd-rag-assistant .
	docker run -p 8080:8080 dnd-rag-assistant
