PY ?= python3
export PYTHONPATH := src
MODEL ?= runs/ppe/weights/best.onnx
EPOCHS ?= 15
IMGSZ ?= 416

.PHONY: install install-train data validate demo test lint serve predict train export eval clean

install:
	$(PY) -m pip install -r requirements-dev.txt

install-train:
	$(PY) -m pip install -r requirements-train.txt

## Construction-PPE (CC BY 4.0, ~650 MB, checksum-verified) into data/
data:
	$(PY) -m ppe download

validate: data
	$(PY) -m ppe validate

## Dataset analysis + compliance engine on ground truth + alert replay → docs/images, reports/summary.json
demo: validate
	$(PY) -m ppe demo

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

serve:
	PPE_MODEL=$(MODEL) $(PY) -m uvicorn ppe.api:app --port 8000

predict:
	$(PY) -m ppe predict data/construction-ppe/images/val --model $(MODEL)

## Needs requirements-train.txt (PyTorch + Ultralytics, AGPL-3.0). The CI train workflow runs the same commands.
train: data
	$(PY) -m ppe data-yaml --out runs/data.yaml
	yolo detect train data=runs/data.yaml model=yolov8n.pt epochs=$(EPOCHS) imgsz=$(IMGSZ) batch=16 device=cpu \
		workers=2 project=runs name=ppe exist_ok=True seed=0 deterministic=True

export:
	yolo export model=runs/ppe/weights/best.pt format=onnx imgsz=$(IMGSZ) opset=12 simplify=False

eval:
	yolo detect val model=runs/ppe/weights/best.pt data=runs/data.yaml imgsz=$(IMGSZ) device=cpu project=runs name=val exist_ok=True
	$(PY) -m ppe eval-onnx --model $(MODEL)

clean:
	rm -rf runs reports
