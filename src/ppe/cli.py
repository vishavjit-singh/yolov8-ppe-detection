"""``python -m ppe <command>``"""
from __future__ import annotations

import json
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = ROOT / "config" / "dataset.yaml"
POLICY = ROOT / "config" / "site_policy.yaml"


@click.group()
def cli():
    """PPE detection: dataset tooling, compliance engine, ONNX inference."""


@cli.command()
@click.option("--dest", default=str(ROOT / "data"), type=click.Path(file_okay=False))
def download(dest):
    """Download Construction-PPE (CC BY 4.0, ~650 MB) and verify its checksum."""
    from .data import download as dl

    click.echo(f"dataset at {dl(Path(dest))}")


@cli.command("data-yaml")
@click.option("--data", "data_yaml", default=str(DATA_YAML))
@click.option("--out", default=str(ROOT / "runs" / "data.yaml"))
def data_yaml_cmd(data_yaml, out):
    """Write a copy of data.yaml with an absolute `path` (Ultralytics resolves relative paths against its own
    datasets directory, not the YAML's folder)."""
    import yaml

    from .dataset import load_spec

    spec = load_spec(data_yaml)
    raw = yaml.safe_load(Path(data_yaml).read_text())
    raw["path"] = str(spec.root)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(yaml.safe_dump(raw, sort_keys=False))
    click.echo(out)


@cli.command()
@click.option("--data", "data_yaml", default=str(DATA_YAML))
@click.option("--leakage/--no-leakage", default=True, help="Also look for train/val duplicates (dHash).")
def validate(data_yaml, leakage):
    """Validate labels and splits; exit 1 on label errors or exact train/val duplicates."""
    from .dataset import leakage as find_leakage
    from .dataset import load_spec, scan

    spec = load_spec(data_yaml)
    bad = False
    for split in spec.splits:
        st = scan(spec, split)
        click.echo(f"{split}: {st.images} images, {st.boxes} boxes, {len(st.issues)} label issues, "
                   f"{st.images_without_labels} unlabelled images, {len(st.orphan_labels)} orphan label files")
        for issue in st.issues[:20]:
            click.echo(f"  {issue.file}:{issue.line} {issue.problem}")
        bad |= bool(st.issues)
    if leakage and {"train", "val"} <= set(spec.splits):
        leak = find_leakage(spec)
        click.echo(f"train/val: {len(leak['exact'])} identical files, {len(leak['near'])} near-duplicates "
                   f"(dHash distance ≤ 4)")
        for a, b, d in leak["near"]:
            click.echo(f"  near-duplicate: train/{a} ~ val/{b} (distance {d})")
        bad |= bool(leak["exact"])
    raise SystemExit(1 if bad else 0)


@cli.command()
@click.option("--data", "data_yaml", default=str(DATA_YAML))
@click.option("--policy", default=str(POLICY))
@click.option("--out", default=str(ROOT / "docs" / "images"))
@click.option("--summary", "summary_path", default=str(ROOT / "reports" / "summary.json"))
def demo(data_yaml, policy, out, summary_path):
    """Dataset analysis + compliance on ground truth + alert replay → docs/images and reports/summary.json."""
    from . import demo as run_demo
    from .compliance import load_policy
    from .dataset import load_spec

    s = run_demo.run(load_spec(data_yaml), load_policy(policy), Path(out), Path(summary_path))
    click.echo(json.dumps(s, indent=2))


@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.option("--model", required=True, type=click.Path(exists=True, dir_okay=False), help="YOLOv8 ONNX export")
@click.option("--data", "data_yaml", default=str(DATA_YAML), help="for class names")
@click.option("--policy", default=str(POLICY))
@click.option("--conf", default=0.25)
@click.option("--out", default=str(ROOT / "runs" / "predict"))
@click.option("--every", default=1, help="video: process every n-th frame")
def predict(source, model, data_yaml, policy, conf, out, every):
    """Run an ONNX model on an image, a folder of images or a video; write annotated output + JSON."""
    import cv2

    from .alerts import AlertDebouncer
    from .compliance import VIOLATION, evaluate, load_policy
    from .dataset import IMAGE_EXTS, load_spec
    from .draw import annotate
    from .inference import OnnxDetector

    spec, pol = load_spec(data_yaml), load_policy(policy)
    det = OnnxDetector(model, spec.names, conf=conf)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(source)
    results = []
    if src.is_dir() or src.suffix.lower() in IMAGE_EXTS:
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS) if src.is_dir() else [src]
        for f in files:
            img = cv2.imread(str(f))
            dets = det(img)
            v = evaluate(dets, pol)
            cv2.imwrite(str(out / f.name), annotate(img, v))
            results.append({"file": f.name, "detections": [d.to_dict() for d in dets], **v.to_dict()})
    else:
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        deb = AlertDebouncer(pol.raise_after, pol.window, pol.clear_after)
        writer, i = None, -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % every:
                continue
            v = evaluate(det(frame), pol)
            reasons = {}
            for p in v.people:
                for k, s in p.items_required.items():
                    if s == VIOLATION:
                        reasons[k] = reasons.get(k, 0) + 1
            if deb.update(i, v.violations, reasons):
                click.echo(f"frame {i}: alert {'raised' if deb.active else 'cleared'}")
            frame = annotate(frame, v)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out / f"{src.stem}_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps / every, (w, h))
            writer.write(frame)
        deb.close(i)
        if writer is not None:
            writer.release()
        results.append({"file": src.name, "frames": i + 1, "alerts": [e.to_dict(fps) for e in deb.events]})
    (out / "results.json").write_text(json.dumps(results, indent=2))
    click.echo(f"wrote {len(results)} result(s) to {out}")


@cli.command("eval-onnx")
@click.option("--model", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--data", "data_yaml", default=str(DATA_YAML))
@click.option("--split", default="val")
@click.option("--policy", default=str(POLICY))
@click.option("--conf", default=0.001, help="low threshold for mAP, as in `yolo val`")
@click.option("--out", default=str(ROOT / "reports" / "onnx_eval.json"))
def eval_onnx(model, data_yaml, split, policy, conf, out):
    """mAP of the exported ONNX model through this repo's own pre/post-processing, plus compliance agreement."""
    import cv2

    from .compliance import evaluate, load_policy
    from .dataset import images, load_spec
    from .gt import gt_detections
    from .inference import OnnxDetector
    from .metrics import evaluate as map_eval

    spec, pol = load_spec(data_yaml), load_policy(policy)
    det = OnnxDetector(model, spec.names, conf=conf, iou=0.6)
    preds, gts, agree, total = [], [], 0, 0
    for img_path in images(spec, split):
        img = cv2.imread(str(img_path))
        p = det(img)
        g = gt_detections(spec, split, img_path)
        preds.append(p)
        gts.append(g)
        vp = evaluate([d for d in p if d.score >= 0.25], pol)
        vg = evaluate(g, pol)
        total += 1
        agree += int(vp.status == vg.status)
    m = map_eval(preds, gts, len(spec.names))
    m["per_class"] = {spec.names[c]: v for c, v in m["per_class"].items()}
    m["frame_verdict_agreement_with_ground_truth"] = round(agree / max(total, 1), 4)
    m["images"] = total
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(m, indent=2))
    click.echo(json.dumps({k: v for k, v in m.items() if k != "per_class"}, indent=2))


def main():
    cli()


if __name__ == "__main__":
    main()
