import shutil

import cv2

from ppe.dataset import dhash, leakage, load_spec, parse_label_file, scan


def test_parse_label_file_reports_every_problem(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("\n".join([
        "0 0.5 0.5 0.2 0.2",        # ok
        "0 0.5 0.5 0.2 0.2",        # duplicate
        "12 0.5 0.5 0.1 0.1",       # class out of range
        "1 0.5 0.5 0 0.1",          # zero width
        "1 0.95 0.5 0.2 0.1",       # spills over the right edge
        "1 0.5 0.5",                # too few values
        "x 0.5 0.5 0.1 0.1",        # not a number
        "",
        "5 0.1 0.1 0.2 0.2",        # ok, touching the corner
    ]))
    labels, issues = parse_label_file(f, num_classes=12)
    assert [lb.cls for lb in labels] == [0, 5]
    assert [i.problem for i in issues] == ["duplicate box", "class 12 out of range", "zero or negative size",
                                           "box outside the image", "expected 5 values, got 3", "non-numeric value"]
    assert issues[0].line == 2


def test_xyxy_conversion():
    from ppe.dataset import Label

    assert Label(0, 0.5, 0.25, 0.2, 0.5).xyxy(200, 100) == (80.0, 0.0, 120.0, 50.0)


def test_scan_counts_and_orphans(synthetic, tmp_path):
    st = scan(synthetic, "train")
    assert st.images == 12 and st.boxes == sum(st.boxes_per_image) and not st.issues
    assert st.class_counts[5] >= 12                     # at least one worker per image
    (synthetic.label_dir("train") / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    assert scan(synthetic, "train").orphan_labels == ["orphan.txt"]
    (synthetic.label_dir("train") / "orphan.txt").unlink()


def test_spec_paths(synthetic):
    assert synthetic.label_dir("val").parts[-2:] == ("labels", "val")
    assert synthetic.class_ids["Person"] == 5


def test_leakage_finds_exact_and_resized_copies(tmp_path):
    from ppe.synthetic import make

    spec = load_spec(make(tmp_path / "d", train=4, val=2))
    assert leakage(spec) == {"exact": [], "near": []}
    src = sorted(spec.image_dir("train").glob("*.jpg"))
    shutil.copy(src[0], spec.image_dir("val") / "copied.jpg")
    img = cv2.imread(str(src[1]))
    cv2.imwrite(str(spec.image_dir("val") / "resized.jpg"), cv2.resize(img, None, fx=0.6, fy=0.6),
                [cv2.IMWRITE_JPEG_QUALITY, 70])
    leak = leakage(spec)
    assert leak["exact"] == [(src[0].name, "copied.jpg")]
    assert [n[:2] for n in leak["near"]] == [(src[1].name, "resized.jpg")]


def test_dhash_differs_for_different_images(synthetic):
    a, b = sorted(synthetic.image_dir("train").glob("*.jpg"))[:2]
    assert dhash(a) != dhash(b)
