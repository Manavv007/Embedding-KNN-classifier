from flask import Flask, jsonify, render_template, request

from classifier import CATEGORY_LABELS, SEED_DATA, load_classifier

app = Flask(__name__)
classifier = load_classifier()


@app.route("/")
def index():
    categories = [
        {"id": key, "label": label}
        for key, label in CATEGORY_LABELS.items()
    ]
    seed_count = sum(
        len(reviews)
        for subcats in SEED_DATA.values()
        for reviews in subcats.values()
    )
    return render_template(
        "index.html", categories=categories, seed_count=seed_count
    )


@app.route("/classify", methods=["POST"])
def classify_review():
    data = request.get_json(silent=True) or {}
    review = (data.get("review") or "").strip()

    if not review:
        return jsonify({"error": "Please enter a review to classify."}), 400

    try:
        result = classifier.classify(review)
        return jsonify({"review": review, **result})
    except Exception as exc:
        return jsonify({"error": f"Classification failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
