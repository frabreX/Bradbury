"""Animate an MNIST prediction with Manim Community."""

import json
import os
from pathlib import Path

import manim as mn
import numpy as np


BACKGROUND = "#000000"
GUIDE = "#41454F"
TEXT = "#F1EEE7"
MUTED = "#A3A7B2"
BLUE = "#83C8EC"
GOLD = "#FFD571"


def load_prediction(sample_index=None):
    scripts = Path(__file__).resolve().parent
    with np.load(scripts / "mnist.npz", allow_pickle=False) as dataset:
        images = dataset["x_test"]
        if sample_index is None:
            sample_index = int(np.random.default_rng().integers(len(images)))
        if not 0 <= sample_index < len(images):
            raise ValueError(f"BRADBURY_SAMPLE must be between 0 and {len(images) - 1}.")
        image = images[sample_index]
        actual = int(dataset["y_test"][sample_index])

    checkpoint = scripts / "weights.json"
    if not checkpoint.exists():
        checkpoint = scripts / "baseline_weights.json"
    weights = json.loads(checkpoint.read_text(encoding="utf-8"))
    pixels = image.reshape(-1, 1) / 255
    logits = np.asarray(weights["W1"]) @ pixels - np.asarray(weights["b1"])
    # Shifting the logits keeps the exponentials finite and preserves the probabilities.
    exponentials = np.exp(logits.ravel() - logits.max())
    return sample_index, image, actual, exponentials / exponentials.sum()


def text_at(value, x, y, size, color=TEXT, align=mn.ORIGIN):
    return mn.Text(value, font_size=size, color=color).move_to(
        (x, y, 0), aligned_edge=align
    )


def pixel_art(image):
    dark = np.array([20, 25, 34])
    bright = np.array([131, 200, 236])
    rgb = (dark + image[..., None] / 255 * (bright - dark)).astype(np.uint8)
    # Give the individual source pixels visible boundaries when enlarged.
    grid = rgb.repeat(8, axis=0).repeat(8, axis=1)
    grid[::8, :] = dark
    grid[:, ::8] = dark
    if mn.config.renderer == mn.RendererType.OPENGL:
        return mn.opengl.OpenGLImageMobject(grid)
    return mn.ImageMobject(grid)


class DigitClassifier(mn.Scene):
    def construct(self):
        if mn.config.renderer == mn.RendererType.OPENGL:
            self.renderer.background_color = BACKGROUND
        else:
            self.camera.background_color = BACKGROUND
        sample = os.environ.get("BRADBURY_SAMPLE")
        try:
            requested_index = int(sample) if sample is not None else None
        except ValueError as exc:
            raise ValueError("BRADBURY_SAMPLE must be an integer.") from exc

        sample_index, image, actual, probabilities = load_prediction(requested_index)
        predicted = int(probabilities.argmax())

        title = text_at("Cypher Identifier", 0, 3.15, 46)
        self.play(mn.FadeIn(title, shift=0.12 * mn.UP), run_time=0.65)
        self.show_input(image, sample_index)
        self.show_scores(probabilities, predicted)

        result = mn.VGroup(
            mn.Text("Prediction", font_size=27, color=TEXT),
            mn.Text(str(predicted), font_size=34, color=GOLD),
            mn.Text("=" if actual == predicted else "≠", font_size=27, color=MUTED),
            mn.Text(str(actual), font_size=34, color=BLUE),
            mn.Text("True label", font_size=27, color=TEXT),
        ).arrange(mn.RIGHT, buff=0.23).move_to((0, -3.22, 0))
        self.play(mn.FadeIn(result, shift=0.12 * mn.UP), run_time=0.6)
        self.wait(2)

    def show_input(self, image, sample_index):
        image_x = -4.45
        digit = pixel_art(image).scale_to_fit_height(3.25).move_to((image_x, 0.05, 0))
        frame = mn.SurroundingRectangle(digit, buff=0.02, color=GUIDE, stroke_width=1.4)
        caption = mn.VGroup(
            text_at("28 × 28 pixels", image_x, -1.98, 25),
            text_at(f"MNIST sample {sample_index}", image_x, -2.39, 18, MUTED),
        )
        self.play(mn.FadeIn(digit), mn.Create(frame), mn.FadeIn(caption), run_time=0.85)

        equation = mn.Text(
            "W x − b", font_size=32, color=TEXT
        ).move_to((-0.97, 0.47, 0))
        arrow = mn.Arrow(
            (-2.45, 0.03, 0),
            (0.51, 0.03, 0),
            buff=0,
            color=MUTED,
            stroke_width=2.5,
            tip_length=0.19,
        )
        softmax = text_at("softmax", -0.97, -0.37, 23, MUTED)
        self.play(mn.FadeIn(equation), mn.GrowArrow(arrow), run_time=0.65)
        self.play(mn.FadeIn(softmax), run_time=0.35)

    def show_scores(self, probabilities, predicted):
        labels = mn.VGroup()
        score_animations = []
        winner = []
        bar_start, bar_span = 2.02, 3.25
        first_y, row_spacing = 1.43, 0.365

        for digit, probability in enumerate(probabilities):
            y = first_y - digit * row_spacing
            label = text_at(str(digit), 1.68, y, 21, TEXT)
            percentage = "<0.1%" if 0 < probability < 0.001 else f"{probability:.1%}"
            value = text_at(percentage, 6.48, y, 19, MUTED, mn.RIGHT)
            labels.add(label)
            row_objects = [label, value]
            reveal = [mn.FadeIn(value)]
            if probability >= 0.001:
                bar = mn.Rectangle(
                    width=bar_span * probability,
                    height=0.15,
                    stroke_width=0,
                    fill_color=BLUE,
                    fill_opacity=1,
                ).move_to((bar_start, y, 0), aligned_edge=mn.LEFT)
                reveal.append(mn.GrowFromPoint(bar, bar.get_left()))
                row_objects.append(bar)
            if digit == predicted:
                winner = row_objects
            score_animations.append(mn.AnimationGroup(*reveal))

        origin = (bar_start, -2.10, 0)
        axis = mn.VGroup(
            mn.Line((bar_start, 1.67, 0), origin, color=GUIDE, stroke_width=1.4),
            mn.Line(origin, (bar_start + bar_span, -2.10, 0), color=GUIDE, stroke_width=1.4),
            text_at("0%", bar_start, -2.35, 16, MUTED),
            text_at("100%", bar_start + bar_span, -2.35, 16, MUTED),
        )
        heading = text_at("Class probabilities", 3.85, 2.00, 27)
        self.play(mn.FadeIn(heading), mn.FadeIn(labels), mn.FadeIn(axis), run_time=0.55)
        self.play(mn.LaggedStart(*score_animations, lag_ratio=0.10), run_time=1.55)
        self.play(*(item.animate.set_color(GOLD) for item in winner), run_time=0.45)
