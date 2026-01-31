from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.inference import InstrumentClassifier
from src.gui.predictor import AudioConfig

DEFAULT_WEIGHTS = Path("saved_weights/chinese_single_label/train_12_class/best_val_acc.pt")

try:
    CALM_SEAFOAM_THEME = gr.themes.ThemeClass.from_hub("gradio/calm_seafoam")
except AttributeError:
    CALM_SEAFOAM_THEME = gr.themes.Theme.from_hub("gradio/calm_seafoam")
except Exception:
    CALM_SEAFOAM_THEME = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="blue",
        neutral_hue="gray",
    )


def _build_classifier() -> InstrumentClassifier:
    audio_cfg = AudioConfig(
        sample_rate=44100,
        clip_duration=3.0,
        n_mels=128,
        win_ms=30.0,
        hop_ms=10.0,
        fmin=20.0,
        fmax=None,
    )
    return InstrumentClassifier(default_weights=DEFAULT_WEIGHTS, audio_config=audio_cfg)


classifier = _build_classifier()


def _mel_to_figure(mel: np.ndarray, cfg: AudioConfig) -> plt.Figure:
    mel_avg = mel.mean(axis=0)
    hop = int(round(cfg.sample_rate * (cfg.hop_ms / 1000.0)))
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    img = librosa.display.specshow(
        mel_avg,
        sr=cfg.sample_rate,
        hop_length=hop,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=ax,
    )
    ax.set_title("Averaged Mel-Spectrogram (2 channels → mono)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def _format_prediction_table(predictions: List[Tuple[str, float]]) -> pd.DataFrame:
    rows = [
        {"Rank": idx + 1, "Label": label, "Probability": prob}
        for idx, (label, prob) in enumerate(predictions)
    ]
    return pd.DataFrame(rows)


def _run_inference(
    audio_path: Optional[str],
    weights_path: Optional[str],
    cache_mel: bool,
):
    if not audio_path:
        return (
            "⚠️ Please provide an audio clip before running inference.",
            {},
            None,
            pd.DataFrame(columns=["Rank", "Label", "Probability"]),
            "",
        )

    try:
        predictions, mel, mel_path, weights = classifier.predict(
            audio_path,
            weights_path=weights_path,
            save_mel=cache_mel,
        )
    except Exception as exc:
        return (
            f"❌ Error during inference: `{exc}`",
            {},
            None,
            pd.DataFrame(columns=["Rank", "Label", "Probability"]),
            "",
        )

    label_probs = {label: prob for label, prob in predictions}
    top_label, top_prob = predictions[0]
    status = (
        f"✅ Top prediction: **{top_label}** ({top_prob:.1%})  \n"
        f"Using weights: `{weights}`  \n"
        f"Device: `{classifier.device.type}`"
    )

    fig = _mel_to_figure(mel, classifier.audio_config)
    table = _format_prediction_table(predictions)
    mel_path_str = mel_path.as_posix() if mel_path else "Not cached to disk"
    return status, label_probs, fig, table, mel_path_str


def _run_long_inference(
    audio_path: Optional[str],
    weights_path: Optional[str],
    chunk_duration: float,
    stride: float,
):
    empty_chunk_df = pd.DataFrame(
        columns=[
            "chunk",
            "start_s",
            "end_s",
            "top_label",
            "top_prob",
            "rank2_label",
            "rank2_prob",
            "rank3_label",
            "rank3_prob",
        ]
    )
    empty_counts_df = pd.DataFrame(columns=["Label", "Count", "Fraction"])

    if not audio_path:
        return (
            "⚠️ Please upload an audio file before running the analysis.",
            empty_chunk_df,
            empty_counts_df,
        )

    try:
        chunk_results, counts, weights = classifier.predict_long_audio(
            audio_path,
            weights_path=weights_path,
            chunk_duration=chunk_duration,
            stride=stride,
        )
    except Exception as exc:
        return (
            f"❌ Error during long-audio analysis: `{exc}`",
            empty_chunk_df,
            empty_counts_df,
        )

    if not chunk_results:
        return (
            "⚠️ No chunks were generated from the provided audio.",
            empty_chunk_df,
            empty_counts_df,
        )

    chunk_df = pd.DataFrame(chunk_results)
    prob_columns = [col for col in chunk_df.columns if col.endswith("_prob")]
    for col in prob_columns:
        chunk_df[col] = chunk_df[col].map(lambda x: round(float(x), 4))

    counts_rows: List[Dict[str, float | str]] = []
    total_chunks = len(chunk_results)
    for label, count in counts.most_common():
        counts_rows.append(
            {
                "Label": label,
                "Count": count,
                "Fraction": round(count / total_chunks, 4),
            }
        )
    counts_df = pd.DataFrame(counts_rows, columns=["Label", "Count", "Fraction"])

    stride_eff = stride if stride and stride > 0 else chunk_duration
    top_summary = ""
    if counts:
        top_label, top_count = counts.most_common(1)[0]
        top_summary = f"Top label occurrences: **{top_label}** ({top_count}/{total_chunks})."

    status = (
        f"✅ Processed {total_chunks} chunks "
        f"(chunk={chunk_duration:.2f}s, stride={stride_eff:.2f}s).  \n"
        f"Using weights: `{weights}`"
    )
    if top_summary:
        status += f"  \n{top_summary}"

    return status, chunk_df, counts_df


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="GUI", theme=CALM_SEAFOAM_THEME) as demo:
        gr.Markdown(
            "#Instrument Classifier\n"
            "Upload short clips for a quick prediction or analyse longer recordings "
            "chunk-by-chunk with the fine-tuned CNN."
        )
        with gr.Tab("Model"):
            with gr.Row():
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Audio Input (3 seconds recommended)",
                )
                with gr.Column():
                    weights_input = gr.Textbox(
                        value=DEFAULT_WEIGHTS.as_posix(),
                        label="Checkpoint Path",
                        placeholder="Path to .pt checkpoint",
                    )
                    cache_checkbox = gr.Checkbox(
                        value=True,
                        label="Cache mel as .npy under .cache/gui_mels",
                    )
                    run_button = gr.Button("Run Prediction", variant="primary")

            status_md = gr.Markdown()
            probs_label = gr.Label(label="Class Probabilities")
            with gr.Row():
                mel_plot = gr.Plot(label="Mel-Spectrogram")
                preds_table = gr.Dataframe(
                    headers=["Rank", "Label", "Probability"],
                    label="Detailed Probabilities",
                    datatype=["number", "str", "number"],
                )
            mel_path_box = gr.Textbox(
                label="Mel Cache (.npy)",
                placeholder="Will appear after running inference",
                interactive=False,
            )

            run_button.click(
                fn=_run_inference,
                inputs=[audio_input, weights_input, cache_checkbox],
                outputs=[status_md, probs_label, mel_plot, preds_table, mel_path_box],
            )

        with gr.Tab("Long Audio"):
            with gr.Row():
                long_audio_input = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="Audio Input (any length, mp3/wav)",
                )
                with gr.Column():
                    long_weights_input = gr.Textbox(
                        value=DEFAULT_WEIGHTS.as_posix(),
                        label="Checkpoint Path",
                        placeholder="Path to .pt checkpoint",
                    )
                    chunk_duration_slider = gr.Slider(
                        minimum=1.0,
                        maximum=12.0,
                        step=0.5,
                        value=3.0,
                        label="Chunk Duration (seconds)",
                        info="Each window will be padded/truncated to this duration.",
                    )
                    stride_slider = gr.Slider(
                        minimum=0.5,
                        maximum=12.0,
                        step=0.5,
                        value=3.0,
                        label="Stride (seconds)",
                        info="Advance between consecutive chunks. Set equal to chunk duration for non-overlapping windows.",
                    )
                    analyze_button = gr.Button("Analyse Long Audio", variant="primary")

            long_status_md = gr.Markdown()
            long_chunk_table = gr.Dataframe(
                headers=[
                    "chunk",
                    "start_s",
                    "end_s",
                    "top_label",
                    "top_prob",
                    "rank2_label",
                    "rank2_prob",
                    "rank3_label",
                    "rank3_prob",
                ],
                label="Per-chunk Predictions",
            )
            long_counts_table = gr.Dataframe(
                headers=["Label", "Count", "Fraction"],
                label="Top-label Occurrence Summary",
            )

            analyze_button.click(
                fn=_run_long_inference,
                inputs=[
                    long_audio_input,
                    long_weights_input,
                    chunk_duration_slider,
                    stride_slider,
                ],
                outputs=[long_status_md, long_chunk_table, long_counts_table],
            )

        with gr.Tab("Info"):
            gr.Markdown(
                """
                ## Project Details

                ### Single-Label CNNs (mel-spectrogram input)
                - **12-class Chinese instrument classifier:**  
                `saved_weights/chinese_single_class/train_12_class/best_val_acc.pt`
                - **Western instrument classifier (IRMAS pretrain):**  
                `saved_weights/irmas_pretrain_single_class/train_2/best_val_acc.pt`

                ### Multi-Label CNNs
                - **Western instruments (multi-label):**  
                `saved_weights/multi_label/train_1/best_micro_f1.pt`
                """
            )
    return demo


def main():
    interface = build_interface()
    interface.launch(share=False)


if __name__ == "__main__":
    main()
