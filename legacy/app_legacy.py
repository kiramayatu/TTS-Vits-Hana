import os
import numpy as np
import torch
from torch import no_grad, LongTensor
import argparse
import commons
from mel_processing import spectrogram_torch
import utils
from models import SynthesizerTrn
import gradio as gr
import librosa
import webbrowser

from text import text_to_sequence

device = "cuda:0" if torch.cuda.is_available() else "cpu"

language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}

lang = ['日本語', '简体中文', 'English', 'Mix']


def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(
        text,
        hps.symbols,
        [] if is_symbol else hps.data.text_cleaners
    )
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    return LongTensor(text_norm)


def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(text, speaker, language, speed):
        if language is not None:
            text = language_marks[language] + text + language_marks[language]

        speaker_id = speaker_ids[speaker]
        stn_tst = get_text(text, hps, False)

        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)

            audio = model.infer(
                x_tst,
                x_tst_lengths,
                sid=sid,
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1.0 / speed
            )[0][0, 0].cpu().float().numpy()

        return "Success", (hps.data.sampling_rate, audio)

    return tts_fn


def create_vc_fn(model, hps, speaker_ids):
    def vc_fn(original_speaker, target_speaker, audio_input):
        if audio_input is None:
            return "You need to record or upload an audio", None

        sampling_rate, audio = audio_input

        original_speaker_id = speaker_ids[original_speaker]
        target_speaker_id = speaker_ids[target_speaker]

        # normalize audio
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / np.max(np.abs(audio))

        if audio.ndim > 1:
            audio = librosa.to_mono(audio.T)

        if sampling_rate != hps.data.sampling_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sampling_rate,
                target_sr=hps.data.sampling_rate
            )

        with no_grad():
            y = torch.FloatTensor(audio).to(device)
            y = y.unsqueeze(0)

            spec = spectrogram_torch(
                y,
                hps.data.filter_length,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                center=False
            )

            spec_lengths = LongTensor([spec.size(-1)]).to(device)
            sid_src = LongTensor([original_speaker_id]).to(device)
            sid_tgt = LongTensor([target_speaker_id]).to(device)

            audio = model.voice_conversion(
                spec,
                spec_lengths,
                sid_src=sid_src,
                sid_tgt=sid_tgt
            )[0][0, 0].cpu().float().numpy()

        return "Success", (hps.data.sampling_rate, audio)

    return vc_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="inference/G_latest.pth",
        help="path to your fine-tuned model"
    )
    parser.add_argument(
        "--config_dir",
        default="inference/fine.json",
        help="path to your model config file"
    )
    parser.add_argument("--share", action="store_true")

    args = parser.parse_args()

    hps = utils.get_hparams_from_file(args.config_dir)

    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    ).to(device)

    net_g.eval()
    utils.load_checkpoint(args.model_dir, net_g, None)

    speaker_ids = hps.speakers
    speakers = list(speaker_ids.keys())

    tts_fn = create_tts_fn(net_g, hps, speaker_ids)
    vc_fn = create_vc_fn(net_g, hps, speaker_ids)

    with gr.Blocks() as app:
        with gr.Tab("Text-to-Speech"):
            with gr.Row():
                with gr.Column():
                    textbox = gr.TextArea(
                        label="Text",
                        value="こんにちわ。"
                    )
                    char_dropdown = gr.Dropdown(
                        choices=speakers,
                        value=speakers[0],
                        label="Character"
                    )
                    language_dropdown = gr.Dropdown(
                        choices=lang,
                        value=lang[0],
                        label="Language"
                    )
                    speed_slider = gr.Slider(
                        minimum=0.1,
                        maximum=5.0,
                        value=1.0,
                        step=0.1,
                        label="Speed"
                    )

                with gr.Column():
                    text_output = gr.Textbox(label="Message")
                    audio_output = gr.Audio(label="Output Audio")

                    btn = gr.Button("Generate")
                    btn.click(
                    tts_fn,
                    inputs=[textbox, char_dropdown, language_dropdown, speed_slider],
                    outputs=[text_output, audio_output],
                    api_name="tts"   # 👈 ADD THIS
                )


        with gr.Tab("Voice Conversion"):
            audio_input = gr.Audio(
                label="Record or Upload Audio",
                sources=["microphone", "upload"],
                type="numpy"
            )

            source_speaker = gr.Dropdown(
                choices=speakers,
                value=speakers[0],
                label="Source Speaker"
            )
            target_speaker = gr.Dropdown(
                choices=speakers,
                value=speakers[0],
                label="Target Speaker"
            )

            message_box = gr.Textbox(label="Message")
            converted_audio = gr.Audio(label="Converted Audio")

            btn = gr.Button("Convert")
            btn.click(
    vc_fn,
    inputs=[source_speaker, target_speaker, audio_input],
    outputs=[message_box, converted_audio],
    api_name="vc"    # 👈 ADD THIS
)


    webbrowser.open("http://127.0.0.1:7860")

    app.launch(
        share=args.share,
        server_name="127.0.0.1",
        server_port=7860,
)




