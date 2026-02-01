import argparse
import tempfile
from io import BytesIO
from pathlib import Path

from gtts import gTTS
from pydub import AudioSegment

def _load_timecodes(path: Path) -> list[tuple[float, float, str]]:
    entries: list[tuple[float, float, str]] = []
    raw = path.read_text(encoding="utf-8").splitlines()
    for line_no, raw_line in enumerate(raw, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid timecode line {line_no}: {raw_line}")
        start_s, end_s = float(parts[0]), float(parts[1])
        word = parts[2]
        entries.append((start_s, end_s, word))
    if not entries:
        raise ValueError(f"No timecoded entries found in {path}")
    return entries


def _synthesize_word(word: str, lang: str, tld: str) -> AudioSegment:
    buffer = BytesIO()
    gTTS(text=word, lang=lang, tld=tld).write_to_fp(buffer)
    buffer.seek(0)
    return AudioSegment.from_file(buffer, format="mp3")


def _pad_to_duration(audio: AudioSegment, target_ms: int) -> AudioSegment:
    current_ms = len(audio)
    if current_ms >= target_ms:
        return audio
    return audio + AudioSegment.silent(duration=target_ms - current_ms)


def generate_audio(
    text_file: str,
    output_file: str,
    lang: str,
    tld: str,
    timecodes: str | None,
    rate: str | None,
) -> None:
    try:
        if rate:
            print("Warning: gTTS does not support custom speaking rates. Ignoring --rate.")
        if timecodes:
            timecode_path = Path(timecodes)
            entries = _load_timecodes(timecode_path)
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_root = Path(temp_dir)
                combined = AudioSegment.silent(duration=0)
                for idx, (start_s, end_s, word) in enumerate(entries):
                    word_path = temp_root / f"word_{idx:03d}.mp3"
                    word_audio = _synthesize_word(word, lang, tld)
                    word_audio.export(word_path, format="mp3")
                    target_ms = max(0, int(round((end_s - start_s) * 1000)))
                    combined += _pad_to_duration(word_audio, target_ms)

                    if idx < len(entries) - 1:
                        next_start = entries[idx + 1][0]
                        gap_ms = max(0, int(round((next_start - end_s) * 1000)))
                        if gap_ms:
                            combined += AudioSegment.silent(duration=gap_ms)

                combined = combined.set_channels(1).set_frame_rate(16000)
                output_path = Path(output_file)
                output_format = output_path.suffix.lstrip(".") or "webm"
                export_kwargs = {"format": output_format}
                if output_format == "webm":
                    export_kwargs["codec"] = "libopus"
                combined.export(output_file, **export_kwargs)
            print(f"Success! File saved as: {output_file}")
            return

        text_path = Path(text_file)
        text = text_path.read_text(encoding="utf-8")
        if not text.strip():
            print("Error: The input text file is empty.")
            return

        buffer = BytesIO()
        gTTS(text=text, lang=lang, tld=tld).write_to_fp(buffer)
        buffer.seek(0)
        output_path = Path(output_file)
        output_format = output_path.suffix.lstrip(".") or "mp3"
        audio = AudioSegment.from_file(buffer, format="mp3")
        export_kwargs = {"format": output_format}
        if output_format == "webm":
            export_kwargs["codec"] = "libopus"
        audio.export(output_file, **export_kwargs)
        print(f"Success! File saved as: {output_file}")

    except FileNotFoundError as exc:
        missing_path = exc.filename or text_file
        print(f"Error: The file '{missing_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate audio from text using gTTS")
    
    # Required arguments
    parser.add_argument("-i", "--input", help="Path to the input text file (.txt)")
    parser.add_argument("-o", "--output", default="output.webm", help="Output file name (default: output.webm)")
    parser.add_argument("--timecodes", help="Optional timecode file to synthesize word-by-word timing.")
    parser.add_argument("--rate", help="Optional speech rate (not supported by gTTS).")

    # Optional language selection
    parser.add_argument("-l", "--lang", default="ru", help="Language code (e.g., ru, en, fi).")
    parser.add_argument("--tld", default="com", help="Top-level domain for gTTS (e.g., com, co.uk).")

    args = parser.parse_args()

    if not args.timecodes and not args.input:
        parser.error("--input is required unless --timecodes is provided.")

    # Run the asynchronous function
    generate_audio(args.input or "", args.output, args.lang, args.tld, args.timecodes, args.rate)

if __name__ == "__main__":
    main()
