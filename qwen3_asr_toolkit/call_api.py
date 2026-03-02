import argparse
import os
import srt
import requests
import concurrent.futures

from tqdm import tqdm
from datetime import timedelta
from collections import Counter
from urllib.parse import urlparse
from silero_vad import load_silero_vad
from qwen3_asr_toolkit.qwen3asr import QwenASR
from qwen3_asr_toolkit.audio_tools import load_audio, process_vad, save_audio_file, WAV_SAMPLE_RATE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Python toolkit for the Qwen3-ASR API—parallel high‑throughput calls, robust long‑audio transcription, multi‑sample‑rate support."
    )
    parser.add_argument("--input-file", '-i', type=str, required=True, help="Input media file path")
    parser.add_argument("--context", '-c', type=str, default="", help="Optional text context for ASR")
    parser.add_argument("--provider", choices=["local", "dashscope"], default="local", help="ASR backend to use")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/v1/chat/completions", help="Local ASR API endpoint")
    parser.add_argument("--model", type=str, default=None, help="Model name for the selected API")
    parser.add_argument("--api-timeout", type=int, default=300, help="API request timeout (seconds)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for the local API")
    parser.add_argument("--dashscope-api-key", '-key', type=str, help="DashScope API key")
    parser.add_argument("--skip-failed", action="store_true", help="Skip failed segments instead of aborting")
    parser.add_argument("--max-retries", type=int, default=10, help="Max retries per segment")
    parser.add_argument("--num-threads", '-j', type=int, default=4, help="Number of threads to use for parallel calls")
    parser.add_argument("--vad-segment-threshold", '-d', type=int, default=120, help="Segment threshold seconds for VAD")
    parser.add_argument("--max-segment-seconds", type=int, default=180, help="Hard limit for segment length (seconds)")
    parser.add_argument("--tmp-dir", '-t', type=str, default=os.path.join(os.path.expanduser("~"), "qwen3-asr-cache"), help="Temp directory path")
    parser.add_argument("--save-srt", '-srt', action="store_true", help="Save SRT subtitle file")
    parser.add_argument("--silence", '-s', action="store_true", help="Reduce the output info on the terminal")
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input_file
    context = args.context
    provider = args.provider
    api_url = args.api_url
    model = args.model
    api_timeout = args.api_timeout
    temperature = args.temperature
    skip_failed = args.skip_failed
    dashscope_api_key = args.dashscope_api_key
    max_retries = args.max_retries
    num_threads = args.num_threads
    vad_segment_threshold = args.vad_segment_threshold
    max_segment_seconds = args.max_segment_seconds
    tmp_dir = args.tmp_dir
    save_srt = args.save_srt
    silence = args.silence

    # check if input file exists
    if input_file.startswith(("http://", "https://")):
        try:
            response = requests.head(input_file, allow_redirects=True, timeout=5)
            if response.status_code >= 400:
                raise FileNotFoundError(f"returned status code {response.status_code}")
        except Exception as e:
            raise FileNotFoundError(f"HTTP link {input_file} does not exist or is inaccessible: {str(e)}")
    elif not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file \"{input_file}\" does not exist!")

    if provider == "dashscope":
        if dashscope_api_key:
            os.environ["DASHSCOPE_API_KEY"] = dashscope_api_key
        elif "DASHSCOPE_API_KEY" not in os.environ:
            raise ValueError("Please set DASHSCOPE_API_KEY or pass --dashscope-api-key.")

    qwen3asr = QwenASR(
        provider=provider,
        api_url=api_url,
        model=model,
        timeout_s=api_timeout,
        temperature=temperature,
        max_retries=max_retries,
    )

    wav = load_audio(input_file)
    if not silence:
        print(f"Loaded wav duration: {len(wav) / WAV_SAMPLE_RATE:.2f}s")

    # Segment wav exceeding 3 minutes
    if len(wav) / WAV_SAMPLE_RATE >= 180:
        if not silence:
            print(f"Wav duration is longer than 3 min, initializing Silero VAD model for segmenting...")
        worker_vad_model = load_silero_vad(onnx=True)
        wav_list = process_vad(
            wav,
            worker_vad_model,
            segment_threshold_s=vad_segment_threshold,
            max_segment_threshold_s=max_segment_seconds,
        )
        if not silence:
            print(f"Segmenting done, total segments: {len(wav_list)}")
    else:
        wav_list = [(0, len(wav), wav)]

    # Save processed audio to tmp dir
    wav_name = os.path.basename(input_file)
    wav_dir_name = os.path.splitext(wav_name)[0]
    save_dir = os.path.join(tmp_dir, wav_dir_name)

    wav_path_list = []
    for idx, (_, _, wav_data) in enumerate(wav_list):
        wav_path = os.path.join(save_dir, f"{wav_name}_{idx}.wav")
        save_audio_file(wav_data, wav_path)
        wav_path_list.append(wav_path)

    # Multithread call qwen3-asr-flash api
    results = []
    languages = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_dict = {
            executor.submit(qwen3asr.asr, wav_path, context): idx
            for idx, wav_path in enumerate(wav_path_list)
        }
        if not silence:
            pbar = tqdm(total=len(future_dict), desc="Calling local ASR API")
        for future in concurrent.futures.as_completed(future_dict):
            idx = future_dict[future]
            try:
                language, recog_text = future.result()
                results.append((idx, recog_text))
                languages.append(language)
            except Exception as e:
                if skip_failed:
                    results.append((idx, ""))
                    languages.append("Unknown")
                    if not silence:
                        error_name = e.__class__.__name__
                        print(f"Segment {idx} failed ({error_name}). Skipped.")
                else:
                    raise
            if not silence:
                pbar.update(1)
        if not silence:
            pbar.close()

    # Sort and splice in the original order
    results.sort(key=lambda x: x[0])
    full_text = " ".join(text for _, text in results)
    language = Counter(languages).most_common(1)[0][0] if languages else "Unknown"

    if not silence:
        print(f"Detected Language: {language}")
        print(f"Full Transcription: {full_text}")

    # Delete tmp save dir
    os.system(f"rm -rf {save_dir}")

    # Save full text to local file
    if os.path.exists(input_file):
        save_file = os.path.splitext(input_file)[0] + ".txt"
    else:
        save_file = os.path.splitext(urlparse(input_file).path)[0].split('/')[-1] + '.txt'

    with open(save_file, 'w') as f:
        f.write(language + '\n')
        f.write(full_text + '\n')

    print(f"Full transcription of \"{input_file}\" from local ASR API saved to \"{save_file}\"!")

    # Save subtitles to local SRT file
    if args.save_srt:
        subtitles = []
        for idx, result in enumerate(results):
            start_time = wav_list[idx][0] / WAV_SAMPLE_RATE
            end_time = wav_list[idx][1] / WAV_SAMPLE_RATE
            content = result[1]
            subtitles.append(srt.Subtitle(
                index=idx,
                start=timedelta(seconds=start_time),
                end=timedelta(seconds=end_time),
                content=content
            ))
        final_srt_content = srt.compose(subtitles)
        srt_path = os.path.splitext(save_file)[0] + ".srt"
        with open(srt_path, 'w') as f:
            f.write(final_srt_content)
        print(f"SRT subtitles of \"{input_file}\" from local ASR API saved to \"{srt_path}\"!")


if __name__ == '__main__':
    main()
