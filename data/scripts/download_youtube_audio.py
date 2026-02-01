#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Optional, Sequence

def _repo_root(script_dir: Path) -> Path:
    return script_dir.parents[1]

def build_output_path(desired_name: str, folder: str, script_dir: Path) -> Path:
    desired = Path(desired_name).expanduser()
    if desired.is_absolute():
        out_path = desired
    else:
        base = Path(folder).expanduser()
        if not base.is_absolute():
            base = _repo_root(script_dir) / base
        out_path = base / desired
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path

def _yt_dlp_output_template(output_path: Path) -> str:
    base = output_path.with_suffix("")
    return f"{base}.%(ext)s"

def _normalize_clients(clients: Optional[str]) -> list[str]:
    if not clients:
        return []
    return [c.strip() for c in clients.split(",") if c.strip()]

def _build_cmd(
    url: str,
    output_path: Path,
    cookies: Optional[Path],
    cookies_from_browser: Optional[str],
    user_agent: Optional[str],
    proxy: Optional[str],
    rate_limit: Optional[str],
    retries: int,
    player_client: Optional[str],
) -> list[str]:
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--no-playlist",
        "-o",
        _yt_dlp_output_template(output_path),
    ]

    if player_client:
        cmd.extend(["--extractor-args", f"youtube:player_client={player_client}"])
    if cookies:
        cmd.extend(["--cookies", str(cookies)])
    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])
    if user_agent:
        cmd.extend(["--user-agent", user_agent])
    if proxy:
        cmd.extend(["--proxy", proxy])
    if rate_limit:
        cmd.extend(["--rate-limit", rate_limit])
    if retries is not None:
        cmd.extend(["--retries", str(retries)])

    cmd.append(url)
    return cmd

def _find_output(output_path: Path) -> Optional[Path]:
    if output_path.exists():
        return output_path
    candidates = sorted(output_path.parent.glob(output_path.with_suffix("").name + ".*"))
    return candidates[0] if candidates else None

def download_audio_once(
    url: str,
    output_path: Path,
    cookies: Optional[Path],
    cookies_from_browser: Optional[str],
    user_agent: Optional[str],
    proxy: Optional[str],
    rate_limit: Optional[str],
    retries: int,
    player_clients: Sequence[str],
) -> None:
    if output_path.exists():
        print(f"✅ Already exists: {output_path}")
        return

    last_error: Optional[Exception] = None
    attempts = list(player_clients) if player_clients else [None]

    for client in attempts:
        cmd = _build_cmd(
            url=url,
            output_path=output_path,
            cookies=cookies,
            cookies_from_browser=cookies_from_browser,
            user_agent=user_agent,
            proxy=proxy,
            rate_limit=rate_limit,
            retries=retries,
            player_client=client,
        )

        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as exc:
            raise SystemExit("yt-dlp not found on PATH. Install it first: pip install yt-dlp") from exc
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"Download attempt failed (player_client={client or 'default'}).")
            continue

        output = _find_output(output_path)
        if output is None:
            last_error = RuntimeError("Download finished but output not found.")
            continue
        if output != output_path:
            output.rename(output_path)
        print(f"✅ Saved: {output_path}")
        return

    raise SystemExit(f"yt-dlp failed after retries. Last error: {last_error}")

def run_embedded_batch(
    entries: list[dict[str, str]],
    args: argparse.Namespace,
    script_dir: Path,
):
    for i, item in enumerate(entries, start=1):
        url = item.get("url")
        out_name = item.get("output")

        if not url or not out_name:
            print(f"Skipping entry {i}: missing url/output")
            continue

        output_path = build_output_path(
            desired_name=out_name,
            folder=args.folder,
            script_dir=script_dir,
        )

        try:
            download_audio_once(
                url=url,
                output_path=output_path,
                cookies=args.cookies,
                cookies_from_browser=args.cookies_from_browser,
                user_agent=args.user_agent,
                proxy=args.proxy,
                rate_limit=args.rate_limit,
                retries=args.retries,
                player_clients=_normalize_clients(args.player_clients),
            )
        except SystemExit as e:
            print(f"❌ Failed item {i}: {url}\n   Reason: {e}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube audio as WAV (auto batch mode)."
    )

    PATH = "orchestral_instruments"
    # PATH = "chinese_instruments"

    DEFAULT_BATCH = [
        {
            "url": "https://www.youtube.com/watch?v=WAoLJ8GbA4Y",
            "output": "adagio_for_strings.wav",
        },
        {
            "url": "https://www.youtube.com/watch?v=MP2_6OLummA",
            "output": "bbc_national_orchestra_of_wales_strings.wav",
        },
        {
            "url": "https://www.youtube.com/watch?v=_2Y1hCgDvNE",
            "output": "shostakovich_waltz_no_2_carion_wind_quintet_woodwind_and_horn.wav",
        },
    ]

    # Optional overrides (still useful)
    parser.add_argument("--folder", default=f"data/raw_sources/{PATH}")
    parser.add_argument("--cookies", type=Path, default=None)
    parser.add_argument("--cookies-from-browser", default=None)
    parser.add_argument("--user-agent", default=None)
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--rate-limit", default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--player-clients",
        default="web,android,ios",
        help="Comma-separated yt-dlp youtube player clients to try.",
    )

    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent

    print("Running embedded YouTube batch downloader\n")
    print(f"Output folder: {args.folder}")
    print(f"Items to download: {len(DEFAULT_BATCH)}\n")

    run_embedded_batch(DEFAULT_BATCH, args, script_dir)


if __name__ == "__main__":
    main()
