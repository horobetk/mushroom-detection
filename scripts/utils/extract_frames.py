#!/usr/bin/env python3
# Skrypt do ekstrakcji klatek z nagrań wideo grzybów.
#
# Autor: Kiril Horobets
# Politechnika Warszawska, 2026
# Praca inżynierska: System rozpoznawania grzybów na urządzeniach mobilnych

import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_frames(video_path, output_dir, frame_rate=30, max_frames=None, resize=None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Nie można otworzyć pliku wideo: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Informacje o wideo:")
    print(f"[INFO]   - Rozdzielczość: {width}x{height}")
    print(f"[INFO]   - FPS: {fps}")
    print(f"[INFO]   - Liczba klatek: {total_frames}")
    print(f"[INFO]   - Co {frame_rate}-ta klatka zostanie zapisana")

    expected_frames = total_frames // frame_rate
    if max_frames:
        expected_frames = min(expected_frames, max_frames)

    print(f"[INFO]   - Spodziewana liczba zapisanych klatek: {expected_frames}\n")

    video_basename = Path(video_path).stem

    frame_count = 0
    saved_count = 0

    pbar = tqdm(total=expected_frames, desc="Ekstrakcja klatek", unit="frame")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_count % frame_rate == 0:
                if resize:
                    frame = cv2.resize(frame, resize)

                output_filename = f"{video_basename}_frame_{saved_count:06d}.jpg"
                output_path = os.path.join(output_dir, output_filename)

                cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                saved_count += 1
                pbar.update(1)

                if max_frames and saved_count >= max_frames:
                    break

            frame_count += 1

    finally:
        cap.release()
        pbar.close()

    print(f"\n[OK] Zakończono")
    print(f"[OK]   - Wyekstrahowano {saved_count} klatek")
    print(f"[OK]   - Zapisano w: {output_dir}")

    return saved_count


def process_directory(input_dir, output_dir, frame_rate=30, max_frames_per_video=None, resize=None):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']

    input_path = Path(input_dir)
    video_files = [f for f in input_path.iterdir()
                   if f.suffix in video_extensions]

    if not video_files:
        print(f"[ERR] Nie znaleziono plików wideo w katalogu: {input_dir}")
        return

    print(f"[INFO] Znaleziono {len(video_files)} plików wideo\n")

    total_frames = 0

    for i, video_file in enumerate(video_files, 1):
        print(f"[INFO] [{i}/{len(video_files)}] Przetwarzanie: {video_file.name}")

        video_output_dir = os.path.join(output_dir, video_file.stem)

        try:
            frames_extracted = extract_frames(
                str(video_file),
                video_output_dir,
                frame_rate=frame_rate,
                max_frames=max_frames_per_video,
                resize=resize
            )
            total_frames += frames_extracted

        except Exception as e:
            print(f"[ERR] Błąd podczas przetwarzania {video_file.name}: {e}")

        print("\n" + "="*60 + "\n")

    print(f"[OK] Wszystkie wideo przetworzone")
    print(f"[OK]   - Całkowita liczba klatek: {total_frames}")


def main():
    parser = argparse.ArgumentParser(
        description="Ekstrakcja klatek z nagrań wideo grzybów"
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Ścieżka do pliku wideo lub katalogu z plikami wideo'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Katalog wyjściowy dla klatek'
    )

    parser.add_argument(
        '--frame-rate', '-f',
        type=int,
        default=30,
        help='Ekstrahuj co N-tą klatkę (domyślnie: 30)'
    )

    parser.add_argument(
        '--max-frames', '-m',
        type=int,
        default=None,
        help='Maksymalna liczba klatek do wyekstrahowania z jednego wideo'
    )

    parser.add_argument(
        '--resize',
        type=str,
        default=None,
        help='Zmień rozmiar klatek, format: WIDTHxHEIGHT (np. 640x640)'
    )

    args = parser.parse_args()

    resize = None
    if args.resize:
        try:
            width, height = map(int, args.resize.split('x'))
            resize = (width, height)
        except ValueError:
            print(f"[ERR] Nieprawidłowy format resize: {args.resize}. Oczekiwano WIDTHxHEIGHT")
            return

    input_path = Path(args.input)

    if input_path.is_file():
        extract_frames(
            args.input,
            args.output,
            frame_rate=args.frame_rate,
            max_frames=args.max_frames,
            resize=resize
        )
    elif input_path.is_dir():
        process_directory(
            args.input,
            args.output,
            frame_rate=args.frame_rate,
            max_frames_per_video=args.max_frames,
            resize=resize
        )
    else:
        print(f"[ERR] Nie znaleziono pliku ani katalogu: {args.input}")


if __name__ == '__main__':
    main()


# PRZYKŁADY UŻYCIA:
#
# 1. Ekstrakcja z pojedynczego pliku:
#    python extract_frames.py --input video.mp4 --output frames/ --frame-rate 30
#
# 2. Ekstrakcja z katalogu:
#    python extract_frames.py --input data/raw/ --output data/frames/ --frame-rate 30
#
# 3. Z resize do 640x640:
#    python extract_frames.py --input video.mp4 --output frames/ --resize 640x640
#
# 4. Maksymalnie 100 klatek z każdego wideo:
#    python extract_frames.py --input data/raw/ --output data/frames/ --max-frames 100
