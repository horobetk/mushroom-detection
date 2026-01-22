#!/usr/bin/env python3
"""
Skrypt do ekstrakcji klatek z nagrań wideo grzybów.

Autor: Kiril Horobets
Politechnika Warszawska, 2026
Praca inżynierska: System rozpoznawania grzybów na urządzeniach mobilnych
"""

import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_frames(video_path, output_dir, frame_rate=30, max_frames=None, resize=None):
    """
    Ekstrahuje klatki z nagrania wideo.
    
    Args:
        video_path (str): Ścieżka do pliku wideo
        output_dir (str): Katalog wyjściowy dla klatek
        frame_rate (int): Ekstrahuj co N-tą klatkę (domyślnie: 30)
        max_frames (int): Maksymalna liczba klatek do wyekstrahowania
        resize (tuple): Opcjonalnie zmień rozmiar (width, height)
    
    Returns:
        int: Liczba wyekstrahowanych klatek
    """
    
    # Utworzenie katalogu wyjściowego
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Otwarcie pliku wideo
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Nie można otworzyć pliku wideo: {video_path}")
    
    # Pobranie informacji o wideo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f" Informacje o wideo:")
    print(f"   - Rozdzielczość: {width}x{height}")
    print(f"   - FPS: {fps}")
    print(f"   - Liczba klatek: {total_frames}")
    print(f"   - Co {frame_rate}-ta klatka zostanie zapisana")
    
    expected_frames = total_frames // frame_rate
    if max_frames:
        expected_frames = min(expected_frames, max_frames)
    
    print(f"   - Spodziewana liczba zapisanych klatek: {expected_frames}\n")
    
    # Nazwa bazowa pliku (bez rozszerzenia)
    video_basename = Path(video_path).stem
    
    frame_count = 0
    saved_count = 0
    
    # Progress bar
    pbar = tqdm(total=expected_frames, desc="Ekstrakcja klatek", unit="frame")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Zapisz co N-tą klatkę
            if frame_count % frame_rate == 0:
                
                # Opcjonalne przeskalowanie
                if resize:
                    frame = cv2.resize(frame, resize)
                
                # Nazwa pliku wyjściowego
                output_filename = f"{video_basename}_frame_{saved_count:06d}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                
                # Zapisz klatkę
                cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                saved_count += 1
                pbar.update(1)
                
                # Sprawdź limit klatek
                if max_frames and saved_count >= max_frames:
                    break
            
            frame_count += 1
    
    finally:
        cap.release()
        pbar.close()
    
    print(f"\n Zakończono!")
    print(f"   - Wyekstrahowano {saved_count} klatek")
    print(f"   - Zapisano w: {output_dir}")
    
    return saved_count


def process_directory(input_dir, output_dir, frame_rate=30, max_frames_per_video=None, resize=None):
    """
    Przetwarza wszystkie pliki wideo z katalogu.
    
    Args:
        input_dir (str): Katalog z plikami wideo
        output_dir (str): Katalog wyjściowy
        frame_rate (int): Co N-ta klatka
        max_frames_per_video (int): Max klatek z jednego wideo
        resize (tuple): Opcjonalny resize
    """
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']
    
    input_path = Path(input_dir)
    video_files = [f for f in input_path.iterdir() 
                   if f.suffix in video_extensions]
    
    if not video_files:
        print(f" Nie znaleziono plików wideo w katalogu: {input_dir}")
        return
    
    print(f" Znaleziono {len(video_files)} plików wideo\n")
    
    total_frames = 0
    
    for i, video_file in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] Przetwarzanie: {video_file.name}")
        
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
            print(f" Błąd podczas przetwarzania {video_file.name}: {e}")
        
        print("\n" + "="*60 + "\n")
    
    print(f" Wszystkie wideo przetworzone!")
    print(f"   - Całkowita liczba klatek: {total_frames}")


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
    
    # Parsing resize argument
    resize = None
    if args.resize:
        try:
            width, height = map(int, args.resize.split('x'))
            resize = (width, height)
        except:
            print(f"  Nieprawidłowy format resize: {args.resize}. Oczekiwano WIDTHxHEIGHT")
            return
    
    input_path = Path(args.input)
    
    # Sprawdź czy input to plik czy katalog
    if input_path.is_file():
        # Pojedynczy plik wideo
        extract_frames(
            args.input,
            args.output,
            frame_rate=args.frame_rate,
            max_frames=args.max_frames,
            resize=resize
        )
    elif input_path.is_dir():
        # Katalog z plikami wideo
        process_directory(
            args.input,
            args.output,
            frame_rate=args.frame_rate,
            max_frames_per_video=args.max_frames,
            resize=resize
        )
    else:
        print(f" Nie znaleziono pliku ani katalogu: {args.input}")


if __name__ == '__main__':
    main()


"""
PRZYKŁADY UŻYCIA:

1. Ekstrakcja z pojedynczego pliku:
   python extract_frames.py --input video.mp4 --output frames/ --frame-rate 30

2. Ekstrakcja z katalogu:
   python extract_frames.py --input data/raw/ --output data/frames/ --frame-rate 30

3. Z resize do 640x640:
   python extract_frames.py --input video.mp4 --output frames/ --resize 640x640

4. Maksymalnie 100 klatek z każdego wideo:
   python extract_frames.py --input data/raw/ --output data/frames/ --max-frames 100
"""
