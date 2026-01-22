#!/usr/bin/env python3
"""
Zaawansowany system tworzenia datasetu z nagrań wideo grzybów.
Automatyczna ekstrakcja + augmentacja + filtracja jakości.

Autor: Kiril Horobets
Politechnika Warszawska, 2026
"""

import cv2
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime


class DatasetCreator:
    def __init__(self, input_dir, output_dir, config):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = config
        
        # Tworzenie struktury katalogów
        self.dirs = {
            'original': self.output_dir / 'original',
            'augmented': self.output_dir / 'augmented',
            'rejected': self.output_dir / 'rejected'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'videos_processed': 0,
            'frames_extracted': 0,
            'frames_augmented': 0,
            'frames_rejected': 0,
            'total_frames': 0,
            'processing_time': 0,
            'videos': []
        }
    
    def detect_blur(self, image, threshold=100):
        """
        Wykrywa rozmyte obrazy używając Laplacian variance.
        Niższa wartość = bardziej rozmyte.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var, laplacian_var >= threshold
    
    def adjust_brightness(self, image, factor):
        """Zmienia jasność obrazu"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def adjust_contrast(self, image, factor):
        """Zmienia kontrast obrazu"""
        return np.clip(127 + factor * (image.astype(np.float32) - 127), 0, 255).astype(np.uint8)
    
    def augment_frame(self, image, frame_id, video_name):
        """
        Tworzy augmentowane wersje klatki:
        - Jasność (x3: ciemniejsza, normalna, jaśniejsza)
        - Kontrast (x2: niższy, wyższy)
        - Flip horizontal
        - Rotacje (niewielkie)
        """
        augmented_frames = []
        
        aug_config = self.config['augmentation']
        
        # 1. Oryginał
        if aug_config['include_original']:
            augmented_frames.append(('original', image))
        
        # 2. Brightness variations
        if aug_config['brightness']['enabled']:
            for i, factor in enumerate(aug_config['brightness']['factors']):
                bright_img = self.adjust_brightness(image, factor)
                augmented_frames.append((f'bright_{i}', bright_img))
        
        # 3. Contrast variations
        if aug_config['contrast']['enabled']:
            for i, factor in enumerate(aug_config['contrast']['factors']):
                contrast_img = self.adjust_contrast(image, factor)
                augmented_frames.append((f'contrast_{i}', contrast_img))
        
        # 4. Horizontal flip
        if aug_config['flip_horizontal']:
            flipped = cv2.flip(image, 1)
            augmented_frames.append(('flip_h', flipped))
        
        # 5. Small rotations
        if aug_config['rotation']['enabled']:
            for i, angle in enumerate(aug_config['rotation']['angles']):
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h), 
                                        borderMode=cv2.BORDER_REFLECT)
                augmented_frames.append((f'rot_{i}', rotated))
        
        # Zapisz wszystkie augmentowane klatki
        saved = []
        for aug_type, aug_img in augmented_frames:
            filename = f"{video_name}_frame_{frame_id:06d}_{aug_type}.jpg"
            output_path = self.dirs['augmented'] / filename
            cv2.imwrite(str(output_path), aug_img, 
                       [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved.append(filename)
            self.stats['frames_augmented'] += 1
        
        return saved
    
    def process_video(self, video_path):
        """Przetwarza pojedyncze wideo"""
        video_name = video_path.stem
        print(f"\n{'='*60}")
        print(f" Przetwarzanie: {video_name}")
        print(f"{'='*60}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f" Nie można otworzyć: {video_path}")
            return
        
        # Informacje o wideo
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f" Parametry:")
        print(f"   - Rozdzielczość: {width}x{height}")
        print(f"   - FPS: {fps:.2f}")
        print(f"   - Klatki: {total_frames}")
        print(f"   - Czas trwania: {duration:.1f}s")
        
        video_stats = {
            'name': video_name,
            'path': str(video_path),
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration,
            'extracted': 0,
            'rejected': 0,
            'augmented': 0
        }
        
        frame_interval = self.config['frame_interval']
        blur_threshold = self.config['blur_threshold']
        resize_target = self.config.get('resize_to', None)
        
        expected_frames = total_frames // frame_interval
        print(f"   - Co {frame_interval}-ta klatka → ~{expected_frames} klatek\n")
        
        frame_count = 0
        saved_count = 0
        rejected_count = 0
        
        pbar = tqdm(total=expected_frames, desc="Ekstrakcja", unit="frame")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Zapisz co N-tą klatkę
                if frame_count % frame_interval == 0:
                    
                    # Opcjonalne resize
                    if resize_target:
                        frame = cv2.resize(frame, resize_target)
                    
                    # Sprawdź jakość (blur detection)
                    blur_score, is_sharp = self.detect_blur(frame, blur_threshold)
                    
                    if is_sharp:
                        # Zapisz oryginalną klatkę
                        filename = f"{video_name}_frame_{saved_count:06d}.jpg"
                        original_path = self.dirs['original'] / filename
                        cv2.imwrite(str(original_path), frame, 
                                  [cv2.IMWRITE_JPEG_QUALITY, 95])
                        
                        video_stats['extracted'] += 1
                        self.stats['frames_extracted'] += 1
                        
                        # Augmentacja
                        if self.config['augmentation']['enabled']:
                            aug_files = self.augment_frame(frame, saved_count, video_name)
                            video_stats['augmented'] += len(aug_files)
                        
                        saved_count += 1
                    else:
                        # Zapisz odrzuconą klatkę (opcjonalnie)
                        if self.config.get('save_rejected', False):
                            filename = f"{video_name}_rejected_{rejected_count:06d}_blur_{blur_score:.1f}.jpg"
                            rejected_path = self.dirs['rejected'] / filename
                            cv2.imwrite(str(rejected_path), frame, 
                                      [cv2.IMWRITE_JPEG_QUALITY, 70])
                        
                        rejected_count += 1
                        video_stats['rejected'] += 1
                        self.stats['frames_rejected'] += 1
                    
                    pbar.update(1)
                
                frame_count += 1
        
        finally:
            cap.release()
            pbar.close()
        
        print(f"\n✅ Zakończono przetwarzanie {video_name}:")
        print(f"   - Zapisane klatki: {saved_count}")
        print(f"   - Augmentowane: {video_stats['augmented']}")
        print(f"   - Odrzucone (blur): {rejected_count}")
        
        self.stats['videos_processed'] += 1
        self.stats['videos'].append(video_stats)
        
        return video_stats
    
    def process_all_videos(self):
        """Przetwarza wszystkie wideo z katalogu"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
        
        video_files = [f for f in self.input_dir.iterdir() 
                      if f.suffix in video_extensions]
        
        if not video_files:
            print(f"❌ Nie znaleziono plików wideo w: {self.input_dir}")
            return
        
        print(f"\n🎬 Znaleziono {len(video_files)} plików wideo")
        print(f" Wejście: {self.input_dir}")
        print(f" Wyjście: {self.output_dir}\n")
        
        start_time = datetime.now()
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] {video_file.name}")
            self.process_video(video_file)
        
        end_time = datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()
        self.stats['total_frames'] = (self.stats['frames_extracted'] + 
                                      self.stats['frames_augmented'])
        
        self.save_statistics()
        self.print_summary()
    
    def save_statistics(self):
        """Zapisuje statystyki do JSON"""
        stats_path = self.output_dir / 'statistics.json'
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n Statystyki zapisane: {stats_path}")
    
    def print_summary(self):
        """Wyświetla podsumowanie"""
        print(f"\n{'='*60}")
        print(f" PODSUMOWANIE")
        print(f"{'='*60}")
        print(f"✅ Przetworzonych wideo: {self.stats['videos_processed']}")
        print(f"✅ Wyekstrahowanych klatek: {self.stats['frames_extracted']}")
        print(f"✅ Augmentowanych klatek: {self.stats['frames_augmented']}")
        print(f" RAZEM: {self.stats['total_frames']} obrazów")
        print(f"❌ Odrzuconych (blur): {self.stats['frames_rejected']}")
        print(f"  Czas: {self.stats['processing_time']:.1f}s")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Zaawansowany system tworzenia datasetu z wideo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:

1. Podstawowe (co 30 klatka, augmentacja):
   python advanced_dataset_creator.py --input data/raw --output data/frames

2. Gęstsza ekstrakcja (co 15 klatka):
   python advanced_dataset_creator.py --input data/raw --output data/frames --interval 15

3. Bez augmentacji (tylko oryginały):
   python advanced_dataset_creator.py --input data/raw --output data/frames --no-augment

4. Resize do 640x640:
   python advanced_dataset_creator.py --input data/raw --output data/frames --resize 640 640

5. Zapisz odrzucone klatki:
   python advanced_dataset_creator.py --input data/raw --output data/frames --save-rejected
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Katalog z plikami wideo')
    
    parser.add_argument('--output', '-o', required=True,
                       help='Katalog wyjściowy dla klatek')
    
    parser.add_argument('--interval', type=int, default=30,
                       help='Ekstrakcja co N-tą klatkę (domyślnie: 30)')
    
    parser.add_argument('--blur-threshold', type=float, default=100,
                       help='Próg wykrywania blur (domyślnie: 100)')
    
    parser.add_argument('--resize', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                       help='Resize klatek do podanego rozmiaru')
    
    parser.add_argument('--no-augment', action='store_true',
                       help='Wyłącz augmentację')
    
    parser.add_argument('--save-rejected', action='store_true',
                       help='Zapisz odrzucone (rozmyte) klatki')
    
    args = parser.parse_args()
    
    # Konfiguracja
    config = {
        'frame_interval': args.interval,
        'blur_threshold': args.blur_threshold,
        'resize_to': tuple(args.resize) if args.resize else None,
        'save_rejected': args.save_rejected,
        'augmentation': {
            'enabled': not args.no_augment,
            'include_original': False,  # Oryginał już zapisany w /original
            'brightness': {
                'enabled': True,
                'factors': [0.7, 1.3]  # Ciemniejszy, Jaśniejszy
            },
            'contrast': {
                'enabled': True,
                'factors': [0.8, 1.2]  # Niższy, Wyższy
            },
            'flip_horizontal': True,
            'rotation': {
                'enabled': True,
                'angles': [-5, 5]  # Niewielkie rotacje
            }
        }
    }
    
    # Uruchom
    creator = DatasetCreator(args.input, args.output, config)
    creator.process_all_videos()


if __name__ == '__main__':
    main()
