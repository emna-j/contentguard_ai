
import numpy as np
import cv2
from PIL import Image
from multiprocessing import Pool, cpu_count
from functools import partial
import time


class DetailedMetrics:

    @staticmethod
    def calculate_blur_score(img_array):
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def calculate_color_metrics(img_array):
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        return {
            "saturation": float(np.mean(hsv[:, :, 1])),
            "value": float(np.mean(hsv[:, :, 2])),
            "hue_variance": float(np.var(hsv[:, :, 0]))
        }

    @staticmethod
    def calculate_texture_complexity(img_array):
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # Gradient magnitude
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        return float(np.mean(magnitude))

    @staticmethod
    def calculate_noise_level(img_array):
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # Utiliser la variance locale comme proxy du bruit
        kernel_size = 3
        local_var = cv2.blur(gray.astype(float) ** 2, (kernel_size, kernel_size)) - \
                    cv2.blur(gray.astype(float), (kernel_size, kernel_size)) ** 2
        return float(np.mean(local_var))

    @classmethod
    def calculate_detailed_analysis(cls, prob_real, prob_fake, image_path):

        try:
            start_time = time.time()

            img = np.array(Image.open(image_path).convert("RGB"))

            tasks = [
                cls.calculate_blur_score,
                cls.calculate_texture_complexity,
                cls.calculate_noise_level
            ]

            with Pool(processes=min(3, cpu_count())) as pool:
                results = pool.map(lambda func: func(img), tasks)

            blur_score = results[0]
            texture_complexity = results[1]
            noise_level = results[2]

            color_metrics = cls.calculate_color_metrics(img)

            uncertainty = abs(prob_real - prob_fake)
            confidence_level = "HIGH" if uncertainty > 0.3 else "MEDIUM" if uncertainty > 0.15 else "LOW"
            risk_level = "HIGH" if prob_fake > 0.7 else "MEDIUM" if prob_fake > 0.5 else "LOW"

            elapsed = time.time() - start_time
            print(f"[METRICS]  Calculées en {elapsed:.3f}s (multiprocessing)")

            return {
                "blur_score": round(float(blur_score), 2),
                "texture_complexity": round(float(texture_complexity), 2),
                "noise_level": round(float(noise_level), 2),
                "saturation": round(color_metrics["saturation"], 2),
                "value": round(color_metrics["value"], 2),
                "hue_variance": round(color_metrics["hue_variance"], 2),
                "uncertainty": round(uncertainty * 100, 2),
                "confidence_level": confidence_level,
                "risk_level": risk_level,
                "processing_time_ms": round(elapsed * 1000, 2)
            }

        except Exception as e:
            print(f"[METRICS]  Erreur: {e}")
            return {
                "blur_score": 0,
                "texture_complexity": 0,
                "noise_level": 0,
                "saturation": 0,
                "value": 0,
                "hue_variance": 0,
                "uncertainty": 0,
                "confidence_level": "UNKNOWN",
                "risk_level": "UNKNOWN",
                "processing_time_ms": 0
            }

    @classmethod
    def calculate_batch_metrics(cls, image_paths):
        """
        Calcule les métriques pour plusieurs images en parallèle

        Args:
            image_paths: Liste de chemins d'images

        Returns:
            list: Liste de dictionnaires de métriques
        """
        print(f"[METRICS-BATCH] Calcul pour {len(image_paths)} images...")

        with Pool(processes=cpu_count()) as pool:
            func = partial(cls.calculate_detailed_analysis, 0.5, 0.5)
            results = pool.map(func, image_paths)

        return results


if __name__ == "__main__":
    print(f"Module metrics avec multiprocessing ({cpu_count()} CPUs)")