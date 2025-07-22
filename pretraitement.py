import os
import cv2
import numpy as np
from pathlib import Path
import logging

# Configuration
IMAGE_SIZE = (416, 416)  # Taille cible pour le redimensionnement (hauteur, largeur)
DATASET_DIR = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\dataset\images"  # Dossier racine des images
LABELS_DIR = r"C:\Users\amari\OneDrive\Desktop\EdgeAI_Project\dataset\labels"  # Dossier racine des labels
OUTPUT_DIR = r"C:\Users\amari\OneDrive\Desktop\dataset_pre\images"  # Dossier de sortie pour les images prétraitées
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")  # Formats d'image supportés

# Configuration du logging pour le rapport
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def create_output_dirs():
    """Crée les dossiers de sortie pour train et val si nécessaire."""
    for split in ["train", "val"]:
        output_path = Path(OUTPUT_DIR) / split
        output_path.mkdir(parents=True, exist_ok=True)

def process_images(input_dir, label_dir, output_dir, split):
    """
    Traite les images d'un dossier (train ou val).
    Args:
        input_dir: Chemin du dossier d'images
        label_dir: Chemin du dossier de labels
        output_dir: Chemin du dossier de sortie
        split: Nom du split ("train" ou "val")
    Returns:
        processed_count: Nombre d'images traitées
        skipped_count: Nombre d'images ignorées (sans label)
    """
    processed_count = 0
    skipped_count = 0
    input_path = Path(input_dir) / split
    label_path = Path(label_dir) / split
    output_path = Path(output_dir) / split

    # Vérifie si les dossiers existent
    if not input_path.exists():
        logger.error(f"Le dossier {input_path} n'existe pas.")
        return 0, 0
    if not label_path.exists():
        logger.error(f"Le dossier {label_path} n'existe pas.")
        return 0, 0

    # Parcours des images
    for img_file in input_path.glob("*"):
        if img_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Ignoré {img_file.name}: format non supporté.")
            continue

        # Vérifie si le label correspondant existe
        label_file = label_path / f"{img_file.stem}.txt"
        if not label_file.exists():
            logger.warning(f"Ignoré {img_file.name}: fichier label manquant.")
            skipped_count += 1
            continue

        try:
            # Lecture et traitement de l'image
            img = cv2.imread(str(img_file))
            if img is None:
                logger.warning(f"Ignoré {img_file.name}: impossible de lire l'image.")
                skipped_count += 1
                continue

            # Redimensionnement
            img_resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

            # Normalisation (conversion en float et division par 255)
            img_normalized = img_resized.astype(np.float32) / 255.0

            # Conversion retour en uint8 pour sauvegarde
            img_to_save = (img_normalized * 255).astype(np.uint8)

            # Sauvegarde de l'image prétraitée
            output_file = output_path / img_file.name
            cv2.imwrite(str(output_file), img_to_save)
            processed_count += 1
            logger.info(f"Traitée: {img_file.name}")

        except Exception as e:
            logger.error(f"Erreur lors du traitement de {img_file.name}: {str(e)}")
            skipped_count += 1

    return processed_count, skipped_count

def main():
    """Fonction principale pour traiter les datasets train et val."""
    create_output_dirs()

    total_processed = 0
    total_skipped = 0

    # Traitement des images train
    logger.info("Traitement des images de train...")
    processed, skipped = process_images(DATASET_DIR, LABELS_DIR, OUTPUT_DIR, "train")
    total_processed += processed
    total_skipped += skipped
    logger.info(f"Train - Images traitées: {processed}, Ignorées: {skipped}")

    # Traitement des images val
    logger.info("Traitement des images de validation...")
    processed, skipped = process_images(DATASET_DIR, LABELS_DIR, OUTPUT_DIR, "val")
    total_processed += processed
    total_skipped += skipped
    logger.info(f"Validation - Images traitées: {processed}, Ignorées: {skipped}")

    # Rapport final
    logger.info("=== Rapport final ===")
    logger.info(f"Total images traitées: {total_processed}")
    logger.info(f"Total images ignorées: {total_skipped}")

if __name__ == "__main__":
    main()