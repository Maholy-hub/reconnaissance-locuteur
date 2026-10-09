"""Reconnaissance du locuteur : moi / personne_choisie / autre.

Usage :
    python predict.py audio.wav
    python predict.py audio.wav --seuil 0.70 --modele model/modele_locuteur.keras
"""
import argparse
import numpy as np
import librosa
import tensorflow as tf

SR, SEG, N_MFCC = 16000, 2.0, 40
CLASSES = ["moi", "personne_choisie", "autre"]


def charger(path):
    """Charge l'audio en 16 kHz mono et supprime les silences."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    intervalles = librosa.effects.split(y, top_db=30)
    if len(intervalles) == 0:
        return y
    return np.concatenate([y[a:b] for a, b in intervalles])


def segments(y, hop_s):
    L, H = int(SEG * SR), int(hop_s * SR)
    return [y[i:i + L] for i in range(0, len(y) - L + 1, H)]


def features(seg):
    """40 MFCC, normalisés par coefficient sur le segment."""
    m = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=N_MFCC)
    m = (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-8)
    return m.astype(np.float32)


def predire(model, path, seuil=0.70, hop_s=1.0):
    y = charger(path)
    segs = segments(y, hop_s)
    if len(segs) == 0:  # audio plus court que 2 s : on complète avec des zéros
        L = int(SEG * SR)
        segs = [np.pad(y, (0, max(0, L - len(y))))[:L]]
    X = np.array([features(s) for s in segs])[..., np.newaxis]
    proba = model.predict(X, verbose=0).mean(axis=0)
    idx = int(proba.argmax())
    confiance = float(proba[idx])
    if idx < 2 and confiance < seuil:   # pas assez sûr : refusé
        idx = 2
    ouvert = idx < 2
    return CLASSES[idx], confiance, proba, ouvert


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Reconnaissance du locuteur")
    p.add_argument("audio", help="fichier audio (wav, flac, mp3...)")
    p.add_argument("--modele", default="model/modele_locuteur.keras")
    p.add_argument("--seuil", type=float, default=0.70)
    a = p.parse_args()

    model = tf.keras.models.load_model(a.modele)
    classe, conf, proba, ouvert = predire(model, a.audio, a.seuil)

    print(f"Locuteur reconnu : {classe} (confiance {conf:.2f})")
    for nom, pr in zip(CLASSES, proba):
        print(f"  {nom:18s} {pr:.3f}")
    print("Résultat : OUVERT" if ouvert else "Résultat : REFUSÉ")
