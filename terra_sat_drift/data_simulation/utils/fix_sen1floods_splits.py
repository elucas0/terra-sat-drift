from pathlib import Path

# Chemin vers vos dossiers de splits
split_dir = Path("./datasets/sen1floods11/v1.1/splits/flood_handlabeled")

for split_file in split_dir.glob("*.txt"):
    with open(split_file, 'r') as f:
        lines = f.readlines()
    
    # On extrait uniquement la racine avant le premier underscore significatif
    # "Ghana_103272_S2Hand.tif" -> "Ghana_103272"
    clean_ids = []
    for line in lines:
        if not line.strip(): continue
        # On prend la première partie (image), on enlève l'extension et le suffixe
        image_part = line.split(',')[0]
        root_id = image_part.replace("_S2Hand", "").replace("_S1Hand", "").strip()
        clean_ids.append(root_id)
    
    with open(split_file, 'w') as f:
        f.write("\n".join(clean_ids))

print("Nettoyage terminé : IDs racines uniquement.")