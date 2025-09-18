import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import numpy as np
import faiss
import json
import os


TRANSFORM_IMAGE = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_image(img_path: str) -> torch.Tensor:
    input_img = Image.open(img_path).convert("RGB")
    transformed_img = TRANSFORM_IMAGE(input_img).unsqueeze(0)
    return transformed_img

def get_embedding(img_path: str, model, device) -> np.ndarray:
    image_tensor = load_image(img_path).to(device)
    with torch.no_grad():
        embedding = model(image_tensor)
    return embedding.cpu().numpy()


def create_indexes(files: list, model, device, index_dim=384):
    print("Creating Faiss indexes (L2 and Cosine)...")
    index_l2 = faiss.IndexFlatL2(index_dim)
    index_cosine = faiss.IndexFlatIP(index_dim)
    
    all_embeddings_map = {}

    model.eval()
    for file_path in tqdm(files):
        embedding = get_embedding(file_path, model, device)
        
        index_l2.add(embedding)

        normalized_embedding = embedding.copy()
        faiss.normalize_L2(normalized_embedding)
        index_cosine.add(normalized_embedding)
        
        all_embeddings_map[file_path] = embedding.tolist()

    with open("all_embeddings.json", "w") as f:
        json.dump(all_embeddings_map, f)
    faiss.write_index(index_l2, "data_l2.bin")
    faiss.write_index(index_cosine, "data_cosine.bin")
    
    return index_l2, index_cosine

def search_index(index, query_embedding: np.ndarray, k=3):
    distances, indices = index.search(query_embedding, k)
    return indices[0]


def main():
    cwd = os.getcwd()
    ROOT_DIR = os.path.join(cwd, "COCO-128-2/train/")
    files = [os.path.join(ROOT_DIR, f) for f in os.listdir(ROOT_DIR) if f.lower().endswith(".jpg")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.to(device)
    model.eval()

    if os.path.exists("data_l2.bin") and os.path.exists("data_cosine.bin"):
        print("Loading existing indexes from files...")
        index_l2 = faiss.read_index("data_l2.bin")
        index_cosine = faiss.read_index("data_cosine.bin")
    else:
        index_l2, index_cosine = create_indexes(files, model, device)

    input_file = "/workspace/COCO-128-2/valid/000000000081_jpg.rf.5262c2db56ea4568d7d32def1bde3d06.jpg"
    print(f"\nSearching for images similar to: {os.path.basename(input_file)}")


    query_embedding = get_embedding(input_file, model, device)

    l2_results = search_index(index_l2, query_embedding, k=3)

    query_embedding_normalized = query_embedding.copy()
    faiss.normalize_L2(query_embedding_normalized)
    cosine_results = search_index(index_cosine, query_embedding_normalized, k=3)


    print("\n--- L2 Distance Results ---")
    for i, index in enumerate(l2_results):
        print(f"Top {i+1}: {files[index]}")

    print("\n--- Cosine Similarity Results ---")
    for i, index in enumerate(cosine_results):
        print(f"Top {i+1}: {files[index]}")

if __name__ == "__main__":
    main()