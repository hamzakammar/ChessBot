from .utils import chess_manager, GameContext
from chess import Move
import torch
from pathlib import Path
from .models.chess_net import ChessNet
from .utils.search import choose_best_move
from .utils.move_index import move_to_index
import time
from huggingface_hub import hf_hub_download
import os

# Load model once on startup
WEIGHTS_PATH = Path(__file__).parent.parent / "weights" / "best.pt"
if not WEIGHTS_PATH.exists():
    print("[Bot] Downloading weights from Hugging Face...")
    WEIGHTS_PATH.parent.mkdir(exist_ok=True)
    # Get HF token from environment (optional, for private repos)
    hf_token = os.environ.get("HF_TOKEN", None)
    downloaded = hf_hub_download(
        repo_id="HamzaAmmar/chesshacks-model",
        filename="best.pt",
        cache_dir=str(WEIGHTS_PATH.parent),
        token=hf_token
    )
    # Move to expected location
    import shutil
    shutil.copy(downloaded, WEIGHTS_PATH)
    print(f"[Bot] Downloaded weights to {WEIGHTS_PATH}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Bot] Using device: {device}")

model = ChessNet().to(device)
ckpt = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"[Bot] Loaded model from {WEIGHTS_PATH}")


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Computing best move...")
    start_time = time.time()

    # Use trained model to choose best move
    best_move, score = choose_best_move(
        ctx.board,
        model,
        device=device,
        depth=3,        # Search depth
        root_k=20,      # Top 20 moves to consider at root
        child_k=10      # Top 10 moves to consider in tree
    )
    
    elapsed = time.time() - start_time
    print(f"[Bot] Move: {best_move}, Score: {score:.3f}, Time: {elapsed:.2f}s")
    
    # Log move probabilities for analysis
    # For now, just give the best move high probability
    legal_moves = list(ctx.board.generate_legal_moves())
    move_probs = {move: 0.01 for move in legal_moves}
    if best_move:
        move_probs[best_move] = 0.99
    ctx.logProbabilities(move_probs)

    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Clear any caches if needed
    print("[Bot] Game reset")
    pass
