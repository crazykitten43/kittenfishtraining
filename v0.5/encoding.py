"""KittenFish 18-plane chess position encoding (compatible with train.py)."""
from __future__ import annotations
import numpy as np
import chess

INPUT_PLANES = 18

_PIECE_INDEX = {
    (chess.WHITE, chess.PAWN): 0, (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2, (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4, (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6, (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8, (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10, (chess.BLACK, chess.KING): 11,
}

def encode_fen(fen: str) -> np.ndarray:
    """Return float32 [18, 8, 8] planes, with rank 8 at row zero."""
    board = chess.Board(fen)
    x = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row, col = 7 - chess.square_rank(square), chess.square_file(square)
        x[_PIECE_INDEX[(piece.color, piece.piece_type)], row, col] = 1.0
    x[12, :, :] = float(board.turn == chess.WHITE)
    # Preserve the exact castling-plane convention used by the existing baseline.
    for i, (color, rook_file) in enumerate([
        (chess.WHITE, 7), (chess.WHITE, 0),
        (chess.BLACK, 7), (chess.BLACK, 0),
    ]):
        king_sq = board.king(color)
        if king_sq is not None:
            rook_sq = chess.square(rook_file, chess.square_rank(king_sq))
            if board.castling_rights & chess.BB_SQUARES[rook_sq]:
                x[13 + i, :, :] = 1.0
    if board.ep_square is not None:
        row, col = 7 - chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
        x[17, row, col] = 1.0
    return x
