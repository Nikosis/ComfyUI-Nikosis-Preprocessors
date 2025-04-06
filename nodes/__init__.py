from .depth_anything_v2_nikosis import DepthAnythingV2Nikosis
from .lineart_nikosis import LineArtPreprocessorNikosis
from .lineart_sketch_nikosis import LineArtSketchPreprocessorNikosis
from .edge_nikosis import EdgePreprocessorNikosis
from .laplacian_nikosis import LaplacianPreprocessorNikosis


NODE_CLASS_MAPPINGS = {
    "DepthAnythingV2Nikosis": DepthAnythingV2Nikosis,
    "LineArtPreprocessorNikosis": LineArtPreprocessorNikosis,
    "LineArtSketchPreprocessorNikosis": LineArtSketchPreprocessorNikosis,
    "EdgePreprocessorNikosis": EdgePreprocessorNikosis,
    "LaplacianPreprocessorNikosis": LaplacianPreprocessorNikosis,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthAnythingV2Nikosis": "Depth Anything V2 (nikosis)",
    "LineArtPreprocessorNikosis": "LineArt Preprocessor (nikosis)",
    "LineArtSketchPreprocessorNikosis": "LineArt Sketch Preprocessor (nikosis)",
    "EdgePreprocessorNikosis": "Edge Preprocessor (nikosis)",
    "LaplacianPreprocessorNikosis": "Laplacian Preprocessor (nikosis)",
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
