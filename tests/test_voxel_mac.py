"""Numerical regression checks for the operators replacing CUDA."""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ['SPARSE_CONV_BACKEND'] = 'none'
os.environ['SPARSE_ATTN_BACKEND'] = 'sdpa'
import unittest
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from trellis2.modules.sparse import SparseTensor, VarLenTensor, SparseConv3d
from trellis2.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention
from trellis2.modules.sparse.attention.rope import SparseRotaryPositionEmbedder
from trellis_vox_runtime import TrellisVoxRuntime
import vox_io


class PortableOperators(unittest.TestCase):
    def test_sparse_conv_matches_dense(self):
        torch.manual_seed(4)
        for device in ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []):
            for dilation in [1, 2]:
                coords = torch.nonzero(torch.rand(2, 5, 6, 7) > 0.5).int()
                feats = torch.randn(len(coords), 3)
                conv = SparseConv3d(3, 4, (3, 1, 3), dilation=dilation).eval()
                dense = torch.zeros(2, 3, 5, 6, 7)
                b, x, y, z = coords.long().T
                dense[b, :, x, y, z] = feats
                expected = F.conv3d(dense, conv.weight.permute(0, 4, 1, 2, 3), conv.bias,
                                    padding=(dilation, 0, dilation), dilation=dilation)[b, :, x, y, z]
                conv.to(device)
                sparse = SparseTensor(feats.to(device), coords.to(device))
                with torch.no_grad():
                    actual = conv(sparse).feats.cpu()
                    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                    torch.testing.assert_close(conv(sparse).feats.cpu(), expected, rtol=1e-4, atol=1e-5)
                    if device == 'mps':
                        conv.half()
                        half_result = conv(sparse.half()).feats.float().cpu()
                        torch.testing.assert_close(half_result, expected, rtol=0.02, atol=0.003)

    def test_ragged_attention(self):
        for device in ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []):
            q = torch.randn(7, 2, 8, device=device)
            k = torch.randn(9, 2, 8, device=device)
            v = torch.randn(9, 2, 6, device=device)
            qs = VarLenTensor(q, [slice(0, 2), slice(2, 7)])
            ks = VarLenTensor(k, [slice(0, 6), slice(6, 9)])
            vs = VarLenTensor(v, [slice(0, 6), slice(6, 9)])
            got = sparse_scaled_dot_product_attention(qs, ks, vs).feats
            ref = torch.cat([F.scaled_dot_product_attention(q[a].transpose(0, 1), k[b].transpose(0, 1),
                                                             v[b].transpose(0, 1)).transpose(0, 1)
                             for a, b in zip(qs.layout, ks.layout)])
            torch.testing.assert_close(got, ref)

    def test_rotary_mps_matches_cpu(self):
        if not torch.backends.mps.is_available():
            self.skipTest('MPS unavailable')
        coords = torch.tensor([[0, 0, 2, 3], [0, 5, 2, 4]], dtype=torch.int32)
        feats = torch.randn(2, 2, 64)
        rope = SparseRotaryPositionEmbedder(64)
        expected = rope(SparseTensor(feats, coords)).feats
        actual = rope(SparseTensor(feats.to('mps'), coords.to('mps'))).feats.cpu()
        torch.testing.assert_close(actual, expected)

    def test_vox_orientation_roundtrip(self):
        # Non-cubic fixture catches swapped metadata and preview dimensions.
        grid = vox_io.VoxelGrid(np.zeros((5, 4, 3), dtype=np.uint8))
        grid.data[4, 3, 2] = 1
        grid = vox_io.swap_yz(grid)
        palette = np.array([[220, 10, 30]], dtype=np.uint8)
        decoded = vox_io.decode(vox_io.encode(grid, palette=palette, use_zstd=True))
        self.assertEqual((decoded.size_x, decoded.size_y, decoded.size_z), (3, 5, 4))
        np.testing.assert_array_equal(decoded.data, grid.data)
        from PIL import Image
        import io
        preview = Image.open(io.BytesIO(TrellisVoxRuntime._preview_png(grid, palette)))
        self.assertEqual(preview.size, (3, 5))

    def test_cli_roundtrip_uses_file_axes(self):
        import tempfile
        from pathlib import Path
        from img_to_vox import _convert_one
        from trellis_vox_runtime import ConvertResult
        grid = vox_io.VoxelGrid.empty(3, 5, 4)
        grid.data[3, 4, 2] = 1
        result = ConvertResult(vox_io.encode(grid), 3, 5, 4, 1, (3, 5, 4), 1, 0.0)
        runtime = SimpleNamespace(convert=lambda *args, **kwargs: result)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(_convert_one(runtime, Path('unused.png'), Path(tmp) / 'test.vox',
                                         pipeline_type='512', seed=0, material_mode='solid',
                                         max_height=128, max_colors=220))


if __name__ == '__main__':
    unittest.main()
