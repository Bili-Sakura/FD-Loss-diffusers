"""Smoke tests for the Diffusers-style package layout."""

import unittest


class TestFdlossDiffusersImports(unittest.TestCase):
    def test_registry_imports(self):
        import fdloss_diffusers as fd

        self.assertIn("JiT_H", fd.JiTDenoiser_models)
        self.assertIn("iMF_XL", fd.iMFDenoiser_models)
        self.assertIn("pMF_B", fd.pMFDenoiser_models)
        self.assertIn("sdvae", fd.VAE_models)

    def test_transformers_subpackage(self):
        from fdloss_diffusers.models.transformers import JiT_models, MiT_models

        self.assertIn("JiT-B", JiT_models)
        self.assertIn("MiT_B", MiT_models)


if __name__ == "__main__":
    unittest.main()
