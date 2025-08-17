import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_model import test_model_r2

# ===== Ejecutar pruebas principales =====
if __name__ == "__main__":
    print("▶️ Ejecutando evaluación de modelos guardados...")
    test_model_r2()
