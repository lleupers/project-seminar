import components.kNN as knn
from datetime import datetime

# Generates unique run ID based on the current timestamp.
def make_run_id(fmt: str = "%Y-%m-%d_%H-%M-%S") -> str:
    return datetime.now().strftime(fmt)

# 1. Data preparation

# 2. Pose estimation

# 3. Autoencoding

# 4. kNN classification
train_latent_vector_path = ...
test_latent_vector_path = ...

X, y_train, ids_train = knn.load_latents(train_latent_vector_path)
X_test, y_test, ids_test = knn.load_latents(test_latent_vector_path)
model = knn.train_knn(X, y_train, k=3)
knn.evaluate(model, X_test, y_test, ids_test, y_train, ids_train, out_dir="results", top_n=3)