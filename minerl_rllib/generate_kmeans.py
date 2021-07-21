import argparse
import os

import minerl
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--env", default=None)
parser.add_argument("--num-actions", type=int, default=32)
parser.add_argument("--data-dir", default=os.getenv("MINERL_DATA_ROOT", "data"))
parser.add_argument("--overwrite", action="store_true")


def main():
    args = parser.parse_args()
    if args.env is None:
        env_list = []
        for env_name in os.listdir(args.data_dir):
            if "VectorObf" in env_name:
                env_list.append(env_name)
    else:
        env_list = [args.env]

    for env_name in env_list:
        print(f"Generating {args.num_actions}-means for {env_name}")

        file_dir = os.path.join(args.data_dir, f"{args.num_actions}-means")
        file = os.path.join(file_dir, env_name + ".npy")
        if os.path.exists(file) and not args.overwrite:
            print(f"k-means file already exists at {file}")
            continue
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)

        data = minerl.data.make(env_name)
        actions = []
        for trajectory_name in tqdm(list(data.get_trajectory_names())):
            try:
                for _, action, _, _, _ in data.load_data(trajectory_name):
                    actions.append(action["vector"])
            except TypeError:
                pass
        actions = np.stack(actions)
        print("computing k-means...")
        kmeans = KMeans(n_clusters=args.num_actions, verbose=1, random_state=0).fit(
            actions
        )
        print(kmeans)
        np.save(file, kmeans.cluster_centers_)


if __name__ == "__main__":
    main()
