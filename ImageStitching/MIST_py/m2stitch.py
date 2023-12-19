


from Images import Images
import numpy as np
import cv2
import m2stitch
import pandas as pd
from os import path
import time

class MIST(Images):

    
    def __init__(self, fileFolder: str, outputFolder: str, filenameFormat: str) -> None:
        super().__init__(fileFolder, outputFolder, filenameFormat)



    def stackImages(self, folder):
        out = np.zeros_like((len(folder), self.shape[0], self.shape[1]))
        # print("Come to stack")
        # print(np.max(cv2.imread(folder[0])))
        out = np.stack([cv2.imread(file, flags=cv2.IMREAD_GRAYSCALE) for file in folder])
        return out


    def stitch(self):

        start = time.time()
        
        script_path = path.dirname(path.realpath(__file__))

        images = self.stackImages([self.fileFolder + "\\" + image for image in self.imgs.keys()])
        rows = [c[0] for c in self.imgs.values()]
        cols = [c[1] for c in self.imgs.values()]

        # image_file_path = path.join(script_path, "../tests/data/testimages.npy")
        # props_file_path = path.join(script_path, "../tests/data/testimages_props.csv")
        # images = np.load(image_file_path)
        # props = pd.read_csv(props_file_path, index_col=0)

        # rows = props["row"].to_list()
        # cols = props["col"].to_list()

        print(images.shape)
        # must be 3-dim, with each dimension meaning (tile_index,x,y)
        print(rows)
        # the row (second-last dim.) indices for each tile index. for example, [1,1,2,2,2,...]
        print(cols)
        # the column (last dim.) indices for each tile index. for example, [2,3,1,2,3,...]

        # Note : the row_col_transpose=True is kept only for the sake of version compatibility.
        # In the mejor version, the row_col_transpose=False will be the default.
        result_df, _ = m2stitch.stitch_images(images, rows, cols, row_col_transpose=False, ncc_threshold=0.3)

        print(result_df["y_pos"])
        # the absolute y (second last dim.) positions of the tiles
        print(result_df["x_pos"])
        # the absolute x (last dim.) positions of the tiles

        # stitching example
        result_df["y_pos2"] = result_df["y_pos"] - result_df["y_pos"].min()
        result_df["x_pos2"] = result_df["x_pos"] - result_df["x_pos"].min()

        size_y = images.shape[1]
        size_x = images.shape[2]

        stitched_image_size = (
            result_df["y_pos2"].max() + size_y,
            result_df["x_pos2"].max() + size_x,
        )
        stitched_image = np.zeros_like(images, shape=stitched_image_size)
        for i, row in result_df.iterrows():
            stitched_image[
                row["y_pos2"] : row["y_pos2"] + size_y,
                row["x_pos2"] : row["x_pos2"] + size_x,
            ] = images[i]


        end = time.time()
        self.timing["stitch"] = end - start
        return stitched_image

        # result_image_file_path = path.join(script_path, "stitched_image.npy")
        # np.save(result_image_file_path, stitched_image)

